# -*- coding: utf-8 -*-
import sys
import os
import time
import torch
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from image_widget import ImageWidget
from common_utils import get_api_from_model
import threading
import qdarkstyle
try:
    import queue
except ImportError:
    import Queue as queue

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class MainWidget(QWidget, cUi):
    log_sig = pyqtSignal(str)
    
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        
        # init imagewidget
        self.cImageWidget = ImageWidget()
        self.cImageWidget.set_alg_handle(self)
        self.tabWidget.addTab(self.cImageWidget, "inference")
        
        # init treewidget
        self.treeModel.header().setVisible(False)
        
        # init log
        self.log_sig.connect(self.slot_log_info)
        
        self.alg = None
        self.alg_name = None
        self.model_name = None
        self.model_cfg = None
        self.model_cfg_widget = {}
        self.alg_model_map = {}
        self.det_thread_flag = True
        self.det_thread_queue = queue.Queue(maxsize=2)
        self.det_thread_handle = threading.Thread(target=self.det_thread_func, args=())
        self.det_thread_handle.start()

    def slot_log_info(self, info):
        if str(info).startswith('cmd:'):
            if 'load models finished' in str(info):
                self.init_model_tree()
        else:
            self.logEdit.append(info)        
                
    def det_thread_func(self):
        self.log_sig.emit('检测线程启动')
        self.search_alg_and_model()
        while self.det_thread_flag:
            try:
                img = self.det_thread_queue.get(block=True, timeout=1.0)
                #self.log_sig.emit('det thread get a img')
            except queue.Empty:
                img = None
                #self.log_sig.emit('det thread get waiting for img')
            if img is not None and self.alg is not None:     
                start_time = time.time()
                ret = self.alg.inference(img)
                if self.cImageWidget is not None:
                    self.cImageWidget.slot_alg_result(img, ret, time.time()-start_time)
        
    def add_img(self, img):
        if self.det_thread_queue.full():
            return
        else:
            self.det_thread_queue.put(img)
        
    def search_alg_and_model(self):
        self.alg_model_map = {}
        self.log_sig.emit('>开始加载模型，请等待所有模型加载成功')
        for sub_dir in os.listdir('./model_zoo'):
            self.log_sig.emit('>>正在加载模型: %s'%str(sub_dir))
            sub_path = os.path.join('./model_zoo', sub_dir)
            if os.path.isdir(sub_path):
                api = get_api_from_model(str(sub_dir))
                if api is not None:
                    self.alg = api.Alg()
                    self.alg_model_map[str(sub_dir)] = self.alg.get_support_models()
                    self.log_sig.emit('>>加载模型: %s 成功'%str(sub_dir))
                else:
                    self.alg_model_map[str(sub_dir)] = []
                    self.log_sig.emit('>>加载模型: %s 失败'%str(sub_dir))
        self.log_sig.emit('>加载模型结束')
        self.log_sig.emit('cmd: load models finished')

    def init_model_tree(self):
        for alg in self.alg_model_map.keys():
            item_alg = QTreeWidgetItem(self.treeModel)
            #item_alg.setFlags(Qt.ItemIsEnabled)
            item_alg.setText(0, alg)
            for model in self.alg_model_map[alg]:
                item_model = QTreeWidgetItem(item_alg)
                item_model.setText(0, model)
                    
    def updaet_model(self):
        self.cImageWidget.change_background('start_load')
        self.log_sig.emit('开始创建模型: %s'%str(self.model_name))
        self.log_sig.emit('  停止ImageWidget')
        self.cImageWidget.stop_all()

        pretrain_path = './model_zoo/' + self.alg_name + '/' + self.model_cfg['weight']
        if not os.path.exists(pretrain_path):
            self.log_sig.emit('  创建模型: %s 失败，预训练模型未下载'%str(self.model_name))
            self.cImageWidget.change_background('load_fail')
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setTextInteractionFlags(Qt.TextSelectableByMouse)
            box.setWindowTitle(u"预训练模型未下载")
            box.setText(u'请到如下地址下载预训练模型\n放到 model_zoo/%s 下面\n下载地址：\n%s'%(self.alg_name, self.model_cfg['url']))
            box.setTextInteractionFlags(Qt.TextSelectableByMouse)
            box.exec()
            '''
            reply = QMessageBox.warning(self,
                u'预训练模型未下载', 
                u'请到如下地址下载预训练模型\n放到 model_zoo/%s 下面\n下载地址：\n%s'%(self.alg_name, self.model_cfg['url']), 
                QMessageBox.Yes)
            '''
            self.alg = None
            
            return

        if self.alg is not None:
            device = 'cuda' if self.radioGpu.isChecked() else 'cpu'
            self.log_sig.emit('  设备类型:' + device)
            self.alg.create_model(self.model_name, device)
            self.log_sig.emit('  创建模型: %s 结束'%str(self.model_name))
            self.cImageWidget.change_background('load_success')
        else:
            self.log_sig.emit('  创建模型: %s 失败，算法句柄尚未创建'%str(self.model_name))
            self.cImageWidget.change_background('load_fail')
            self.alg = None
        
    def on_treeModel_itemClicked(self, item, seq):
        print(item.text(0), item.parent())
        if item.parent() is None:
            print('you select alg')
        else:
            print('yolo select model: ', item.parent().text(0), item.text(0))
            
            # clear the cfg edit
            for i in range(self.cfg_layout.count()):
                widget = self.cfg_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
            self.model_cfg_widget = {}
            
            self.alg_name = item.parent().text(0)
            self.model_name = item.text(0)
            api = get_api_from_model(self.alg_name)
            if api is None:
                self.alg = None
                print('error, the api can not import')
            else:
                self.alg = api.Alg()
                self.model_cfg = self.alg.get_model_cfg(self.model_name)
                group_box = QGroupBox()
                group_box.setTitle(self.model_name)
                group_layout = QVBoxLayout()
                for key in self.model_cfg.keys():
                    edit_layout = QHBoxLayout()
                    edit_key = QLineEdit()
                    edit_value = QLineEdit()
                    edit_key.setText(key)
                    edit_key.setReadOnly(False)
                    edit_key.setFocusPolicy(Qt.NoFocus)
                    edit_value.setText(str(self.model_cfg[key]))
                    edit_layout.addWidget(edit_key)
                    edit_layout.addWidget(edit_value)
                    edit_layout.setStretch(0, 1)
                    edit_layout.setStretch(1, 2)
                    group_layout.addLayout(edit_layout)
                    self.model_cfg_widget[key] = edit_value
                
                spacer = QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
                group_layout.addItem(spacer)
                group_box.setLayout(group_layout)  
                self.cfg_layout.addWidget(group_box)
                self.updaet_model()
    
    @pyqtSlot()
    def on_btnSaveCfg_clicked(self):
        print('button btnSaveCfg clicked')
        for key in self.model_cfg_widget.keys():
            edit_widget = self.model_cfg_widget[key]
            old_cfg_value = self.model_cfg[key]
            new_cfg_value = edit_widget.text()
            self.model_cfg[key] = new_cfg_value
        self.alg.put_model_cfg(self.model_name, self.model_cfg)
        self.updaet_model()

    def on_radioGpu_toggled(self):
        device = 'cuda' if self.radioGpu.isChecked() else 'cpu'
        if self.radioGpu.isChecked():
            if not torch.cuda.is_available():
                reply = QMessageBox.warning(self,
	                  u'警告', 
	                  u'cuda不可用，请检查', 
	                  QMessageBox.Yes)
                self.radioGpu.setChecked(False)


    def closeEvent(self, event):        
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.cImageWidget.stop_all()
            self.det_thread_flag = False
            self.det_thread_handle.join()
        else:
            event.ignore()

        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = MainWidget()
    cApp.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    cMainWidget.show()
    sys.exit(cApp.exec_())