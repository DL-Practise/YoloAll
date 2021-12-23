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
from PyQt5.QtWebEngineWidgets import *
from image_widget import ImageWidget
from common_utils import get_api_from_model
import threading
import qdarkstyle
import json
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
        
        # init title
        self.setWindowTitle('YoloAll V2.0.1')

        # init imagewidget
        self.cImageWidget = ImageWidget()
        self.cImageWidget.set_alg_handle(self)
        self.tabWidget.insertTab(0, self.cImageWidget, "预测")
        self.tabWidget.setTabIcon(0, QIcon(QPixmap("./icons/no_news.png")))
        
        # init config widget
        self.btnSaveCfg.hide()
        self.tabWidget.setTabIcon(1, QIcon(QPixmap("./icons/no_news.png")))

        # init help widget
        self.has_news = False
        with open('./news_id.json', 'r') as f:
            self.news_id = json.load(f)
        self.cBrowser = QWebEngineView()
        webEngineSettings = self.cBrowser.settings()
        webEngineSettings.setAttribute(QWebEngineSettings.LocalStorageEnabled, False)
        engineProfile = self.cBrowser.page().profile()
        engineProfile.clearHttpCache()
        cookie = engineProfile.cookieStore()
        cookie.deleteAllCookies()
        self.cBrowser.load(QUrl('http://www.lgddx.cn/projects/yolo_all/news/index.htm'))
        self.tabWidget.insertTab(2, self.cBrowser, "帮助")
        self.tabWidget.setTabIcon(2, QIcon(QPixmap("./icons/no_news.png")))

        # show imagewidget
        self.tabWidget.setCurrentIndex(0)
    
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
        self.update_model_flag = False
        self.create_model_process = 0
        self.create_process_dialog = None

    def slot_log_info(self, info):
        if str(info).startswith('cmd:'):
            if 'load models finished' in str(info):
                self.init_model_tree()
            if 'start create model' in str(info):
                self.tabWidget.setCurrentIndex(0)
                self.cImageWidget.change_background('start_load')               
            if 'create model failed' in str(info):
                self.cImageWidget.change_background('load_fail')
            if 'create model success' in str(info):
                self.cImageWidget.change_background('load_success')
            if 'pretrain unget' in str(info):
                box_message = str(info).split('=')[-1]
                box = QMessageBox()
                box.setIcon(QMessageBox.Critical)
                box.setTextInteractionFlags(Qt.TextSelectableByMouse)
                box.setWindowTitle(u"预训练模型未下载")
                box.setText(box_message)
                box.setTextInteractionFlags(Qt.TextSelectableByMouse)
                box.exec()
            if 'update title' in str(info):
                title_name = str(info).split('=')[-1]
                self.setWindowTitle(title_name)
        elif str(info).startswith('news_id'):
            self.tabWidget.setTabIcon(2, QIcon(QPixmap("./icons/news.png")))
        else:
            self.logEdit.append('<font color="#FF9090">%s</font>'%(info))    
                
    def check_news(self, x):
        lines = x.split('\n')
        for line in lines:
            if 'news_id' in line:
                id = int(line.split(':')[-1])
                if id != self.news_id['news_id']:
                    self.news_id['news_id'] = id
                    self.has_news = True
                    with open('./news_id.json', 'w') as f:
                        json.dump(self.news_id, f)
                    self.log_sig.emit('news_id')
                    break
                
    def det_thread_func(self):
        self.log_sig.emit('检测线程启动')
        
        # search all algs
        self.search_alg_and_model()
        
        # check news_id 
        self.cBrowser.page().toPlainText(self.check_news)
        
        while self.det_thread_flag:
            if self.update_model_flag:
                self.updaet_model()
                self.update_model_flag = False
            try:
                img = self.det_thread_queue.get(block=True, timeout=0.2)
                #self.log_sig.emit('det thread get a img')
            except queue.Empty:
                img = None
                #self.log_sig.emit('det thread get waiting for img')
            if img is not None and self.alg is not None:     
                start_time = time.time()
                ret = self.alg.inference(img)
                if self.cImageWidget is not None:
                    time_spend = time.time()-start_time
                    if 'result' not in self.model_cfg.keys():
                        save_result = 0
                        save_path = None
                    else:
                        save_result = int(self.model_cfg['result']['save_result'])
                        save_path = self.model_cfg['result']['save_dir']
                    self.cImageWidget.slot_alg_result(img, ret, time_spend, save_result, save_path)
        
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
        self.log_sig.emit('cmd:load models finished')

    def init_model_tree(self):
        for alg in self.alg_model_map.keys():
            item_alg = QTreeWidgetItem(self.treeModel)
            #item_alg.setFlags(Qt.ItemIsEnabled)
            item_alg.setText(0, alg)
            for model in self.alg_model_map[alg]:
                item_model = QTreeWidgetItem(item_alg)
                item_model.setText(0, model)
                    
    def updaet_model(self):
        self.log_sig.emit('cmd:start create model')
        self.log_sig.emit('开始创建模型: %s'%str(self.model_name))
        self.log_sig.emit('  停止ImageWidget')
        self.cImageWidget.stop_all()
        title_name = 'YoloAll V2.0.0 当前模型:' + self.model_name

        pretrain_path = './model_zoo/' + self.alg_name + '/' + self.model_cfg['normal']['weight']
        if not os.path.exists(pretrain_path):
            self.log_sig.emit('  创建模型: %s 失败，预训练模型未下载'%str(self.model_name))
            box_info = u'请到如下地址下载预训练模型\n放到 model_zoo/%s 下面\n下载地址：\n%s'%(self.alg_name, self.model_cfg['normal']['url'])
            self.log_sig.emit('cmd:pretrain unget=%s'%box_info)
            self.alg = None
            return
        if self.alg is not None:
            device = 'cuda' if self.model_cfg['device']['dev_type'] == 'gpu' else 'cpu'
            title_name += ' 设备类型:' + device
            self.log_sig.emit('  设备类型:' + device)
            self.alg.create_model(self.model_name, device)
            self.log_sig.emit('cmd:create model success')
            self.log_sig.emit('  创建模型: %s 结束'%str(self.model_name))
        else:
            self.log_sig.emit('cmd:create model failed')
            self.log_sig.emit('  创建模型: %s 失败，算法句柄尚未创建'%str(self.model_name))
            self.alg = None
        self.log_sig.emit('cmd:update title=%s'%(title_name))

    def _translate_str(self, ori_str):
        translate_map = {'device': '设备配置',
                         'dev_type': '设备类型(cpu/gpu)',
                         'result': '检测结果配置',
                         'save_result': '是否保存结果',
                         'save_dir': '保存路径',
                         'normal': '通用配置',
                         }
        if ori_str in translate_map.keys():
            return translate_map[ori_str]
        else:
            return ori_str

    def _init_cfg_widget(self):
        old_items = []
        for i in range(self.cfg_layout.count()):
            old_items.append(self.cfg_layout.itemAt(i))
            
        for old_item in old_items:
            self.cfg_layout.removeItem(old_item) 
 
        self.model_cfg_widget = {}
        if self.alg is not None:
            self.btnSaveCfg.show()
            self.model_cfg = self.alg.get_model_cfg(self.model_name)
            for key in self.model_cfg.keys():
                label_title = QLabel()
                label_title.setText('<font color="#FF9090">%s</font>'%(self._translate_str(key)))
                self.cfg_layout.addWidget(label_title)
                self.model_cfg_widget[key] = {}
                for sub_key in self.model_cfg[key]:
                    frame = QFrame()
                    edit_layout = QHBoxLayout()
                    edit_key = QLineEdit()
                    edit_value = QLineEdit()
                    edit_key.setText(self._translate_str(sub_key))
                    edit_key.setReadOnly(False)
                    edit_key.setFocusPolicy(Qt.NoFocus)
                    edit_value.setText(str(self.model_cfg[key][sub_key]))
                    edit_layout.addWidget(edit_key)
                    edit_layout.addWidget(edit_value)
                    edit_layout.setStretch(0, 1)
                    edit_layout.setStretch(1, 2)
                    self.cfg_layout.addLayout(edit_layout)
                    self.model_cfg_widget[key][sub_key] = edit_value
                label_space = QLabel()
                self.cfg_layout.addWidget(label_space)
            
            spacer = QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.cfg_layout.addItem(spacer)
            
    def _get_cfg_widget(self):
        for key in self.model_cfg_widget.keys():
            for sub_key in self.model_cfg_widget[key].keys():
                edit_widget = self.model_cfg_widget[key][sub_key]
                old_cfg_value = self.model_cfg[key][sub_key]
                new_cfg_value = edit_widget.text()
                if sub_key == 'dev_type':
                    if new_cfg_value != 'cpu':
                        if not torch.cuda.is_available():
                            reply = QMessageBox.warning(self,
                                u'警告', 
                                u'当前pytorch不支持cuda, 将创建cpu模型', 
                                QMessageBox.Yes)
                            edit_widget.setText('cpu')
                            new_cfg_value = 'cpu'
                self.model_cfg[key][sub_key] = new_cfg_value

    def on_treeModel_itemClicked(self, item, seq):
        print(item.text(0), item.parent())
        if item.parent() is None:
            print('you select alg')
        else:
            print('yolo select model: ', item.parent().text(0), item.text(0))
            self.alg_name = item.parent().text(0)
            self.model_name = item.text(0)
            api = get_api_from_model(self.alg_name)
            if api is None:
                self.alg = None
                print('error, the api can not import')
            else:
                self.alg = api.Alg()
                self._init_cfg_widget()
                #self.updaet_model()
                self.update_model_flag = True
    
    @pyqtSlot()
    def on_btnSaveCfg_clicked(self):
        print('button btnSaveCfg clicked')
        self._get_cfg_widget()
        self.alg.put_model_cfg(self.model_name, self.model_cfg)
        #self.updaet_model()
        self.update_model_flag = True

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
