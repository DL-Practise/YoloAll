# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from image_widget import ImageWidget
from alg_pytorch import AlgPytorch 
from model_manager import ModelManager
from common_utils import get_api_from_model

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class MainWidget(QWidget, cUi):
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        
        # create show widget
        self.cImageWidget = ImageWidget()
        self.cAlg = AlgPytorch(self.alg_result_cb)
        self.cModelManager = ModelManager(self.alg_select_cb)
        self.cImageWidget.set_alg_handle(self.cAlg)        
        self.tabWidget.addTab(self.cImageWidget, "inference")
        self.gridLayout.addWidget(self.cModelManager)
        
        
    def alg_select_cb(self, model_name, model_path, device):
        print(model_name, model_path)
        self.cAlg.create_model(model_dir=model_path, model_name=model_name, device=device)
        self.logEdit.append('select alg: ' + model_name + ', device:' 
            + device + ', please wait for creating model!!!')

    def alg_result_cb(self, img, result, time_spend):
        if result['type'] == 'info':
            self.logEdit.append(result['result'])
        self.cImageWidget.slot_alg_result(img, result, time_spend)

        
    def closeEvent(self, event):        
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.cImageWidget.stop_all()
            self.cAlg.quit()
        else:
            event.ignore()

        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = MainWidget()
    cMainWidget.show()
    sys.exit(cApp.exec_())