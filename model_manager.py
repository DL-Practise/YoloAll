# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import copy
import xml.etree.cElementTree as et
import os
import cv2
import math
from PIL import Image
import importlib
import torch
from common_utils import get_api_from_model

# ui配置文件
cUi, cBase = uic.loadUiType("model_manager.ui")

# 主界面
class ModelManager(QWidget, cUi):
    def __init__(self, alg_change_cb): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)

        self.alg_change_cb = alg_change_cb
        self.search_alg_path()
        self.dev = 'cpu'
        
        
    def search_alg_path(self):
        for i in range(self.alg_layout.count()):
            self.alg_layout.itemAt(i).widget().deleteLater() 
    
        self.alg_map = {}
        for sub_dir in os.listdir('./model_zoo'):
            sub_path = os.path.join('./model_zoo', sub_dir)
            if os.path.isdir(sub_path):
                self.alg_layout.addWidget(QLabel(str(sub_dir)))
                #api = importlib.import_module('model_zoo.' + str(sub_dir) + '.api')
                api = get_api_from_model(str(sub_dir))
                for model_name in api.get_support_models():
                        self.alg_map[model_name] = str(sub_dir)
                        btn = QRadioButton(model_name)
                        btn.toggled.connect(self.slot_alg_select)
                        self.alg_layout.addWidget(btn)
                self.alg_layout.addWidget(QLabel(''))              
        
    def get_select_alg(self):
        check_name = None
        for i in range(self.alg_layout.count()):
            if type(self.alg_layout.itemAt(i).widget()) == QRadioButton:
                if self.alg_layout.itemAt(i).widget().isChecked():
                    check_name = self.alg_layout.itemAt(i).widget().text()
                    break
        if check_name is not None:
            return check_name, self.alg_map[check_name], self.dev
        else:
            return None, None, None
            
            
    @pyqtSlot()
    def on_btnDev_clicked(self):
        print('on_btnDev_clicked')
        if self.dev == 'cpu':
            print('check cuda is available')
            if torch.cuda.is_available():
                self.dev = 'cuda'
                self.btnDev.setText('CUDA')
            else:
                print('cuda is not available')
        else:
            self.dev = 'cpu'
            self.btnDev.setText('CPU')

        alg_name, alg_path, device = self.get_select_alg()
        if alg_name is not None:
            self.alg_change_cb(alg_name, alg_path, device)
    
    def slot_alg_select(self, isChecked):
        if isChecked:
            alg_name, alg_path, device = self.get_select_alg()
            self.alg_change_cb(alg_name, alg_path, device)


if __name__ == "__main__":
    def alg_change_cb(a,b):
        print(a,b)

    cApp = QApplication(sys.argv)
    cModelManager = ModelManager(alg_change_cb)
    cModelManager.show()
    sys.exit(cApp.exec_())
    
    
    
    
    