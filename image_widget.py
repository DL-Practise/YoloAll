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
import time
from PIL import Image

# ui配置文件
cUi, cBase = uic.loadUiType("image_widget.ui")

# 主界面
class ImageWidget(QWidget, cUi):
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)

        self.comboBoxCamera.addItem('0')
        self.comboBoxCamera.addItem('1')
        self.comboBoxCamera.addItem('2')
        
        self.timer = QTimer()
        self.video_cap = None
        self.camera_cap = None
        
        self.qpixmap = None
        self.qpixmap_bg = None
        self.cAlg = None
        self.infer = None
        self.class_map  = None
        self.alg_time = None
        self.save_result = False
        
        self.color_list = [QColor(255,0,0),
                      QColor(0,255,0),
                      QColor(0,0,255),
                      QColor(0,255,255),
                      QColor(255,0,255),
                      QColor(8,46,84),
                      QColor(199,97,20),
                      QColor(255,227,132),
                      QColor(255,255,0),
                      QColor(128,138,135)]

        self.change_background('normal')

        
    @pyqtSlot()
    def on_btnPhoto_clicked(self):
        print('on_btnPhoto_clicked')
        img_path = QFileDialog.getOpenFileName(self,  "选取图片", "./", "Images (*.jpg);;Images (*.png)") 
        img_path = img_path[0]
        if img_path != '':
            self.slot_photo_frame(img_path)
    
    @pyqtSlot()
    def on_btnVideo_clicked(self):
        print('on_btnVideo_clicked')
        video_path = QFileDialog.getOpenFileName(self,  "选取视频", "./", "Videos (*.mp4);;Images (*.3gp)") 
        video_path = video_path[0]
        if video_path != '':
            self.video_cap = cv2.VideoCapture(video_path)
            self.timer.start()
            self.timer.setInterval(int(1000 / float(30.0)))
            self.timer.timeout.connect(self.slot_video_frame)
                    
    @pyqtSlot()    
    def on_btnCamera_clicked(self):
        print('on_btnCamera_clicked')
        if self.camera_cap is None:
            self.camera_cap = cv2.VideoCapture(int(0))
            self.timer.start()
            self.timer.setInterval(int(1000 / float(30.0)))
            self.timer.timeout.connect(self.slot_camera_frame)
        else:
            self.camera_cap.release()
            self.camera_cap = None
            self.timer.stop()
                    
    @pyqtSlot()    
    def on_btnStop_clicked(self):
        self.stop_all()
            
    def slot_photo_frame(self, photo_path):          
        img = cv2.imread(photo_path)        
        self.cAlg.add_img(img)
                       
    def slot_camera_frame(self):
        if self.camera_cap is not None:
            # get a frame
            ret, img = self.camera_cap.read()
            if ret is False:
                self.stop_all()
                return
            self.cAlg.add_img(img)
        
    def slot_video_frame(self):
        if self.video_cap is not None:
            ret, img = self.video_cap.read()
            if ret is False:
                self.stop_all()
                return 
            self.cAlg.add_img(img)
                      
    def slot_alg_result(self, img, result, time_spend, save_result, save_path):
        if result['type'] == 'info':
            print(result['result'])
            return
        elif result['type'] == 'img':
            img = result['result']
            self.infer = None
            # need save the result
            if save_result == 1:
                #txt_color = (0, 0, 255)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #h,w,c = img.shape
                #cv2.putText(img, "save result", (w-100, h-10), font, 0.4, txt_color, thickness=1)
                self.save_result = True
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if os.path.exists(save_path):
                    save_file = os.path.join(save_path, '%f.jpg'%time.time())
                    cv2.imwrite(save_file, img)
            else:
                self.save_result = False

        else:
            self.infer = result
                
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.qpixmap = QPixmap.fromImage(image) 
        self.alg_time = time_spend
        self.update()
    
    def stop_all(self):
        self.timer.stop()
        self.qpixmap = None
        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
            
    def set_alg_handle(self, handle):
        self.cAlg = handle
        
    def change_background(self, bg_name):
        self.qpixmap_bg = None
        bg_path = './icons/bg_' + bg_name + '.png'
        self.qpixmap_bg = QPixmap(bg_path)
        self.update()
        
    def draw_image(self, painter):
        pen = QPen()
        font = QFont("Microsoft YaHei")
        if self.qpixmap is not None:
            painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.qpixmap)
            pen.setColor(self.getColor(0))
            painter.setPen(pen)
            pointsize = font.pointSize()
            font.setPixelSize(pointsize*180/72)
            painter.setFont(font)
            painter.drawText(10, 30, 'time=%.4f seconds fps=%.4f' % (self.alg_time, 1 / self.alg_time))
            if self.save_result:
                painter.drawText(int(self.width() * 0.75), int(self.height() * 0.98), 'save result')
        else:
            if self.qpixmap_bg is not None:
                painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.qpixmap_bg)
            pen.setColor(QColor(0, 0, 0))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(0, 0, self.width(), self.height())
            
    def draw_infer(self, painter):
        if self.infer is None:
            return
        # class
        if self.infer['type'] == 'classify':
            self.draw_infer_class(painter)
        # det
        elif self.infer['type'] == 'detection':
            self.draw_infer_det(painter)
        # kp
        elif self.infer['type'] == 'keypoint':
            self.draw_infer_kp(painter)
        else:
            print('unknown info type')
            assert(False)
    
    def draw_infer_class(self, painter):
        font = QFont("宋体")
        pointsize = font.pointSize()
        font.setPixelSize(pointsize*90/72)
        painter.setFont(font)
        
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(QColor(0, 255, 0))
        painter.setPen(pen)

        top1 = self.infer['result'][0]
        name = self.infer['result'][1]
        score = self.infer['result'][2]
        painter.drawText(10, 50, 'top1=%s(%.4f)' % (name, score))
        
    def draw_infer_det(self, painter):
        pass
    
    def draw_infer_kp(self, painter):
        x_scale = self.width() / self.qpixmap.width()
        y_scale = self.height() / self.qpixmap.height()
        for kps in self.infer['result']:
            kps[:,0] = kps[:,0] * y_scale
            kps[:,1] = kps[:,1] * x_scale

            nose = kps[0]
            left_shoulder = kps[5]
            right_shoulder = kps[6]
            center_shoulder = (left_shoulder + right_shoulder) / 2
            right_shoulder = kps[6]
            left_elbow = kps[7]
            right_elbow = kps[8]
            left_wrist = kps[9]
            right_wrist = kps[10]
            left_hip = kps[11]
            right_hip = kps[12]
            center_hip = (left_hip + right_hip) / 2
            left_knee = kps[13]
            right_knee = kps[14]
            left_ankle = kps[15]
            right_ankle = kps[16]

            pen = QPen()
            pen.setColor(self.getColor(0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(nose[1], nose[0], center_shoulder[1], center_shoulder[0])
            pen.setColor(self.getColor(1))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(center_shoulder[1], center_shoulder[0], center_hip[1], center_hip[0])
            pen.setColor(self.getColor(2))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_shoulder[1], left_shoulder[0], right_shoulder[1], right_shoulder[0])
            pen.setColor(self.getColor(3))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_shoulder[1], left_shoulder[0], left_elbow[1], left_elbow[0])
            pen.setColor(self.getColor(4))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_elbow[1], left_elbow[0], left_wrist[1], left_wrist[0])
            pen.setColor(self.getColor(5))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(right_shoulder[1], right_shoulder[0], right_elbow[1], right_elbow[0])
            pen.setColor(self.getColor(6))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(right_elbow[1], right_elbow[0], right_wrist[1], right_wrist[0])
            pen.setColor(self.getColor(7))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_hip[1], left_hip[0], right_hip[1], right_hip[0])
            pen.setColor(self.getColor(8))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_hip[1], left_hip[0], left_knee[1], left_knee[0])
            pen.setColor(self.getColor(9))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(left_knee[1], left_knee[0], left_ankle[1], left_ankle[0])
            pen.setColor(self.getColor(10))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(right_hip[1], right_hip[0], right_knee[1], right_knee[0])
            pen.setColor(self.getColor(11))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(right_knee[1], right_knee[0], right_ankle[1], right_ankle[0])
            
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_image(painter)
        self.draw_infer(painter)
        
    def getColor(self, index):
        return self.color_list[index % len(self.color_list)]
        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cImageWidget = ImageWidget()
    cImageWidget.show()
    sys.exit(cApp.exec_())