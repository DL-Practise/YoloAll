import cv2
import numpy as np
import ctypes  
from ctypes import *
import matplotlib.pyplot as plt
import os
import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue


class AlgProcess(threading.Thread):
    def __init__(self, lib_path):
        super(AlgProcess, self).__init__()
        self.lib = CDLL(lib_path)
        self.img_queue = queue.Queue(maxsize=2)
        self.flag = True
        self.alg_threds = 8
        self.cb_func = None
        self.class_map = {}
        self.start()
        
    def load_model(self, model_path):
        self.class_map = {}
        for file in os.listdir(model_path):
            file_name = str(file)
            if file_name.endswith('.cfg'):
                cfg_file = bytes(model_path + '/' + file_name, "gbk")
            if file_name.endswith('.param.bin') or file_name.endswith('.param'):
                prasm_file = bytes(model_path + '/' + file_name, "gbk")
            if file_name.endswith('.bin') and not file_name.endswith('.param.bin'):
                bin_file = bytes(model_path + '/' + file_name,  "gbk")
            if file_name.endswith('class_map.txt'):
                with open(model_path + '/' + file_name, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if len(line) > 0:
                            cls = int(line.split(':')[0])
                            name = line.split(':')[1]
                            self.class_map[cls] = name

        self.lib.alg_init(cfg_file, prasm_file, bin_file)
        
    def set_alg_threds(num):
        self.alg_threds = num
        
    def set_alg_callback(self, func):
        self.cb_func = func
        
    def add_img(self, img):
        if self.img_queue.full():
            return
        else:
            self.img_queue.put(img)
            
    def run(self):
        while self.flag:
            try:
                img = self.img_queue.get(block=True, timeout=1)
                print('alg processing thread: get a img')
            except queue.Empty:
                img = None
                #print('alg processing thread: get no img')
            
            if img is not None:
                print('get img from img_queue')
                img_height, img_width = img.shape[:2]
                image_p = img.ctypes.data_as(ctypes.c_char_p)
                self.lib.alg_run.restype = ctypes.POINTER(ctypes.c_float)
                start_time = time.time()
                print('***', self.alg_threds)
                result = self.lib.alg_run(image_p, 11, img_height, img_width, self.alg_threds, 66)
                if self.cb_func is not None:
                    self.cb_func(img, result, self.class_map, time.time()-start_time)

            
    def quit(self):
        self.flag = False
        
        
if __name__ == "__main__":
    cAlg = AlgProcess("sdk/build_win64_vs2017/alg.dll")
    cAlg.load_model("model_zoo/yolov5s-coco-motorcycle")
    
    img = cv2.imread('./1.jpg')
    cAlg.add_img(img)
    
    input("press any key to quit")
    cAlg.quit()