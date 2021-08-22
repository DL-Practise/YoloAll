import cv2
import numpy as np
import ctypes  
from ctypes import *
import matplotlib.pyplot as plt
import os
import time
import threading
import importlib
import sys
try:
    import queue
except ImportError:
    import Queue as queue
import torch
from common_utils import get_api_from_model


class AlgPytorch(threading.Thread):
    def __init__(self, cb_func):
        super(AlgPytorch, self).__init__()

        self.img_queue = queue.Queue(maxsize=2)
        self.thread_flag = True
        self.cb_func = cb_func
        self.alg_api = None        
        self.alg_dir = ''
        self.alg_name = ''
        self.alg_dev = ''
        self.start()


    def create_model(self, model_dir='shufflenetv2', model_name='shufflenet_v2_x1_0', device='cpu'):
        self.alg_api = None
        self.alg_dir = model_dir
        self.alg_name = model_name
        self.alg_dev = device


    def add_img(self, img):
        if self.img_queue.full():
            return
        else:
            self.img_queue.put(img)
            

    def run(self):
        while self.thread_flag:
            try:
                img = self.img_queue.get(block=True, timeout=1)
                print('alg processing thread: get a img')
            except queue.Empty:
                img = None
                print('alg processing thread: get no img')
            
            if img is not None and self.alg_api is not None:     
                start_time = time.time()
                ret = self.alg_api.inference(img)
                if self.cb_func is not None:
                    self.cb_func(img, ret, time.time()-start_time)

            if self.alg_api is None and self.alg_dir != '' and self.alg_name != '' and self.alg_dev != '':
                #self.alg_api = importlib.import_module('model_zoo.' + self.alg_dir + '.api')
                self.alg_api = get_api_from_model(self.alg_dir)
                self.alg_api.create_model(self.alg_name, self.alg_dev)
                ret = {'type': 'info', 'result':'alg create success'}
                self.cb_func(None, ret, None)


    def quit(self):
        self.thread_flag = False
        
        
if __name__ == "__main__":

    for dir in os.listdir('./model_zoo'):
        if os.path.isdir(os.path.join('./model_zoo',dir)):
            sub_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_zoo/'+str(dir))
            sys.path.append(sub_dir)

    def alg_cb(img, result, time):
        print('get alg callback: ', result['type'])

    cAlg = AlgPytorch(alg_cb)
    cAlg.create_model(model_dir='shufflenetv2', model_name='shufflenet_v2_x1_0', device='cuda')
    #cAlg.create_model(model_dir='higher_hrnet', model_name='higher_hrnet_w32_512', device='cuda')

    time.sleep(5)
    image = cv2.imread('imgs/shark.jpg', cv2.IMREAD_COLOR)
    cAlg.add_img(image)

    input("press any key to quit")
    cAlg.quit()
    
    