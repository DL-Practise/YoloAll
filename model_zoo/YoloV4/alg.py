import sys
import os
import time
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import numpy as np
from common_utils import vis
from common_utils import AlgBase
import yaml
import cv2

class Alg(AlgBase):
    def __init__(self):
        self.cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg.yaml")
        self.cfg_info = {}
        self.model_name = None
        self.model = None
        self.device = "cpu"
        self.ignore_keys = []
        self.load_cfg()
        self.class_names = None
        
    def create_model(self, model_name="yolov4", dev="cpu"):
        if model_name not in self.cfg_info.keys():
            print('unknown model name:', model_name, 'create failed')
            return
        self.device = dev
        self.model_name = model_name

        cfgfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                     self.cfg_info[model_name]['normal']['cfgfile']) 
        weightfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                       self.cfg_info[model_name]['normal']['weight'])  
        namesfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                      self.cfg_info[model_name]['normal']['namesfile']) 
                                      
        self.class_names = load_class_names(namesfile)
            
        # create model
        self.model = Darknet(cfgfile)
        self.model.print_network()
        
        # Load pretrain
        if not os.path.exists(weightfile):
            return 'error: weight is not download, please download it from: %s'%self.cfg_info[model_name]['normal']['url']
        self.model.load_weights(weightfile)
        self.model.eval()

        if self.device == 'cuda':
            self.model.cuda()

        return None
    
    def inference(self, img_array):
        map_result = {'type':'img'}
        img_resize = cv2.resize(img_array,  tuple(self.cfg_info[self.model_name]['normal']['infer_size']))
        
        sized = cv2.resize(img_array, (self.model.width, self.model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        use_cuda = True if self.device == 'cuda' else False
        boxes = do_detect(self.model, 
                          sized, 
                          float(self.cfg_info[self.model_name]['normal']['infer_conf']), 
                          float(self.cfg_info[self.model_name]['normal']['nms_thre']), 
                          use_cuda)
        finish = time.time()
        print(boxes)
        boxes = np.squeeze(np.array(boxes), axis=0)
        print(boxes)
        width = img_array.shape[1]
        height = img_array.shape[0]
        boxes[:, 0] = boxes[:, 0] * width
        boxes[:, 1] = boxes[:, 1] * height
        boxes[:, 2] = boxes[:, 2] * width
        boxes[:, 3] = boxes[:, 3] * height
        vis(img_array, boxes[:, 0:4], boxes[:, 5], boxes[:, 6], conf=float(self.cfg_info[self.model_name]['normal']['infer_conf']))
        map_result['result'] = img_array
        return map_result