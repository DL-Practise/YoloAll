from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
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
        
    def create_model(self, model_name='yolov3_tiny', dev='cpu'):
        if model_name not in self.cfg_info.keys():
            print('error: unknown model name ', model_name, 'create failed')
            return
        self.device = dev
        self.model_name = model_name

        model_cfg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config/%s'%(self.cfg_info[model_name]['normal']['cfg_file']))
        model_weight = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s'%(self.cfg_info[model_name]['normal']['weight']))
        self.model = load_model(model_cfg, self.device, model_weight)
        self.model.eval()

    
    def inference(self, img_array):
        map_result = {'type':'img'}
        img_tensor = transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(int(self.cfg_info[self.model_name]['normal']['infer_size']))])(
                (img_array, np.zeros((1, 5))))[0].unsqueeze(0)
        if self.device == "cuda":
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            detections = self.model(img_tensor)
            detections = non_max_suppression(detections, float(self.cfg_info[self.model_name]['normal']['infer_conf']), float(self.cfg_info[self.model_name]['normal']['nms_thre']))
            detections = rescale_boxes(detections[0], int(self.cfg_info[self.model_name]['normal']['infer_size']), img_array.shape[:2])
        
        valid_pred = detections.cpu()
        boxes = valid_pred[:,0:4]
        cls = valid_pred[:, 5]
        scores = valid_pred[:, 4]
        vis(img_array, boxes, scores, cls, conf=0.0)
        map_result['result'] = img_array
        return map_result
 