#! /usr/bin/env python3

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


model = None
device = 'cpu'
img_size = 416
conf_thres =0.5
nms_thres=0.4


def get_support_models():
    model_list=[]
    now_dir = os.path.dirname(os.path.realpath(__file__))
    for file in os.listdir(now_dir):
        if str(file).endswith('.weights') and 'yolov3' in str(file):
            model_list.append(str(file).replace('.weights', ''))
    return model_list


def create_model(model_name='yolov3-tiny', dev='cpu'):
    global model
    global device
    model = None
    device = dev

    model_cfg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config/%s.cfg'%(model_name))
    model_weight = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.weights'%(model_name))
    model = load_model(model_cfg, device, model_weight)
    model.eval()

def inference(img_array):
    global model
    global device
    global img_size
    global conf_thres
    global nms_thres

    map_result = {'type':'img'}
    img_tensor = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (img_array, np.zeros((1, 5))))[0].unsqueeze(0)
    if device == "cuda":
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, img_array.shape[:2])
    
    valid_pred = detections.cpu()
    boxes = valid_pred[:,0:4]
    cls = valid_pred[:, 5]
    scores = valid_pred[:, 4]
    #x_rate = img_array.shape[1] /  size[0]
    #y_rate = img_array.shape[0] /  size[1]
    #boxes[:,0:4:2] = boxes[:,0:4:2] * x_rate
    #boxes[:,1:4:2] = boxes[:,1:4:2] * y_rate
    vis(img_array, boxes, scores, cls, conf=0.5)
    map_result['result'] = img_array
    return map_result

