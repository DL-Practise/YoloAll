import sys
import os
import time
from pathlib import Path
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import numpy as np
from common_utils import vis

model = None
device = 'cpu'
infer_conf = 0.25
nms_thre = 0.45
size=(640,640)
mean=[0.,0.,0.] # rbr
std=[255., 255., 255.] #bgr


def get_support_models():
    model_list=[]
    now_dir = os.path.dirname(os.path.realpath(__file__))
    for file in os.listdir(now_dir):
        if str(file).endswith('.pt') and 'yolo' in str(file):
            model_list.append(str(file).replace('.pt', ''))
    return model_list

def create_model(model_name='yolov5s', dev='cpu'):
    global model
    global device
    model = None
    device = dev

    # Load model
    pth_name = model_name + '.pt'
    pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s'%(pth_name))

    model = attempt_load(pre_train, map_location='cpu')  # load FP32 model
    model.eval()


    if device == 'cuda':
        model.cuda()

def inference(img_array):
    global model
    global device
    global size
    global mean
    global std
    global infer_conf
    global nms_thre


    map_result = {'type':'img'}
    img_resize = cv2.resize(img_array, size)
    img = (img_resize - mean) / std
    img = img.transpose((2,0,1))    
    img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

    if device == "cuda":
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, infer_conf, nms_thre, classes=None, agnostic=False)
    
    valid_pred = pred[0].cpu()

    boxes = valid_pred[:,0:4]
    cls = valid_pred[:, 5]
    scores = valid_pred[:, 4]

    x_rate = img_array.shape[1] /  size[0]
    y_rate = img_array.shape[0] /  size[1]
    boxes[:,0:4:2] = boxes[:,0:4:2] * x_rate
    boxes[:,1:4:2] = boxes[:,1:4:2] * y_rate

    vis(img_array, boxes, scores, cls, conf=0.5)
    map_result['result'] = img_array
    return map_result


if __name__ == '__main__':
    create_model(model_name='yolov5s', dev='cpu')
    image = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
    ret = inference(image)
    #print(ret)