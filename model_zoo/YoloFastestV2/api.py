import os
import cv2
import time
import argparse
import torch
from model import detector
from the_utils import utils
from common_utils import vis

model = None
cfg = None
device = 'cpu'
conf_thres = 0.3
nms_thres=0.4


def get_support_models():
    model_list=['yolo_fastest_v2']
    return model_list


def create_model(model_name='yolo_fastest_v2', dev='cpu'):
    global model
    global device
    global cfg
    model = None
    device = dev

    model_cfg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/coco.data')
    model_weight = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modelzoo/coco2017-0.241078ap-model.pth')
    cfg = utils.load_datafile(model_cfg)
    model = detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.eval()

def inference(img_array):
    global model
    global device
    global cfg
    global conf_thres
    global nms_thres

    map_result = {'type':'img'}
   
     #数据预处理
    res_img = cv2.resize(img_array, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    img = img.to(device).float() / 255.0

    with torch.no_grad():
        preds = model(img)
        output = utils.handel_preds(preds, cfg, device)
        output_boxes = utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

    #加载label names
    LABEL_NAMES = []
    cfg["names"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/coco.names')
    with open(cfg["names"], 'r') as f:
	    for line in f.readlines():
	        LABEL_NAMES.append(line.strip())
    
    h, w, _ = img_array.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    #绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()
       
        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img_array, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        cv2.putText(img_array, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    map_result['result'] = img_array
    return map_result

