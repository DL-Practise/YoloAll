import os
import cv2
import time
import argparse
import torch
from model import detector
from utils import utils
from common_utils import vis
from common_utils import AlgBase
import cv2

class Alg(AlgBase):
    def __init__(self):
        self.cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg.yaml")
        self.cfg_info = {}
        self.model_name = None
        self.model = None
        self.device = "cpu"
        self.ignore_keys = ['model_cfg', 'name_file']
        self.load_cfg()

    def create_model(self, model_name='yolov3_tiny', dev='cpu'):
        if model_name not in self.cfg_info.keys():
            print('error: unknown model name ', model_name, 'create failed')
            return
        self.device = dev
        self.model_name = model_name

        model_cfg = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.cfg_info[self.model_name]['normal']['model_cfg'])
        model_weight = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.cfg_info[self.model_name]['normal']['weight'])
        self.cfg = utils.load_datafile(model_cfg)
        self.model = detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(model_weight, map_location=self.device))
        self.model.eval()

        return None
    
    def inference(self, img_array):
        map_result = {'type':'img'}
   
        #数据预处理
        res_img = cv2.resize(img_array, (self.cfg["width"], self.cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0

        with torch.no_grad():
            preds = self.model(img)
            output = utils.handel_preds(preds, self.cfg, self.device)
            output_boxes = utils.non_max_suppression(output, 
                                                    conf_thres = float(self.cfg_info[self.model_name]['normal']['infer_conf']), 
                                                    iou_thres = float(self.cfg_info[self.model_name]['normal']['nms_thre']))

        #加载label names
        LABEL_NAMES = []
        self.cfg["names"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.cfg_info[self.model_name]['normal']['name_file'])
        with open(self.cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        h, w, _ = img_array.shape
        scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]

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
