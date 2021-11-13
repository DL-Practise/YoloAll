import argparse
import os
import time
import cv2
import torch
from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import matplotlib.pyplot as plt
from yolox.utils.visualize import vis
from common_utils import AlgBase
import yaml

class Alg(AlgBase):
    def __init__(self):
        self.cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg.yaml")
        self.cfg_info = {}
        self.model_name = None
        self.model = None
        self.device = "cpu"
        self.ignore_keys = []
        self.load_cfg()
        
    def create_model(self, model_name="yolox_nano", dev="cpu"):
        if model_name not in self.cfg_info.keys():
            print('unknown model name:', model_name, 'create failed')
            return
        self.device = dev
        self.model_name = model_name
        exp = get_exp(None, model_name.replace("_", "-"))
        exp.test_conf = float(self.cfg_info[model_name]['infer_conf'])
        exp.nmsthre = float(self.cfg_info[model_name]['nms_thre'])
        exp.test_size = self.cfg_info[model_name]['infer_size']
        cls_num = int(self.cfg_info[model_name]['num_classes'])

        self.model = exp.get_model()
        self.model.eval()

        pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.cfg_info[model_name]['weight'])
        if not os.path.exists(pre_train):
            return 'error: weight is not download, please download it from: %s'%self.cfg_info[model_name]['url']

        ckpt = torch.load(pre_train, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        
        if self.device == "cuda":
            self.model.cuda()
            
        return None
    
    def inference(self, img):
        map_result = {"type":"img"}
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img, ratio = preproc(img, 
                             self.cfg_info[self.model_name]['infer_size'], 
                             self.cfg_info[self.model_name]['rgb_means'], 
                             self.cfg_info[self.model_name]['std'])
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "cuda":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(outputs, 
                                  int(self.cfg_info[self.model_name]['num_classes']), 
                                  float(self.cfg_info[self.model_name]['infer_conf']), 
                                  float(self.cfg_info[self.model_name]['nms_thre']))
        
        # draw det
        if outputs[0] is None:
            map_result["result"] = img
        else:
            outputs = outputs[0].cpu()
            bboxes = outputs[:, 0:4]
            bboxes /= img_info["ratio"]
            cls = outputs[:, 6]
            scores = outputs[:, 4] * outputs[:, 5]
            if len(self.cfg_info[self.model_name]['class_names']) == 0:
                cls_names = COCO_CLASSES
            else:
                cls_names = self.cfg_info[self.model_name]['class_names']
            vis_res = vis(img_info["raw_img"], bboxes, scores, cls, float(self.cfg_info[self.model_name]['infer_conf']), cls_names)
            map_result["result"] = vis_res
        
        return map_result

