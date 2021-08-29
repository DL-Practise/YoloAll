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


model = None
device = 'cpu'
num_classes = 80
infer_conf = 0.25
nms_thre = 0.45
infer_size=(640,640)
rgb_means = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_support_models():
    model_list=[]
    now_dir = os.path.dirname(os.path.realpath(__file__))
    for file in os.listdir(now_dir):
        if str(file).endswith('.pth.tar') and 'yolox' in str(file):
            model_list.append(str(file).replace('.pth.tar', ''))
    return model_list

def create_model(model_name='yolox_nano', dev='cpu'):
    global model
    global device
    global num_classes
    global infer_conf
    global nms_thre
    global infer_size
    model = None
    device = dev

    exp = get_exp(None, model_name.replace('_', '-'))
    exp.test_conf = infer_conf
    exp.nmsthre = nms_thre
    exp.test_size=infer_size
    cls_num = num_classes

    model = exp.get_model()
    model.eval()

    pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.pth.tar'%(model_name))
    ckpt = torch.load(pre_train, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if device == 'cuda':
        model.cuda()

def inference(img):
    global model
    global device
    global num_classes
    global infer_conf
    global nms_thre
    global infer_size
    global rgb_means
    global std

    map_result = {'type':'img'}
    img_info = {'id': 0}
    height, width = img.shape[:2]
    img_info['height'] = height
    img_info['width'] = width
    img_info['raw_img'] = img
    img, ratio = preproc(img, infer_size, rgb_means, std)
    img_info['ratio'] = ratio
    img = torch.from_numpy(img).unsqueeze(0)
    if device == "cuda":
        img = img.cuda()

    with torch.no_grad():
        t0 = time.time()
        outputs = model(img)
        outputs = postprocess(outputs, num_classes, infer_conf, nms_thre)
    

    # draw det
    if outputs[0] is None:
        map_result['result'] = img
    else:
        outputs = outputs[0].cpu()
        bboxes = outputs[:, 0:4]
        bboxes /= img_info["ratio"]
        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]
        vis_res = vis(img_info["raw_img"], bboxes, scores, cls, infer_conf, COCO_CLASSES)

        map_result['result'] = vis_res
    
    return map_result
