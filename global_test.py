import sys
import importlib
import os
import cv2
from common_utils import get_api_from_model



if __name__ == '__main__':
    ALG_TEST = True
    TEST_MODEL = False

    # model test
    if TEST_MODEL:
        print('test yolov-fastest')
        api = get_api_from_model('YoloFastestV2')
        print(api.get_support_models())
        api.create_model()
        img = cv2.imread('imgs/people.jpg')
        ret = api.inference(img)
        
    # alg test
    if ALG_TEST:
        print('test YoloFastest::alg.py')
        api = get_api_from_model('YoloFastest')
        if api is not None:
            alg = api.Alg()
            alg.load_cfg()
            alg.save_cfg()
            print(alg.cfg_info)

        print('test YoloV3::alg.py')
        api = get_api_from_model('YoloV3')
        if api is not None:
            alg = api.Alg()
            alg.load_cfg()
            alg.save_cfg()
            print(alg.cfg_info)
        
        print('test YoloV5::alg.py')
        api = get_api_from_model('YoloV5')
        if api is not None:
            alg = api.Alg()
            alg.load_cfg()
            alg.save_cfg()
            print(alg.cfg_info)

        print('test YoloX::alg.py')
        api = get_api_from_model('YoloX')
        if api is not None:
            alg = api.Alg()
            alg.load_cfg()
            alg.save_cfg()
            print(alg.cfg_info)


