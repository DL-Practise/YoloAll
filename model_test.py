import sys
import importlib
import os
import cv2
from common_utils import get_api_from_model

if __name__ == '__main__':
    '''
    api = get_api_from_model('YoloV5')
    api.create_model()
    img = cv2.imread('imgs/people.jpg')
    ret = api.inference(img)
    print(api.get_support_models())
    print(type(ret))
    '''
    
    '''
    api = get_api_from_model('YoloX')
    api.create_model()
    img = cv2.imread('imgs/people.jpg')
    ret = api.inference(img)
    print(api.get_support_models())
    print(type(ret))
    '''

    api = get_api_from_model('YoloV3')
    print(api.get_support_models())
    api.create_model()
    img = cv2.imread('imgs/people.jpg')
    ret = api.inference(img)


