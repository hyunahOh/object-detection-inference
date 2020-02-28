import numpy as np
import tensorflow as tf
import json
import os, sys
import cv2
import base64

from io import BytesIO
from keras import backend as K
from PIL import Image
from matplotlib import pyplot as plt
from source_modules.yolo import YOLO


CLASS_NAMES = ['pedestrian', 'rider']

def get_session(gpu_fraction=0.1):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

# model class
class YOLOv3:
    def __init__(self):
        self.model = YOLO()
        print("Succesfully loaded model")

    def object_detect(self, input_img_b64):
        '''
        Detect objects in an image.
        input: image given by user (class 'werkzeug.datastructures.FileStorage')
        return: BytesIO(), image with class bounding boxes

        Predict class with YOLO v3 model: image with bounding boxes is saved with name "result.jpg" (see yolo.detect_image)
        Send this image file to front-end
        '''

        image = Image.open(BytesIO(base64.b64decode(input_img_b64)))
        out_boxes, out_scores, class_ids = self.model.detect_image(image)

        class_names = [CLASS_NAMES[idx] for idx in class_ids]

        return out_boxes, out_scores, class_ids, class_names
