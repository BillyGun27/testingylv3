"""YOLO v3 output
"""
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model
from model.yolo_model import YOLO

yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
with open(file) as f:
        class_names = f.readlines()
class_names = [c.strip() for c in class_names]

all_classes = class_names

img = cv2.imread("london.jpg")

pimage = cv2.resize(img, (416, 416),
                interpolation=cv2.INTER_CUBIC)
pimage = np.array(image, dtype='float32')
pimage /= 255.
pimage = np.expand_dims(image, axis=0)

print( yolo.predict(pimage, img.shape) )