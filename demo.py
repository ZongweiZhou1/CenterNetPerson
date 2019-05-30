#!/usr/bin/env python
import os
import cv2
import time
from tqdm import tqdm
from test.vis_utils import draw_bboxes
from test.detector import CenterNet as Detector

os.environ['CUDA_VISIBLE_DEVICES']='2'
detector = Detector("CenterNet-52", iter=10000)
t0 = time.time()
image_names = [img for img in os.listdir('data/demo') if img[-3:]=='jpg']
for i in tqdm(range(len(image_names))):
  image = cv2.imread('data/demo/{}'.format(image_names[i]))
  bboxes = detector(image)
  image  = draw_bboxes(image, bboxes)
  cv2.imwrite("tmp_squeeze/{}.jpg".format(str(i).zfill(6)), image)
  cv2.imshow('image', image)
  cv2.waitKey(10)

t1 = time.time()
print("speed: %f s"%((t1-t0)/100))
