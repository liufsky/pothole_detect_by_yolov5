import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

class image_detect:
    def __init__(self,images = ['https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg']):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print(self.model)
        #此处位置更改，可以采用不同的模型
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp_with_pretrain/weights/best.pt', force_reload=True)
        self.images = images

    def __str__(self):
        return f"{self.model}"

    def detect_images(self):
        #打印出对应图片的结果。
        self.results = self.model(self.images)
        print(self.results)

    def show_images_results(self):
        self.results.show()

    def show_images_xyxy(self):
        self.results_xyxy = self.results.pandas().xyxy[0]
        print(self.results_xyxy)

m = image_detect()
m.detect_images()
m.show_images_results()
m.show_images_xyxy()
