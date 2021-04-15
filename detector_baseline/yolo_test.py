#!/usr/bin/env python

from torchvision import transforms
from detector import Detector
from config import *
import matplotlib.pyplot as plt
import utils
import torch
import os
import cv2
import time
from feature_detector import Feature_detector


class Yolo_detector:

    def __init__(self,file_name,device_type):
        # set device and path arguemnt
        load_device = device_type
        dir = os.path.dirname(__file__)
        load_path = os.path.join(dir, file_name)

        # initialize the detector
        detector = Detector(device='cpu').to(load_device)
        self.model = utils.load_model(detector, load_path, load_device)
        self.model.eval()
        self.category_dict = CATEGORY_DICT
        self.threshold = CONF_THRESHOLD

    def publish_pose(self,img,bbs):

        ROI_list = []

        if bbs[0]:
            for box in bbs[0]:
                
                start_point = (box["x"], box["y"])
                text_point = (box["x"], box["y"] - 5)
                end_point = (box["x"] + box["width"], box["y"]+box["height"])
                sign_type = self.category_dict[box["category"]]["name"]

                cv_image = cv2.rectangle(img.copy(), start_point, end_point, (0,0,255),1)
                cv_image = cv2.putText(cv_image,sign_type,text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("image",cv_image)
                cv2.waitKey()         
        else:
            print("no detection found")


    def feedback(self,data):
        # The threshold above which a bounding box will be accepted
        img = data
        # adjust input dimension -> 3d to 4d
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        # get bounding boxes
        with torch.no_grad():
                torch.set_num_threads(4)
                out = self.model(image)
                bbs = self.model.decode_output(out, self.threshold)
        self.publish_pose(img,bbs)
        fet_detector = Feature_detector(img,bbs,exp=30)
        q_list, t_list = fet_detector.pose_estimation()

        for i in range(len(q_list)):
            print(q_list[i])
            print(t_list[i])

        return None


# relative path of training result
file = 'trained_model/det_2021-04-11_13-56-50-15class.pt'

device = 'cpu'
detector = Yolo_detector(file,device)

# uncomment when testing locally
dir = os.path.dirname(__file__)
img_path = os.path.join(dir,'test_images/b_001.jpg')
img = cv2.imread(img_path)

def main():
    detector.feedback(img)

if __name__ == "__main__":
    main()
    
