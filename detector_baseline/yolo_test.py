#!/usr/bin/env python

from torchvision import transforms
from detector import Detector
from config import *
import utils
import torch
import os
import cv2
import time


class yolo_detector:

    def __init__(self,file_name,device_type):
        # set device and path arguemnt
        load_device = device_type
        dir = os.path.dirname(__file__)
        load_path = os.path.join(dir, file_name)

        # initialize the detector
        detector = Detector().to(load_device)
        self.model = utils.load_model(detector, load_path, load_device)
        self.model.eval()
        self.category_dict = CATEGORY_DICT

    def publish_pose(self,img,bbs):

        cv_image = img
        ROI_list = []

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if bbs[0]:
            for box in bbs[0]:
                start_point = (box["x"], box["y"])
                text_point = (box["x"], box["y"] - 5)
                end_point = (box["x"] + box["height"], box["y"]+box["width"])
                sign_type = self.category_dict[box["category"]]["name"]

                cv2.rectangle(cv_image, start_point, end_point, (0,0,255),2)
                cv2.putText(img,sign_type,text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                ROI = img[int(start_point[1]):int(end_point[1]),int(start_point[0]):int(end_point[0]),:]
                ROI_list.append(ROI)
                cv2.imshow("image",cv_image)
                cv2.waitKey()

                kp1, res1 = orb.detectAndCompute(ROI, None)
        else:
            print("no detection found")

    def feedback(self,data):
        # The threshold above which a bounding box will be accepted
        threshold = 0.6
        img = data
        # adjust input dimension -> 3d to 4d
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        # get bounding boxes
        with torch.no_grad():
                torch.set_num_threads(4)
                out = self.model(image)
                bbs = self.model.decode_output(out, threshold)
        self.publish_pose(img,bbs)

        return None


def main():

    # relative path of training result
    file = 'trained_model/det_2021-03-01_py2.pt'

    device = 'cpu'
    detector = yolo_detector(file,device)

    # uncomment when testing locally
    dir = os.path.dirname(__file__)
    img_path = os.path.join(dir,'test_images/img_1.jpg')
    img = cv2.imread(img_path)
    detector.feedback(img)

if __name__ == "__main__":

    main()
    
