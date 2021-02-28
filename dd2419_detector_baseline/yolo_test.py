#!/usr/bin/env python3

from torchvision import transforms
from detector import Detector
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

    # def publish_pose(self,img,bbs):
    #     T_x = 0.2
    #     f = 
    #     Z = f * T_x / box["width"]
    #     cv_image = img
    #     for box in bbs[0]:
    #         start_point = (box["x"], box["y"])
    #         end_point = (box["x"] + box["height"], box["y"]+box["width"])
    #         cv2.circle(cv_image,(320,240),10, (0,0,255),1)
    #         cv2.rectangle(cv_image, start_point, end_point, (0,0,255),1)
    #         cv2.imshow("image",cv_image)
    #         cv2.waitKey(0)
        
    #         print(box["x"])
    #         print(box["y"])
    #         print(box["height"])
    #         print(box["width"])


    def feedback(self,data):
        # The threshold above which a bounding box will be accepted
        threshold = 0.2
        img = data
        # adjust input dimension -> 3d to 4d
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        # get bounding boxes
        with torch.no_grad():
            for i in range(17):
                torch.set_num_threads(i+1)
                s_t = time.time()
                out = self.model(image)
                print(str(i) + " thread using time " + str(time.time()- s_t) +" seconds")
                bbs = self.model.decode_output(out, threshold)
        #self.publish_pose(img,bbs)

        return None


def main():

    # relative path of training result
    file = 'trained_model/det_2021-02-20_12-15-03-144981.pt'

    device = 'cpu'
    detector = yolo_detector(file,device)

    # uncomment when testing locally
    dir = os.path.dirname(__file__)
    img_path = os.path.join(dir,'test_images/img_1.jpg')
    img = cv2.imread(img_path)
    detector.feedback(img)

if __name__ == "__main__":

    main()
    
