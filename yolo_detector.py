#!/usr/bin/env python
from sensor_msgs.msg import Image
import tf2_geometry_msgs
import tf2_ros

from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError
from detector import Detector
import utils
import torch
import rospy
import os
import cv2

import matplotlib.image as mpimg # for testing only
import matplotlib.pyplot as plt

class yolo_detector:

    def __init__(self,file_name,device_type):
        # set device and path arguemnt
        load_device = device_type
        dir = os.path.dirname(__file__)
        load_path = os.path.join(dir, file_name)

        # initialize the detector
        detector = Detector().to(load_device)
        self.model = utils.load_model(detector, load_path, load_device)

        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.feedback)
        self.img_pub = rospy.Publisher("yolo_detector/result", Image, queue_size = 2)

    def cvadd_bounding_boxes(self, img, bbs):
        # Convert the image from OpenCV to ROS format

        cv_image = img

        for box in bbs[0]:
            start_point = (box["x"], box["y"])
            end_point = (box["x"] + box["height"], box["y"]+box["width"])
            cv2.rectangle(cv_image, start_point, end_point, (0,0,255),1)
        
        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def feedback(self,data):
        # The threshold above which a bounding box will be accepted
        threshold = 0.2

        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
          
        # adjust input dimension -> 3d to 4d
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        # get bounding boxes
        out = self.model(image)
        bbs = self.model.decode_output(out, threshold)

        self.cvadd_bounding_boxes(img,bbs)
        return None


def main():
    rate = rospy.Rate(10)
    
    # relative path of training result
    file = 'trained_model/det_2021-02-20_12-15-03-144981.pt'

    device = 'cpu'
    detector = yolo_detector(file, device)

    # uncomment when testing locally
    # dir = os.path.dirname(__file__)
    # img_path = os.path.join(dir,'test_images/img_5.jpg')
    # img = cv2.imread(img_path)
    
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('yolodetector', anonymous=True)
    main()
