#!/usr/bin/env python
from sensor_msgs.msg import Image
from perception.msg import Sign, SignArray
from geometry_msgs.msg import TransformStamped, Vector3, PoseStamped
from config import *

import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler
import matplotlib as plt

from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError
from detector import Detector
from feature_detector import Feature_detector
import utils
import torch

import math
import os
import cv2
import sys

import time

PUB_POSE = True
EXP = 20 

class yolo_detector:

    def __init__(self,file_name,device_type):
        
        # set device and path arguemnt
        load_device = device_type
        dir = os.path.dirname(__file__)
        load_path = os.path.join(dir, file_name)

        # initialize model
        detector = Detector(device='cpu').to(load_device)
        self.model = utils.load_model(detector, load_path, load_device)
        self.model.eval()
        self.bridge = CvBridge()

        # initialize sub/pub
        self.img_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.feedback, queue_size=1, buff_size=2**28)
        self.img_pub = rospy.Publisher("sign/image", Image, queue_size = 1)
        self.pose_pub = rospy.Publisher("sign/detected", SignArray, queue_size= 2)

        # set transform buffer
        self.tf_buf   = tf2_ros.Buffer()
        self.tf_lstn  = tf2_ros.TransformListener(self.tf_buf)
        self.br = tf2_ros.TransformBroadcaster()

        self.category_dict = CATEGORY_DICT
        self.threshold = CONF_THRESHOLD


    def publish_bounding_image(self, img, bbs):

        cv_image = img
        if bbs[0]:
            for box in bbs[0]:
                # add bounding box
                start_point = (box["x"], box["y"])
                text_point = (box["x"], box["y"] - 5)
                end_point = (box["x"] + box["width"], box["y"]+box["height"])
                sign_type = self.category_dict[box["category"]]["name"]
                cv2.rectangle(cv_image, start_point, end_point, (0,255,0),2)
                start_point = (box["x"]-EXP, box["y"]-EXP)
                end_point = (box["x"] + box["width"]+EXP, box["y"]+box["height"]+EXP)
                cv2.rectangle(cv_image, start_point, end_point, (0,0,255),2)
                cv2.putText(img,sign_type,text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        try:
            # Publish image
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))  
        except CvBridgeError as e:
            print(e)



    def publish_pose(self, img, bbs, q, tr):

        list_sign = SignArray()
        list_sign.header.stamp = img.header.stamp
        list_sign.header.frame_id = "cf1/camera_link"
        idx = 0
        for box in bbs[0]:
            t = Sign()
            t.header.stamp = img.header.stamp
            t.header.frame_id = "cf1/camera_link"
            t.type = CATEGORY_DICT[box["category"]]["name"]
            t.confidence = float(box["confidence"])

            if q and q[idx] is not None:
                    t.pose.pose.position.x = tr[idx][0]
                    t.pose.pose.position.y = tr[idx][1]
                    t.pose.pose.position.z = tr[idx][2]    
                    (t.pose.pose.orientation.x,
                    t.pose.pose.orientation.y,
                    t.pose.pose.orientation.z,
                    t.pose.pose.orientation.w) = (q[idx][0],q[idx][1],q[idx][2],q[idx][3])

                    list_sign.signs.append(t) 
                    
            idx += 1

        self.pose_pub.publish(list_sign)

        return list_sign
    

    def publish_trans(self, sign_list):
        sign_tflist = []
        if sign_list.signs:
            for sign in sign_list.signs:
                
                sign_type = sign.type

                # t = PoseStamped()
                
                # t.header.stamp = sign.header.stamp
                # t.header.frame_id = 'cf1/camera_link'
                # t.pose = marker.pose.pose
                # if not tf_buf.can_transform(t.header.frame_id, 'map', t.header.stamp):
                #     rospy.logwarn_throttle(5.0, 'No transform from %s to map' % t.header.frame_id)
                #     return

                tf_sign = TransformStamped()
                tf_sign.header.stamp = sign.header.stamp
                tf_sign.header.frame_id = 'cf1/camera_link'
                tf_sign.child_frame_id = 'sign/detected_' + str(sign_type)
                tf_sign.transform.translation = sign.pose.pose.position
                tf_sign.transform.rotation = sign.pose.pose.orientation
                sign_tflist.append(tf_sign)

            for tf in sign_tflist:
                self.br.sendTransform(tf)


    def feedback(self,data):
        """
        Feedback function for Image topic
        Going through Training Model/Feature detector
        """
        
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # adjust input dimension -> 3d to 4d
        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        # get bounding boxes
        with torch.no_grad():
            s_t = time.time()
            out = self.model(image)
            bbs = self.model.decode_output(out, self.threshold)
        
        # publish image
        self.publish_bounding_image(img,bbs)

        if bbs[0] and PUB_POSE:  # if any detection
            fet_detector = Feature_detector(img,bbs,EXP)
            q, t = fet_detector.pose_estimation()
            sign_list = self.publish_pose(data, bbs, q, t)
            self.publish_trans(sign_list)



def main(args):
    rospy.init_node('yolo_detector', anonymous=True)
    rospy.loginfo("Yolo Detector Staring Working")

    # initialize detector
    file = 'trained_model/det_2021-04-11_13-56-50-15class.pt'
    device = 'cpu'
    detector = yolo_detector(file,device)

    # uncomment when testing locally
    
    #while not rospy.is_shutdown():
    # dir = os.path.dirname(__file__)
    # img_path = os.path.join(dir,'test_images/img_3.jpg')
    # img = cv2.imread(img_path)
    # detector.feedback(img)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
    
    
