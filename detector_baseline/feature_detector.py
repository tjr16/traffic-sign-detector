#!/usr/bin/env python

import cv2
import os
import matplotlib.pyplot as plt
from tf.transformations import quaternion_from_euler
import numpy as np
import random
from config import *

MIN_MATCH_COUNT = 4

CAM_DATA =[231.250001, 0.000000, 320.519378, 0.000000, 231.065552, 240.631482, 0.000000, 0.000000, 1.000000]
DIS_DATA = [0.061687, -0.049761, -0.008166, 0.004284, 0.0]
CAM_MTS = np.array(CAM_DATA).reshape(3,3)
CAM_DIS = np.array(DIS_DATA)

class Feature_detector:

    def __init__(self, img, bbs):

        self.img = img
        self.bbs = bbs
        self.dir = os.path.dirname(__file__)

    def detect_and_match(self, imgA, imgB, maskA = None, maskB = None):
        

        # Modify on scale space
        # scale_new = (imgA.shape[1], int(imgB.shape[0] * imgA.shape[1]/imgB.shape[1]))
        # imgB = cv2.resize(imgB, scale_new, interpolation = cv2.INTER_AREA)

        grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        
        # SIFT detector 
        sift = cv2.xfeatures2d.SIFT_create()
        kpA, desA = sift.detectAndCompute(grayA,maskA)
        kpB, desB = sift.detectAndCompute(grayB,maskB)
        # ORB detector
        # orb = cv2.ORB_create()
        # kpA, desA = orb.detectAndCompute(grayA, maskA)
        # kpB, desB = orb.detectAndCompute(grayB, maskB)

        """
        Brute-Force Matcher 
        NORM_L2 for SIFT & SURF, NORM_HAMMING for binary method like ORB,BRIEF, BRISK
        """

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desA, desB)
        dmatches = sorted(matches, key=lambda x: x.distance)

        ## extract the matched keypoints
        img_pts  = np.float32([kpA[m.queryIdx].pt for m in dmatches]).reshape(-1,2)
        src_pts  = np.float32([kpB[m.trainIdx].pt for m in dmatches]).reshape(-1,2)

        obj_pts = self.kp_trans(src_pts,imgB)
        rvecs, tvecs, inliers= self.pnp_iter(obj_pts,img_pts)
        q, t = self.inv_trans(rvecs,tvecs)

        # for testing 
        # print("rvec=", rvecs)
        # print("tvec=", tvecs)
        # print("inliers=", inliers)
        # print("inv_tvec=",t)
        # print(q)

        return q,t

        # matched_image =cv2.drawMatches(imgA,kpA,imgB,kpB,dmatches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        # plt.show()
    
    def pnp_iter(self, obj_pts, img_pts, iter = 100,pair=10):
        max_in = 0
        r_vecs, t_vecs, in_liers = None, None, None
        for i in range(iter):
            obj_sub, img_sub = zip(*random.sample(zip(obj_pts, img_pts), pair))
            obj_sub = np.array(obj_sub)
            img_sub = np.array(img_sub)
            _, rvecs, tvecs, inliers= cv2.solvePnPRansac(obj_sub, img_sub, CAM_MTS, CAM_DIS)
            if inliers is not None:
                n_in = len(inliers)
                if n_in > max_in:
                    max_in = n_in
                    r_vecs, t_vecs, in_liers = rvecs, tvecs, inliers
        return r_vecs, t_vecs, in_liers


    def kp_trans(self, pts, base_img,ratio = 0.2645/1000):
        """
        transform image coordinate to real world coordinate
        image size: 794 * 1123
        A4 size: 210mm * 297mm
        """
        height,width,channel = base_img.shape
        wp = []
        for point in pts:
            new_point = [(point[1]-height/2)*ratio,(point[0]-width/2)*ratio,0]
            wp.append(new_point)
        wp = np.array(wp)
        return wp


    def inv_trans(self,rvecs,tvecs):
        R_mat,_ = cv2.Rodrigues(rvecs)
        R_mat =np.asmatrix(R_mat)
        R = R_mat.transpose()
        t = - R * tvecs
        rvet = cv2.Rodrigues(R)[0].flatten()
        q = [0,0,0,0]
        # x, y, z, w
        (q[0],q[1],q[2],q[3]) = quaternion_from_euler(rvet[0],rvet[1],rvet[2])
        return q,t


    def main(self):

        for box in self.bbs[0]:
            start_point = (box["x"], box["y"])
            end_point = (box["x"] + box["width"], box["y"]+box["height"])
            sign_type = CATEGORY_DICT[box["category"]]["name"]

            exp = 0  # expansion on bounding box
            ROI = self.img[int(start_point[1])-exp:int(end_point[1])+exp,int(start_point[0])-exp:int(end_point[0])+exp,:]
            
            # for each ROI extract feature compare with baseline
            baseline_path = os.path.join(self.dir,'traffic_sign2/' + sign_type + '.jpg')
            #baseline_path = os.path.join(self.dir,'traffic_sign2/roundabout.jpg')
            base_img = cv2.imread(baseline_path)
            w, h = self.img.shape[:2]
            mask = np.zeros([w,h], dtype=np.uint8)
            mask[int(start_point[1])-exp:int(end_point[1])+exp,int(start_point[0])-exp:int(end_point[0])+exp]=1
            self.detect_and_match(self.img, base_img, maskA = mask)
