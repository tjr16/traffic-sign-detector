#!/usr/bin/env python

import cv2
import os
import matplotlib.pyplot as plt
from config import *

class Feature_detector:

    def __init__(self, img, bbs):

        self.img = img
        self.bbs = bbs
        self.dir = os.path.dirname(__file__)

    def detect_and_match(self, imgA, imgB):
        
        # Modify on scale space
        # scale_new = (imgA.shape[1], int(imgB.shape[0] * imgA.shape[1]/imgB.shape[1]))
        # imgB = cv2.resize(imgB, scale_new, interpolation = cv2.INTER_AREA)

        grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        
        # SIFT detector 
        # sift = cv2.SIFT_create()
        # kpA, desA = sift.detectAndCompute(grayA,None)
        # kpB, desB = sift.detectAndCompute(grayB,None)

        # ORB detector
        orb = cv2.ORB_create()
        kpA, desA = orb.detectAndCompute(grayA, None)
        kpB, desB = orb.detectAndCompute(grayB, None)

        """
        Brute-Force Matcher 
        NORM_L2 for SIFT & SURF, NORM_HAMMING for binary method like ORB,BRIEF, BRISK
        """

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desA, desB)
        matches = sorted(matches, key=lambda x: x.distance)
        matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches[:100], None, flags=2)

        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.show() 

    def main(self):

        for box in self.bbs[0]:
            start_point = (box["x"], box["y"])
            end_point = (box["x"] + box["height"], box["y"]+box["width"])
            sign_type = CATEGORY_DICT[box["category"]]["name"]

            exp = 10  # expansion on bounding box
            ROI = self.img[int(start_point[1])-exp:int(end_point[1])+exp,int(start_point[0])-exp:int(end_point[0])+exp,:]

            # for each ROI extract feature compare with baseline
            baseline_path = os.path.join(self.dir,'traffic_sign/' + sign_type + '.jpg')
            base_img = cv2.imread(baseline_path)
            self.detect_and_match(ROI, base_img)
