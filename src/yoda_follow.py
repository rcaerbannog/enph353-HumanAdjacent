#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class yoda_follow:
    #constructor
    def __init__(self):
        #initialize move command
        self.move_cmd = Twist()
        self.previous_error = 0
        self.int_error = 0
        self.move = True

    #Line following code
    def yoda_drive(self, image):
        #PROCESSING THE IMAGE
        #image width = 800, center = 400, height = 720
        cropped = image[0:500, 0:800]

        blurred_image = cv2.GaussianBlur(cropped, (51, 51), 0)

        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        #upper and lower brown bounds
        upper_brown = np.array([10, 255, 255])
        lower_brown = np.array([0, 70, 70])

        #create mask
        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

        #apply mask
        brown_binary = cv2.bitwise_and(cropped, cropped, mask=mask)

        # cv2.imshow("Image Window", brown_binary)
        # cv2.waitKey(3) 
        
        #FINDING THE LINE
        #find the contours of the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) >= 1):
            #order contours
            #remove zero area contours
            contours = [contour for contour in contours if int(cv2.moments(contour)['m00']) != 0]
            #remove contours with area less than 500
            contours = [contour for contour in contours if int(cv2.moments(contour)['m00']) >= 500]
            #sort contours by leftmost center of mass
            sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m10']/cv2.moments(x)['m00'], reverse=True))
            #print(cv2.moments(sorted_contours[0])['m00'])

            #add green contours to the original OpenCV image
            contour_color = (0, 255, 0)
            contour_thick = 5
            wcontours_image = cv2.drawContours(cropped, sorted_contours, 0, contour_color, contour_thick)
        
            #print image
            # cv2.imshow("Image Window", wcontours_image)
            # cv2.waitKey(3)        
        
            #centroid of contours
            M = cv2.moments(sorted_contours[0])
            cx = int(M['m10']/M['m00'])

            print(cx)  

            #PUBLISHING MOTION
            #determining error
            center = 600
            error = center - cx  
            dif_error = error - self.previous_error
            self.int_error += error

            #write motion command
            kpa = 0.015
            kda = 0.001
            kia = 0.001

            self.move_cmd.linear.x = 0.25
            self.move_cmd.angular.z = kpa * error + kda * dif_error + kia * self.int_error
            self.move = True

            self.previous_error = error

            #publish motion command
            return self.move, self.move_cmd
        else:
            self.move = False
            return self.move, self.move_cmd