#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class grass:
    #constructor
    def __init__(self):
        #initialize move command
        self.move_cmd = Twist()
        self.previous_error = 0
        self.int_error = 0
        self.move = True

    #Line following code
    def line_drive(self, image, threshold):
        #PROCESSING THE IMAGE
        #image width = 800, center = 400, height = 720
        
        #crop image
        cropped_image = image[360:720, 0:400]

        #grayscale image
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        #blur image (to remove noise)
        blurred_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        #inverse binary threshold image
        temp, binary_gray_image = cv2.threshold(blurred_gray_image, threshold, 255, cv2.THRESH_BINARY) #uphill shade theshold = 155, regular threshold = 175

        #FINDING THE LINE
        #find the contours of the binary image
        contours, hierarchy = cv2.findContours(binary_gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) >= 1):
            #order contours
            #remove zero area contours
            contours = [contour for contour in contours if int(cv2.moments(contour)['m00']) != 0]
            #remove contours with area less than 5000
            contours = [contour for contour in contours if int(cv2.moments(contour)['m00']) >= 5000]
            #sort contours by leftmost center of mass
            sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m10']/cv2.moments(x)['m00']))
            #print(cv2.moments(sorted_contours[0])['m00'])

            #add green contours to the original OpenCV image
            contour_color = (0, 255, 0)
            contour_thick = 5
            wcontours_image = cv2.drawContours(cropped_image, sorted_contours, 0, contour_color, contour_thick)       

            # cv2.imshow("Cropped Window", binary_gray_image)
            # cv2.waitKey(3)

            if len(sorted_contours) >= 1:
                #centroid of contours
                M = cv2.moments(sorted_contours[0])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # print(cy)  
                
                #print image
                cv2.imshow("Cropped Window", wcontours_image)
                cv2.waitKey(3)

                #PUBLISHING MOTION
                #determining error
                center = 195
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
        else:
            self.move = False
            return self.move, self.move_cmd

