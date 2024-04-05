#!/usr/bin/env python3
import cv2
import numpy as np
import datetime

def check_clue_board(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #upper and lower blue bounds
    upper_blue = np.array([150, 255, 255])
    lower_blue = np.array([120, 50, 50])

    #create mask
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    #apply mask
    blue_binary = cv2.bitwise_and(image, image, mask=mask)

    #grayscale image
    gray_image = cv2.cvtColor(blue_binary, cv2.COLOR_BGR2GRAY)
        
    #binary threshold image
    temp, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

    #find contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if int(cv2.moments(contour)['m00']) >= 2500]
    sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))

    if len(sorted_contours) >= 1:
        x, y, w, h = cv2.boundingRect(sorted_contours[0])

        #crop image around bounding box
        cropped_image = image[y:y+h, x:x+w]

        #check if cropped image is a good clue
        return good_clue(cropped_image)

    return False, image


#use SIFT to see if cropped image is useful
def good_clue(image):
    good = True

    #check size of the image
    height, width = image.shape[:2]
    if (height <= 130):
        good = False
        return good, image

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #upper and lower blue bounds
    upper_blue = np.array([150, 255, 255])
    lower_blue = np.array([120, 50, 50])

    #create mask
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    #apply mask
    blue_binary = cv2.bitwise_and(image, image, mask=mask)

    #grayscale image
    gray_image = cv2.cvtColor(blue_binary, cv2.COLOR_BGR2GRAY)
        
    #binary threshold image
    temp, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)

    #find contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))
    
    #check if contour crosses the edge
    for point in sorted_contours[0]:
        # Get the coordinates of the point
        x, y = point[0]

        # Check if the point is on the edge of the image
        if x == 0 or x == width - 1 or y == 0 or y == height - 1:
            good = False

    # print(good)
    # if good:
    #     #print image
    #     cv2.imshow("Good Clue", image)
    #     cv2.waitKey(3)
        
    #     print(image.shape[:2])

        # file_name = "good_clue_" + str(datetime.datetime.now()) + ".jpg" 
        # cv2.imwrite(file_name, image)

    return good, image
