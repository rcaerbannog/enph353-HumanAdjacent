#!/usr/bin/env python3
import cv2

def pedestrian_crossing(image, prev_image):
    #crop images
    x = 400
    y = 380
    h = 200
    w = 400

    cropped_image = image[y:y+h, x:x+w]
    cropped_prev_image = prev_image[y:y+h, x:x+w] 

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_prev_image = cv2.cvtColor(cropped_prev_image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_image, gray_prev_image)
    _, thresholded_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    num_different_pixels = cv2.countNonZero(thresholded_diff)

    # cv2.imshow("Pedestrian Crossing", thresholded_diff)
    # cv2.waitKey(3)
    # print(num_different_pixels)

    if(num_different_pixels >= 100):
        crossing = True
    else:
        crossing = False
    
    return crossing