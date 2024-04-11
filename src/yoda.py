#!/usr/bin/env python3
import cv2

def yoda_position(image, prev_image):
    #crop images
    left_cropped_image = image[0:720, 0:400]
    right_cropped_image = image[0:720, 400:800]
    left_cropped_prev_image = prev_image[0:720, 0:400]
    right_cropped_prev_image = prev_image[0:720, 400:800] 

    left_gray_image = cv2.cvtColor(left_cropped_image, cv2.COLOR_BGR2GRAY)
    right_gray_image = cv2.cvtColor(right_cropped_image, cv2.COLOR_BGR2GRAY)
    left_gray_prev_image = cv2.cvtColor(left_cropped_prev_image, cv2.COLOR_BGR2GRAY)
    right_gray_prev_image = cv2.cvtColor(right_cropped_prev_image, cv2.COLOR_BGR2GRAY)

    left_diff = cv2.absdiff(left_gray_image, left_gray_prev_image)
    right_diff = cv2.absdiff(right_gray_image, right_gray_prev_image)

    _, left_thresholded_diff = cv2.threshold(left_diff, 30, 255, cv2.THRESH_BINARY)
    _, right_thresholded_diff = cv2.threshold(right_diff, 30, 255, cv2.THRESH_BINARY)

    left_num_different_pixels = cv2.countNonZero(left_thresholded_diff)
    right_num_different_pixels = cv2.countNonZero(right_thresholded_diff)

    # cv2.imshow("Yoda Crossing", right_thresholded_diff)
    # cv2.waitKey(3)
    # print(left_num_different_pixels)
    # print(right_num_different_pixels)

    if(left_num_different_pixels >= 100):
        go = True
    elif(left_num_different_pixels <= 100 and right_num_different_pixels <= 100):
        go = True
    else:
        go = False
    
    return go