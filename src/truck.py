#!/usr/bin/env python3
import cv2

def truck_position(image, prev_image):
    #crop images
    top_cropped_image = image[200:380, 266:700]
    
    # cv2.imshow("Cropped Window", top_cropped_image)
    # cv2.waitKey(3)

    top_cropped_prev_image = prev_image[200:380, 266:700]

    top_gray_image = cv2.cvtColor(top_cropped_image, cv2.COLOR_BGR2GRAY)

    top_gray_prev_image = cv2.cvtColor(top_cropped_prev_image, cv2.COLOR_BGR2GRAY)

    top_diff = cv2.absdiff(top_gray_image, top_gray_prev_image)

    _, top_thresholded_diff = cv2.threshold(top_diff, 30, 255, cv2.THRESH_BINARY)

    top_num_different_pixels = cv2.countNonZero(top_thresholded_diff)

    if top_num_different_pixels >= 100:
        go = True
    else:
        go = False

    return go