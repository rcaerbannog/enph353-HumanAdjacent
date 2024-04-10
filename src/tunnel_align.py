#!/usr/bin/env python3

import cv2
import numpy as np

def rectangle_parallel(image):
    tolerance_degrees = 1
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #check magenta
    lower_magenta = np.array([150, 100, 100])
    upper_magenta = np.array([180, 255, 255])
    magenta_mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)
    magenta_binary = cv2.bitwise_and(image, image, mask=magenta_mask)

    #find contours
    contours, hierarchy = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))

    #find the top line
    epsilon = 0.02 * cv2.arcLength(sorted_contours[0], True)
    approx = cv2.approxPolyDP(sorted_contours[0], epsilon*0.8, True)

    # Draw the top line on the image
    wcontours = cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)   

    cv2.imshow("line", wcontours)
    cv2.waitKey(3)

    if approx.shape[0] != 4:
        return False

    # Get the endpoints of the top line of the contour
    sorted_y_points = sorted(approx, key = lambda x: x[0][1])
    top_points = sorted_y_points[0:2]
    left_point = top_points[0]
    right_point = top_points[1]
    
    print(approx)
    print(top_points)

    # Calculate the angle of rotation of the top line
    angle = np.arctan2(right_point[0][1] - left_point[0][1], right_point[0][0] - left_point[0][0]) * 180 / np.pi

    print(angle)

    # Calculate the deviation of the angle from 0 or 180 degrees
    deviation_from_0 = abs(angle)
    deviation_from_180 = abs(180 - angle)

    # Determine if the deviation is within the tolerance
    is_parallel_to_bottom = deviation_from_0 <= tolerance_degrees or deviation_from_180 <= tolerance_degrees


    return is_parallel_to_bottom
