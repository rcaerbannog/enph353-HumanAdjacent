#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from tensorflow.keras import models

# LETTER READING SETUP
letter_thresh_minHSV = (int(230/2), 128, 0)
letter_thresh_maxHSV = (int(250/2), 255, 255)
LETTER_IMG_DIM_Y = 80
LETTER_IMG_DIM_X = 64
LETTER_IMG_SHAPE = (LETTER_IMG_DIM_Y, LETTER_IMG_DIM_X)
MIN_LETTER_HEIGHT = 20

# IMPORT LETTER-RECOGNITION CNN
model = models.load_model('letter-recog-model.keras')

def is_good_clueboard_contour(contour, in_frame):
    frame_dimY = in_frame.shape[0]
    frame_dimX = in_frame.shape[1]
    x, y, w, h = cv.boundingRect(contour)
    # Failure conditions: height too small, contour touches edge of screen (indicates partly out of frame)
    CLUEBOARD_MIN_HEIGHT = 130
    return not (h < CLUEBOARD_MIN_HEIGHT or x == 0 or y == 0 or x+w >= frame_dimX-1 or y+h >= frame_dimY-1)

## Call this
def clueboard_img_from_frame(frame):
    DIM_Y = 400
    DIM_X = 600
    CLUEBOARD_MIN_AREA = 10000
    
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #upper and lower blue bounds
    upper_blue = np.array([150, 255, 255])
    lower_blue = np.array([120, 50, 50])

    #create mask
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    #find contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda x : cv.contourArea(x) >= CLUEBOARD_MIN_AREA, contours)
    sorted_contours = tuple(sorted(contours, key = lambda x: cv.contourArea(x), reverse = True)) # largest to smallest contours
    
    # At the very least, we should detect an inner and outer contour of the blue border.
    # If part of the clueboard is out-of-frame, we can tell by the largest contour touching the edge. (Check bounding box)
    if len(sorted_contours) >= 2:
        clueboard_contour = sorted_contours[1] # Hopefully the second contour is the interior one
        if is_good_clueboard_contour(clueboard_contour, mask):
            # Get bounding quadrilateral
            perim = cv.arcLength(clueboard_contour, closed=True)
            quadrilateral = cv.approxPolyDP(clueboard_contour, epsilon=0.02*perim, closed=True)
            if len(quadrilateral) == 4:
                quad_pts = np.array([item for sublist in quadrilateral for item in sublist], np.float32)
                # Rotate the quadrilateral to order: top-left, top-right, bottom-right, bottom-left
                s = np.array(sorted(quad_pts, key=lambda i: i[1], reverse=False)) # Sort by y-coord. Top / Bot sep
                top_left, top_right = (0, 1) if s[0][0] < s[1][0] else (1, 0)
                bot_left, bot_right = (2, 3) if s[2][0] < s[3][0] else (3, 2)
                quad_pts_std = np.array([s[top_left], s[top_right], s[bot_right], s[bot_left]], dtype=np.float32)
                # the points to transform to (clueboard template image 600x400)
                h = np.array([ [0,0],[DIM_X-1,0],[DIM_X-1,DIM_Y-1],[0,DIM_Y-1] ],np.float32)
                transform = cv.getPerspectiveTransform(quad_pts_std, h)
                clueboard_img = cv.warpPerspective(frame, transform, (DIM_X, DIM_Y))
                return True, clueboard_img # Warped clueboard image (masked), ready for processing
        
    return False, frame # Did not detect clue

# returns text corresponding to array letter_imgs in order (may give incorrect result)
# images must be of shape (LETTER_DIM_Y, LETTER_DIM_X)
# for empty array, returns empty string
# NOTE: Consider adding a CONFIDENCE check (sth like RANSAC ratio test)
def model_wrapper(letter_imgs):
    if len(letter_imgs) == 0: 
        return ''
    else:
        predictions = model.predict(letter_imgs)
        text = ''
        for prediction in predictions:
            text += chr(ord('A') + np.argmax(prediction))
        return text

def box_contains_letter(bounding_rect):
    x, y, w, h = bounding_rect
    # Failure conditions:
    # 1) box exceeds letter image size (cannot squeeze larger array into smaller one)
    # 2) height is way too small (likely a small artifact)
    # 3) box touches edges (likely a blue border contour)
    return not (w > LETTER_IMG_DIM_X or h > LETTER_IMG_DIM_Y 
                or h < MIN_LETTER_HEIGHT 
                or x == 0 or y == 0 or x+w == LETTER_IMG_DIM_X - 1 or y + h == LETTER_IMG_DIM_Y - 1)


## API Method
# Takes a clueboard subimage with a line of text (capital letters A-Z in Ubuntu Monospace font ~size 90) and returns the text on that image
    # If no letters are detected, an empty string ''. 
    # Correctness NOT guaranteed.
# May see if can work on grayscale or BGR image for computational efficiency (but HSV is most robust due to brightness invariance)
# NOTE: might implement attemped erosion of contours which look like 2 or 3 letters merged together
# NOTE: optimizable: make all operations in terms of HSV image
# @param img_BGR (np.ndarray) an image
# @return (string) the line of text detected on the image 
def text_from_line_image(img_BGR):
    img_HSV = cv.cvtColor(img_BGR, cv.COLOR_BGR2HSV)
    img_mask = cv.inRange(img_HSV, letter_thresh_minHSV, letter_thresh_maxHSV)
    contours_ext, hierarchy_ext = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_rects_ext = [cv.boundingRect(contour) for contour in contours_ext]
    bounding_rects_ext = sorted(list(filter(box_contains_letter, bounding_rects_ext)))
    # contours_ext_sorted = sorted(contours_ext, key=cv.boundingRect, reverse=True)

    letter_imgs = []
    for bounding_rect in bounding_rects_ext:
        # NOTE: for existing network, must use solid image, and not outline. 
        # NOTE: Changed 2024-04-07: if a bounding box exceeds the letter image shape, return none (to avoid broadcasting error)
        # NOTE: Changed 2024-04-09: filtered out bad bounding boxes in bounding_rects_ext a few lines before
        x, y, w, h = bounding_rect
        
        subimg = np.copy(img_mask[y:y+h, x:x+w])
        letter_img = np.zeros(LETTER_IMG_SHAPE, dtype=np.uint8)
        start_x = (LETTER_IMG_DIM_X - w) // 2
        start_y = (LETTER_IMG_DIM_Y - h) // 2
        letter_img[start_y : start_y + h, start_x : start_x + w] = subimg
        letter_imgs.append(letter_img)

    letter_imgs_arr = np.array(letter_imgs).reshape(-1, LETTER_IMG_DIM_Y, LETTER_IMG_DIM_X)
    text = model_wrapper(letter_imgs_arr)
    return text

## Clue type and value from clueboard image in same shape as `clue-banner-filled.png`. 
# Image must be SIFTed to match clueboard template size. 
# Correctness not guaranteed.
# @param clueboard_img (np.ndarray) a cv2-BGR-compatible SIFTed clueboard image in same shape as `clue-banner-filled.png`
# @return a 2-tuple of (clue_type, clue_value), where the entries may be a string or None if no text detected (possibly one each)
def clue_type_and_value(clueboard_img):
    # First, split image into type and image sections
    # NOTE: 2024-04-09: expanding the slice occasionally causes letters to drop out??? 
    # TODO: Check reasons why contours are being discarded!!!
    clue_type_img = clueboard_img[35:115, 240:580]
    clue_value_img = clueboard_img[255:335, 20:580]

    clue_type = text_from_line_image(clue_type_img)
    clue_value = text_from_line_image(clue_value_img)

    return clue_type, clue_value

# Strings is not empty
# Obtaining consensus defined as winning a majority vote. (50% is not good enough -- if 4 entries, need at least 3 matching)
# NOTE: if confident we tend to remove letters than add, may also be worth it to add subsequence checking
def consensus(entries):
    unique = {} # entries (entry, num of occurences)
    for entry in entries:
        if entry in unique:
            unique[entry] += 1
        else:
            unique[entry] = 1
    unique_list = list(unique)
    unique_list.sort(key = lambda x : x[1], reverse=True)
    mode, occurences = unique_list[0]
    if occurences > len(entries) // 2:
        return True, mode
    else:
        return False, None


