#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from line_follow import line_follow
from clue_boards import check_clue_board
from pedestrian import pedestrian_crossing
# from sift_and_cnn import clueboard_img_from_frame, clue_type_and_value
import datetime
from grass import grass

class controller:
    #constructor
    def __init__(self):
        #state machine
        self.state = "line_follow" #RESET TO "line_follow"
        self.counter = 0
        self.magenta_counter = 1
        self.prev_image = None
        self.cross_counter = 0
        self.crossed = False #RESET TO FALSE
        self.prev_good_clue = False
        self.prev_clueboard = None
        self.clueboard_counter = 1

        #create a subscriber object
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        
        #create a movement publisher object
        self.move_cmd = Twist()
        self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        #create a scoretracker publisher object
        self.pub_score_tracker = rospy.Publisher("/score_tracker", String, queue_size=1)
        rospy.sleep(1)
        self.score_tracker_msg = String()
        self.score_tracker_msg.data = str('HMNADJ,CODE,0,000')
        self.pub_score_tracker.publish(self.score_tracker_msg)
    
    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        #only check state on 5th frame
        if self.counter >= 5:
            self.state = self.check_state(cv_image, self.state)
            self.counter = 0
        else:
            self.counter += 1

        #apply state
        if self.state == "line_follow":
            #check for clue board
            clue_board, clueboard_image = check_clue_board(cv_image)

            #publish to score tracker
            # if clue_board:
            #     file_name = "good_clue_" + str(datetime.datetime.now()) + ".jpg" 
            #     cv2.imwrite(file_name, clueboard_image)
                # print('entered if statement')
                # cropped_clueboard = clueboard_img_from_frame(clueboard_image)
                # cv2.imshow("cropped clue", cropped_clueboard)
                # cv2.waitKey(3)
                # print(cropped_clueboard.shape)
                # clue_type, clue_value = clue_type_and_value(cropped_clueboard)
                # self.clueboard_counter = check_cluetype(clue_type)
                # self.score_tracker_msg.data = str('HMNADJ,CODE,'+ self.clueboard_counter + ',' + clue_value)
                # self.pub_score_tracker.publish(self.score_tracker_msg)

            #set previous clueboard to current clueboard
            # self.prev_good_clue = clue_board
            # self.prev_clueboard = clueboard_image

            # cv2.imshow("Clue Board", clueboard_image)
            # cv2.waitKey(3) 
            # print(clue_board)

            follow = line_follow()
            move, move_cmd = follow.line_drive(cv_image)

            #publish motion command
            if move:
                self.pub.publish(move_cmd)

        elif self.state == "cross_walk":
            #Stop
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            self.pub.publish(self.move_cmd)
            #print("PEDESTRIAN!!!")
            
            if pedestrian_crossing(cv_image, self.prev_image) and self.cross_counter >= 10:
                #cross cross walk
                #TODO
                self.move_cmd.angular.z = 0.2
                self.pub.publish(self.move_cmd)
                rospy.sleep(0.75)

                self.move_cmd.angular.z = 0
                self.move_cmd.linear.x = 0.25
                self.pub.publish(self.move_cmd)
                rospy.sleep(2.5)

                #set state to line follow
                self.state = "line_follow"
                self.crossed = True

            self.cross_counter += 1

        elif self.state == "grass":
            print("ooooo grass")
            
            self.move_cmd.angular.z = -0.2
            self.pub.publish(self.move_cmd)
            rospy.sleep(0.5)

            self.move_cmd.angular.z = 0
            self.move_cmd.linear.x = 0.25
            self.pub.publish(self.move_cmd)
            rospy.sleep(1)
            
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            self.pub.publish(self.move_cmd)
            # rospy.sleep(100000000000)
            self.magenta_counter += 1
            self.state = "grass_follow"
        
        elif self.state == "grass_follow":
            grass_follow = grass()
            move, move_cmd = grass_follow.line_drive(cv_image)

            #publish motion command
            if move:
                self.pub.publish(move_cmd)

        
        self.prev_image = cv_image

    def check_state(self, image, state):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #BEFORE CROSSWALK CHECK FOR CROSSWALK
        if(state == "line_follow" or state == "cross_walk") and not self.crossed:
            #check red
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

            red_binary = cv2.bitwise_and(image, image, mask=red_mask)
    
            #grayscale image
            gray_image = cv2.cvtColor(red_binary, cv2.COLOR_BGR2GRAY)
        
            #binary threshold image
            temp, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

            #find contours
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))

            if (len(sorted_contours) >= 1):
                if (cv2.moments(sorted_contours[0])['m00'] >= 10000):
                    return "cross_walk"
        
        #CHECK FOR FIRST MAGENTA LINE        
        elif state == "line_follow":
            #check red
            lower_magenta = np.array([150, 100, 100])
            upper_magenta = np.array([180, 255, 255])

            magenta_mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)

            magenta_binary = cv2.bitwise_and(image, image, mask=magenta_mask)
    
            #grayscale image
            gray_image = cv2.cvtColor(magenta_binary, cv2.COLOR_BGR2GRAY)
        
            #binary threshold image
            temp, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

            #find contours
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))

            # cv2.imshow("Magenta", magenta_binary)
            # cv2.waitKey(3) 
            # print(cv2.moments(sorted_contours[0])['m00'])

            if (len(sorted_contours) >= 1):
                if (cv2.moments(sorted_contours[0])['m00'] >= 10000):
                    return "grass"
                
        elif state == "grass_follow":
            return "grass_follow"
        # cv2.imshow("Pedestrian Check", binary_image)
        # cv2.waitKey(3) 

        #check magenta

        return "line_follow"

def check_cluetype(clue_type):
    if clue_type == "SIZE":
        return 1
    elif clue_type == "VICTIM":
        return 2
    elif clue_type == "CRIME":
        return 3
    elif clue_type == "TIME":
        return 4
    elif clue_type == "PLACE":
        return 5
    elif clue_type == "MOTIVE":
        return 6
    elif clue_type == "WEAPON":
        return 7
    elif clue_type == "BANDIT":
        return 8
    return 0

if __name__ == '__main__':
    #initialize node
    rospy.init_node('controller')

    #create image reader object
    image = controller()
    
    try:
        #create infinite while loop (that runs as fast as it can)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
