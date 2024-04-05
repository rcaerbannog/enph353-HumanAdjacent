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

class controller:
    #constructor
    def __init__(self):
        #state machine
        self.state = "line_follow"
        self.counter = 0
        self.magenta_counter = 0

        #create a subscriber object
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        
        #create a movement publisher object
        self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        #create a scoretracker publisher object
        self.pub_score_tracker = rospy.Publisher("/score_tracker", String, queue_size=1)
        rospy.sleep(1)
        self.score_tracker_msg = String()
        self.score_tracker_msg.data = str('HMNADJ,CODE,0,000')
        self.pub_score_tracker.publish(self.score_tracker_msg)


    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        # #only check state on 10th frame
        # if self.counter >= 10:
        #     self.state = check_state(cv_image)
        #     self.counter = 0
        # else:
        #     self.counter += 1
        
        #apply state
        if self.state == "line_follow":
            #check for clue board
            clue_board, clueboard_image = check_clue_board(cv_image)
            cv2.imshow("Clue Board", clueboard_image)
            cv2.waitKey(3) 
            print(clue_board)

            follow = line_follow()
            move, move_cmd = follow.line_drive(cv_image)

            #publish motion command
            if move:
                self.pub.publish(move_cmd)

        elif self.state == "cross_walk":
            print("PEDESTRIAN!!!")

        elif self.state == "grass":
            print("ooooo grass")


def check_state(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #check red
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    #check magenta

    state = ""
    return state


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
