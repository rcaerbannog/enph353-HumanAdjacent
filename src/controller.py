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
# from clue_boards import check_clue_board
from pedestrian import pedestrian_crossing
# from sift_and_cnn import clueboard_img_from_frame, clue_type_and_value
import datetime
from grass import grass
from yoda_follow import yoda_follow
from tunnel_align import rectangle_parallel
import clueboards
from truck import truck_position
from yoda import yoda_position

class controller:
    #constructor
    def __init__(self):
        #state machine
        self.state = "yoda_follow" #RESET TO "line_follow"
        self.counter = 0
        self.magenta_counter = 1 #RESET to 1
        self.prev_image = None
        self.cross_counter = 0
        self.crossed = False #RESET TO FALSE
        self.truck_counter = 0
        self.truck_check = False 
        self.prev_good_clue = False
        self.prev_clueboard = None
        self.clueboard_counter = 1
        self.yoda_counter = 0
        self.go = False
        self.turn_left = True
        self.turn_count = 0
        self.tunnel_align_count = 0
        self.tunnel_count = 0
        self.shade_count = 0
        self.hill_count = 0
        self.top_count = 0

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

        self.clueboard_img_queue = []
        self.QUEUE_SIZE = 3

    #################################
    ### CLUEBOARD READING METHODS ###
    #################################

    ## From single frame, submit clue if good clue is found
    def compute_clueboard_YOLO(self, frame):
        is_good_clue, clueboard_image = clueboards.clueboard_img_from_frame(frame)
        if is_good_clue:
            clue_type, clue_value = clueboards.clue_type_and_value(clueboard_image)
            # TODO: check for good answer
            self.clueboard_counter = check_cluetype(clue_type)
            self.score_tracker_msg.data = str('HMNADJ,CODE,'+ str(self.clueboard_counter) + ',' + clue_value)
            self.pub_score_tracker.publish(self.score_tracker_msg)
            print("compute_clueboard_YOLO: computed " + str((clue_type, clue_value)))
            return True
        print("compute_clueboard_YOLO: FAILED")
        return False

    ## Call this once in every state where you want to check if a frame has a clueboard (and add that clueboard image to the queue)
    def check_for_clueboard(self, frame):
        #check for clue board
        is_good_clue, clueboard_image = clueboards.clueboard_img_from_frame(frame)
        if is_good_clue:
            # file_name = "good_clue_" + str(datetime.datetime.now()) + ".jpg" 
            # cv2.imwrite(file_name, frame)
            if len(self.clueboard_img_queue) >= self.QUEUE_SIZE: # Keep the last 3 images
                self.clueboard_img_queue.pop(0) # pop HEAD of queue
            self.clueboard_img_queue.append(clueboard_image) # append to END of queue
            self.prev_good_clue = True
            print("check_for_clueboard: ")
        return is_good_clue

    ## Call this once in every state where you want to compute the last 3 images
    def compute_clueboard(self):
        if self.prev_good_clue:
            if len(self.clueboard_img_queue) >= self.QUEUE_SIZE: # Minimum number of good images
                # Compute values and obtain type + value, then pick best by consensus
                # output is tuple (type, value)
                # Perhaps stop the car while computing?
                results = [clueboards.clue_type_and_value(img) for img in self.clueboard_img_queue]
                print("compute_clueboard: " + str(results))
                # Maybe discard results with a certain amount of error?
                # Consensus of images
                have_type_consensus, clue_type = clueboards.consensus([result[0] for result in results])
                have_value_consensus, clue_value = clueboards.consensus([result[1] for result in results])

                # Submit clues
                if have_value_consensus:
                    self.clueboard_counter = check_cluetype(clue_type)
                    self.score_tracker_msg.data = str('HMNADJ,CODE,'+ str(self.clueboard_counter) + ',' + clue_value)
                    self.pub_score_tracker.publish(self.score_tracker_msg)
                    print("compute_clueboard: OBTAINED VALUE CONSENSUS AND SUBMITTED: " + str((clue_type, clue_value)))
                    # TODO: If clue submitted, disable clueboard reading for X frames to reduce chance of double-image?
            else:
                print(f"compute_clueboard: QUEUE NOT FILLED: ONLY {len(self.clueboard_img_queue)}/{self.QUEUE_SIZE} FRAMES IN A ROW")
            # Wipe queue and flag
            self.prev_good_clue = False
            self.clueboard_img_queue.clear()

    ################################
    ### CALLBACK / STATE MACHINE ###
    ################################
    
    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        #Check state every 5th frame
        if self.counter >= 5:
            self.state = self.check_state(cv_image, self.state)
            self.counter = 0
        else:
            self.counter += 1

        #apply state
        #LINE_FOLLOW: line following algorithm for the paved section of the competition
        if self.state == "line_follow":
            print("line_follow")
            if not self.check_for_clueboard(cv_image):
                self.compute_clueboard()            

            #compute and publish move_cmd
            follow = line_follow()
            move, move_cmd = follow.line_drive(cv_image)

            if move:
                self.pub.publish(move_cmd)

        #CROSS_WALK: stops until motion is sensed then moves forward for 2.5 seconds
        elif self.state == "cross_walk":
            #Stop
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            self.pub.publish(self.move_cmd)
            
            if pedestrian_crossing(cv_image, self.prev_image) and self.cross_counter >= 10: #cross_counter is a delay to account for the robot stopping
                #cross cross walk
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

        elif self.state == "truck":
            print("truck")
            if truck_position(cv_image and self.truck_counter >= 10):
                self.state = "line_follow"
                self.truck_check = True
            else:
                self.move_cmd.angular.z = 0
                self.move_cmd.linear.x = 0.0
                self.pub.publish(self.move_cmd)
                self.truck_counter += 1
                
        #GRASS: Transition state between pavement line following and grass line following
        elif self.state == "grass":
            print("oooooo grass")
            #move past magenta line
            self.move_cmd.angular.z = -0.2
            self.move_cmd.linear.x = 0.25
            self.pub.publish(self.move_cmd)
            rospy.sleep(1)
            
            #set state to grass follow
            self.magenta_counter += 1
            self.state = "grass_follow"
        
        #GRASS_FOLLOW: line following algorithm for the grass section
        elif self.state == "grass_follow":
            if not self.check_for_clueboard(cv_image):
                self.compute_clueboard()     

            print("grass_follow")
            grass_follow = grass()
            move, move_cmd = grass_follow.line_drive(cv_image, 175)

            #publish motion command
            if move:
                self.pub.publish(move_cmd)
        
        #YODA: turn 90 degrees to face the tunnel
        elif self.state == "yoda":
            #turn 90 degrees
            self.move_cmd.angular.z = 1.5
            self.move_cmd.linear.x = 0.0
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.5)

            #set state to yoda follow
            self.magenta_counter += 1
            self.state = "yoda_follow"
        
        #YODA_FOLLOW: tunnel following
        elif self.state == "yoda_follow":
            print("yoda_follow")
            follow = yoda_follow()
            move, move_cmd = follow.yoda_drive(cv_image)

            if move:
                self.pub.publish(move_cmd)

        elif self.state == "find_yoda":
            print("find_yoda")
            if not yoda_position(cv_image, self.prev_image) and self.yoda_counter >= 10:
                print("yoda_spotted")
                self.counter = 5
                rospy.sleep(2)
                self.go = True
            
            self.yoda_counter += 1

        #TUNNEL: transition to the entrance of the tunnel
        elif self.state == "tunnel":
            print("tunnel")
            self.move_cmd.angular.z = -1.5
            self.move_cmd.linear.x = 0.0
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.4) 

            self.move_cmd.angular.z = 0.0
            self.move_cmd.linear.x = 0.0
            self.pub.publish(self.move_cmd) 

            self.state = "find_yoda"

        elif self.state == "tunnel_end":
            print("tunnel_end")
            self.move_cmd.angular.z = 0.0
            self.move_cmd.linear.x = 0.5
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.15) 

            self.move_cmd.angular.z = 1.5
            self.move_cmd.linear.x = 0
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.5)  

            self.move_cmd.angular.z = 0.0
            self.move_cmd.linear.x = 0.5
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.325)

            self.move_cmd.angular.z = 1.5
            self.move_cmd.linear.x = 0.0
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.1)
            
            self.counter = 5

        #TUNNEL_ALIGN: aligns perpendicular to the magenta line to go straight through the tunnel
        elif self.state == "tunnel_align":
            print("tunnel_align")
            if self.turn_left:
                if self.turn_count <= 60:
                    self.move_cmd.angular.z = 0.5
                    self.move_cmd.linear.x = 0.0
                    self.pub.publish(self.move_cmd)

                    self.turn_count += 1
                else:
                    self.turn_left = False
                    self.turn_count = 0
            else:
                if self.turn_count <= 60:
                    self.move_cmd.angular.z = -0.5
                    self.move_cmd.linear.x = 0.0
                    self.pub.publish(self.move_cmd)

                    self.turn_count += 1
                else:
                    self.turn_left = True
                    self.turn_count = 0

            if rectangle_parallel(cv_image) and self.tunnel_align_count >= 10:
                print("parallel")
                self.state = "tunnel_drive"
            self.tunnel_align_count += 1

        #TUNNEL_DRIVE: drive straight through the tunnel
        elif self.state == "tunnel_drive":
            if not self.check_for_clueboard(cv_image):
                self.compute_clueboard()  
                
            print("tunnel_drive")
            if self.tunnel_count <= 50:
                self.move_cmd.angular.z = 0.0
                self.move_cmd.linear.x = 0.5
                self.pub.publish(self.move_cmd)
                self.tunnel_count += 1  
            else:
                self.state = "shade_follow"  

        #SHADE_FOLLOW: grass follow but with a different threshold due to shade
        elif self.state == "shade_follow":
            print("shade_follow")
            if self.shade_count <= 90:
                grass_follow = grass()
                move, move_cmd = grass_follow.line_drive(cv_image, 155)

                #publish motion command
                if move:
                    self.pub.publish(move_cmd)

                self.shade_count += 1
            else:
                self.state = "hill_follow"     

        elif self.state == "hill_follow":
            if not self.check_for_clueboard(cv_image):
                self.compute_clueboard()     

            print("hill_follow")
            
            if self.hill_count <= 350:
                grass_follow = grass()
                move, move_cmd = grass_follow.line_drive(cv_image, 175)

                #publish motion command
                if move:
                    self.pub.publish(move_cmd)

                self.hill_count += 1
                print(self.hill_count)
            else:
                self.counter = 5
                self.state = "top_turn"

        elif self.state == "top_turn":
            print("top_turn")
            self.move_cmd.angular.z = 0.0
            self.move_cmd.linear.x = 0.25
            self.pub.publish(self.move_cmd)
            rospy.sleep(2.5)

            self.move_cmd.angular.z = 1.5
            self.move_cmd.linear.x = 0.0
            self.pub.publish(self.move_cmd)
            rospy.sleep(1.3)

            self.counter = 5
            self.state = "top"

        elif self.state == "top":
            print("top")
            if self.top_count <= 85:
                if not self.check_for_clueboard(cv_image):
                    self.compute_clueboard()  

                self.move_cmd.angular.z = 0.0
                self.move_cmd.linear.x = 0.25
                self.pub.publish(self.move_cmd)

                self.top_count += 1
            else:
                if (self.prev_good_clue):
                    self.compute_clueboard()
                self.move_cmd.angular.z = 0.0
                self.move_cmd.linear.x = 0.0
                self.pub.publish(self.move_cmd)
                self.score_tracker_msg.data = str('HMNADJ,CODE,-1,000')
                self.pub_score_tracker.publish(self.score_tracker_msg)
                self.counter = 5
                self.state = "end"

        elif self.state == "end":
            print("DONE \(^o^)/")


        self.prev_image = cv_image

    #LOOKS AT CURRENT FRAME AND RETURNS CURRENT STATE
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
        
        #CHECK FOR MAGENTA LINE (1: before grass section, 2: before yoda section, 3: before tunnel)       
        elif state == "line_follow" or state == "grass_follow":
            #check magenta
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

            if (len(sorted_contours) >= 1):
                if (cv2.moments(sorted_contours[0])['m00'] >= 10000):
                    if self.magenta_counter == 1:
                        return "grass"
                    if self.magenta_counter == 2:
                        return "yoda"

        #CHECK FOR CLUEBOARD NEXT TO THE TUNNEL
        elif state == "yoda_follow":
            #upper and lower blue bounds
            upper_blue = np.array([150, 255, 255])
            lower_blue = np.array([120, 50, 50])
            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            blue_binary = cv2.bitwise_and(image, image, mask=mask)           

            #find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = tuple(sorted(contours, key = lambda x: cv2.moments(x)['m00'], reverse = True))                        
            
            if (len(sorted_contours) >= 1):
                if (cv2.moments(sorted_contours[0])['m00'] >= 10000):
                    return "tunnel"
                
        elif state == "line_follow" and self.clueboard_counter == 3 and not self.truck_check:
            return "truck"
        
        elif state == "find_yoda" and self.go:
            return "tunnel_end"
        
        elif state == "tunnel_end":
            return "tunnel_align"

        if state == "end":
            return "end"

        if state == "top":
            return "top"
        
        if state == "top_turn":
            return "top_turn"

        if state == "find_yoda":
            return "find_yoda"

        if state == "hill_follow":
            return "hill_follow"
        
        if state == "truck":
            return "truck"
        
        if state == "shade_follow":
            return "shade_follow"
        
        if state == "tunnel_drive":
            return "tunnel_drive"

        if state == "tunnel_align":
            return "tunnel_align"
        
        if state == "yoda_follow":
            return "yoda_follow"
                
        if state == "grass_follow":
            return "grass_follow"

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
