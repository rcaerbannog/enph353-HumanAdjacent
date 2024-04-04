#!/usr/bin/env python3

import sys
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

def spawn_position(position):

        msg = ModelState()
        msg.model_name = 'R1'

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( msg )
            return resp

        except rospy.ServiceException:
            print ("Service call failed")

if __name__ == '__main__':
     args = list(sys.argv[1:])
     args_floats = [float(x) for x in args]

     if len(args_floats) >= 1:
          resp = spawn_position(args_floats)
    