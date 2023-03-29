#!/usr/bin/env python3  
import numpy as np
import roslib
import rospy
from sensor_msgs.msg import JointState as JointState_sensor
# from limb_core_msgs.msg import JointState as JointState_limb
import rospy
from std_msgs.msg import String
# from contact_taxels_publish import ContactPositionPub
from publisher import *
import math


# import kinpy as kp

class MySubscriber(object):
    def __init__(self):
        # rospy.Subscriber('/fpt_hand_left/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_callback)
        self.trans_obj = TransformjointPub()

    def joint_callback(self, data):
        joint_state = JointState_sensor()
        # joint_state.name = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_4.0', 'joint_5.0',
        #                     'joint_6.0',
        #                     'joint_7.0', 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_12.0',
        #                     'joint_13.0',
        #                     'joint_14.0', 'joint_15.0']
        joint_state.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
                            'joint_6',
                            'joint_7', 'joint_8', 'joint_9', 'joint_10', 'joint_11', 'joint_12',
                            'joint_13',
                            'joint_14', 'joint_15']

        joint_state.position = data.position
        joint_state.header.stamp = rospy.Time.now()
        self.trans_obj.publish_transformjoint(joint_state)

    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('subscriber_allegro_joints_transform', anonymous=True, log_level=rospy.WARN)
    my_subs = MySubscriber()
    my_subs.loop()
