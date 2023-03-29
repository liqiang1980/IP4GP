#!/usr/bin/env python3  

from numpy import dtype, float32
import numpy as np
# from numpy.typing import _64Bit
import roslib
import rospy
# from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
import rospy
# from std_msgs.msg import String


class MarkerPub(object):
    def __init__(self):
        self.pub = rospy.Publisher('marker', Marker, queue_size=10)

    def publish_marker(self, data):
        
        rospy.loginfo(data)
        self.pub.publish(data)
        # rospy.logwarn("Pub Marker pose ="+str(data))


class EKFPub(object):
    def __init__(self):
        self.pub = rospy.Publisher('ekf', Marker, queue_size=10)

    def publish_marker(self, data):
        rospy.loginfo(data)
        self.pub.publish(data)
        rospy.logwarn("Pub EKF pose =" + str(data))


class ArrowPub(object):
    def __init__(self):
        self.pub = rospy.Publisher('arrow', Marker, queue_size=10)

    def publish_marker(self, data):
        rospy.loginfo(data)
        self.pub.publish(data)
        # rospy.logwarn("Pub Arrow pose =" + str(data))


class TransformjointPub(object):

    def __init__(self):
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        # self.pub = rospy.Publisher('joint_states', JointState.position, queue_size=10)

    def publish_transformjoint(self, data):
        rospy.loginfo(data)
        self.pub.publish(data)
        # rospy.logwarn("PUB Contact Position=" + str(data))
