#!/usr/bin/env python3  
import re
import numpy as np
import roslib
import rospy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose
from publisher import *
import math
from cswFunc import *


class MySubscriber(object):
    def __init__(self):
        self.wrist_pose = np.eye(4)
        pose = np.array([ 0.355282059139296, 3.7631652033987613,  0.09464072782403515,  0.9998104459162866,
        -0.015224695007694435, 0.005591871918033725,  0.010770880514198894])
        pose[:3] += np.array([-0.13, 0.14, 0])
        '''test for real, 0.13, 0.026'''
        self.wrist_pose_T = trans_from_pos_quat(pose)
        self.wrist_pose_Tinv = np.linalg.inv(self.wrist_pose_T)

        # rospy.Subscriber('/in-hand/pose', Pose, self.pose_callback)
        rospy.Subscriber('/vicon/jeffrey_cup/jeffrey_cup', TransformStamped, self.pose_callback)
        self.marker_obj = MarkerPub()

    def pose_callback(self, data):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = marker.MESH_RESOURCE
        marker.mesh_resource = "package://fpt_vis/largeshaker.STL"
        marker.action = marker.ADD
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        marker.color.a = 0.8
        marker.color.r = 0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # marker.pose.orientation.x = data.orientation.x
        # marker.pose.orientation.y = data.orientation.y
        # marker.pose.orientation.z = data.orientation.z
        # marker.pose.orientation.w = data.orientation.w
        # marker.pose.position.x = data.position.x
        # marker.pose.position.y = data.position.y
        # marker.pose.position.z = data.position.z
        object_W_posquat = np.array([data.transform.translation.x,
                                     data.transform.translation.y,
                                     data.transform.translation.z,
                                     data.transform.rotation.x,
                                     data.transform.rotation.y,
                                     data.transform.rotation.z,
                                     data.transform.rotation.w])
        T_object_W = trans_from_pos_quat(object_W_posquat)
        T_object_P = np.matmul(self.wrist_pose_Tinv, T_object_W)
        T_r_90 = np.mat([[0, -1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T_p_90 = np.mat([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 0, 0, 1]])
        T_y_90 = np.mat([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        '''Jeffrey'''
        # T_object_P_90 = np.matmul(T_y_90, T_object_P)  # The object rotate 90 degree around y axis
        T_object_P_90 = T_object_P
        object_P_posquat = trans_compute_pos_quan(T_object_P_90)

        marker.pose.position.x = object_P_posquat[0]
        marker.pose.position.y = object_P_posquat[1]
        marker.pose.position.z = object_P_posquat[2]
        marker.pose.orientation.x = object_P_posquat[3]
        marker.pose.orientation.y = object_P_posquat[4]
        marker.pose.orientation.z = object_P_posquat[5]
        marker.pose.orientation.w = object_P_posquat[6]
        self.marker_obj.publish_marker(marker)

    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('sub_vicon_compute_pose', anonymous=True, log_level=rospy.WARN)
    my_subs = MySubscriber()
    my_subs.loop()
