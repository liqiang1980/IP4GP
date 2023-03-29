#!/usr/bin/env python3
import copy
import re

import genpy
import numpy as np
import roslib
import rospy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose
from publisher import *
import math
from cswFunc import *
from allegro_tactile_sensor.msg import tactile_msgs
import qgFunc as qg
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from sensor_msgs.msg import JointState


class MySubscriber(object):
    def __init__(self):
        self.wrist_pose = np.eye(4)
        pose = np.array([ 0.355282059139296, 3.7631652033987613,  0.09464072782403515,  0.9998104459162866,
        -0.015224695007694435, 0.005591871918033725,  0.010770880514198894])
        '''Jeffrey: wrist pose 2 hand pose'''
        # pose[:3] += np.array([0.1, 0.1, 0])
        self.wrist_pose_T = trans_from_pos_quat(pose)
        self.wrist_pose_Tinv = np.linalg.inv(self.wrist_pose_T)

        # init kdl_JntArrays for each finger
        self.index_qpos = kdl.JntArray(4)
        self.mid_qpos = kdl.JntArray(4)
        self.ring_qpos = kdl.JntArray(4)
        self.thumb_qpos = kdl.JntArray(4)
        kdl.SetToZero(self.index_qpos)
        kdl.SetToZero(self.mid_qpos)
        kdl.SetToZero(self.ring_qpos)
        kdl.SetToZero(self.thumb_qpos)
        # init kdl_Frames for each finger
        self.index_pos = kdl.Frame()  # Construct an identity frame
        self.mid_pos = kdl.Frame()  # Construct an identity frame
        self.ring_pos = kdl.Frame()  # Construct an identity frame
        self.thumb_pos = kdl.Frame()  # Construct an identity frame
        self.hand_description = URDF.from_xml_file(
            '/home/manipulation-vnc/catkin_ws/src/fpt_vis/launch/allegro_hand_tactile_v1.4.urdf')  # for FK
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)  # for FK
        # chain
        # self.index_chain = self.hand_tree.getChain("palm_link", "index_tip_tactile")
        # self.mid_chain = self.hand_tree.getChain("palm_link", "mid_tip_tactile")
        # self.ring_chain = self.hand_tree.getChain("palm_link", "ring_tip_tactile")
        # self.thumb_chain = self.hand_tree.getChain("palm_link", "thumb_tip_tactile")
        self.index_chain = self.hand_tree.getChain("palm_link", "link_3.0_tip_tactile")
        self.mid_chain = self.hand_tree.getChain("palm_link", "link_7.0_tip_tactile")
        self.ring_chain = self.hand_tree.getChain("palm_link", "link_11.0_tip_tactile")
        self.thumb_chain = self.hand_tree.getChain("palm_link", "link_15.0_tip_tactile")
        # forward kinematicsallegro_jstates
        self.index_fk = kdl.ChainFkSolverPos_recursive(self.index_chain)
        self.mid_fk = kdl.ChainFkSolverPos_recursive(self.mid_chain)
        self.ring_fk = kdl.ChainFkSolverPos_recursive(self.ring_chain)
        self.thumb_fk = kdl.ChainFkSolverPos_recursive(self.thumb_chain)
        # T_tip_tactile in {P}. Update by FK.
        self.index_in_palm_T = np.mat(np.eye(4))
        self.mid_in_palm_T = np.mat(np.eye(4))
        self.ring_in_palm_T = np.mat(np.eye(4))
        self.thumb_in_palm_T = np.mat(np.eye(4))
        self.marker_id = 0

        # rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_callback)
        self.threshold = 20  # Judge if taxel is contacted
        rospy.Subscriber('/allegro_tactile', tactile_msgs, self.taxel_callback)
        self.marker_taxel = ArrowPub()

    # def joint_callback(self, data):
    #     all_joint = data.position
    #     self.index_qpos[0] = all_joint[0]
    #     self.index_qpos[1] = all_joint[1]
    #     self.index_qpos[2] = all_joint[2]
    #     self.index_qpos[3] = all_joint[3]
    #     self.mid_qpos[0] = all_joint[4]
    #     self.mid_qpos[1] = all_joint[5]
    #     self.mid_qpos[2] = all_joint[6]
    #     self.mid_qpos[3] = all_joint[7]
    #     self.ring_qpos[0] = all_joint[8]
    #     self.ring_qpos[1] = all_joint[9]
    #     self.ring_qpos[2] = all_joint[10]
    #     self.ring_qpos[3] = all_joint[11]
    #     self.thumb_qpos[0] = all_joint[12]
    #     self.thumb_qpos[1] = all_joint[13]
    #     self.thumb_qpos[2] = all_joint[14]
    #     self.thumb_qpos[3] = all_joint[15]
    #     self.index_in_palm_T, self.mid_in_palm_T, self.ring_in_palm_T, self.thumb_in_palm_T = self.fk_dealer()

    # def fk_dealer(self):
    #     """
    #     Get T (tips in palm) and J by FK method
    #     joint positions are updated in main_process()
    #     """
    #     M = np.mat(np.zeros((3, 3)))
    #     p = np.zeros([3, 1])
    #     index_in_palm_T = np.mat(np.eye(4))
    #     mid_in_palm_T = np.mat(np.eye(4))
    #     ring_in_palm_T = np.mat(np.eye(4))
    #     thumb_in_palm_T = np.mat(np.eye(4))
    #
    #     # forward kinematics
    #     qg.kdl_calc_fk(self.index_fk, self.index_qpos, self.index_pos)
    #     M[0, 0] = copy.deepcopy(self.index_pos.M[0, 0])
    #     M[0, 1] = copy.deepcopy(self.index_pos.M[0, 1])
    #     M[0, 2] = copy.deepcopy(self.index_pos.M[0, 2])
    #     M[1, 0] = copy.deepcopy(self.index_pos.M[1, 0])
    #     M[1, 1] = copy.deepcopy(self.index_pos.M[1, 1])
    #     M[1, 2] = copy.deepcopy(self.index_pos.M[1, 2])
    #     M[2, 0] = copy.deepcopy(self.index_pos.M[2, 0])
    #     M[2, 1] = copy.deepcopy(self.index_pos.M[2, 1])
    #     M[2, 2] = copy.deepcopy(self.index_pos.M[2, 2])
    #     p[0, 0] = copy.deepcopy(self.index_pos.p[0])
    #     p[1, 0] = copy.deepcopy(self.index_pos.p[1])
    #     p[2, 0] = copy.deepcopy(self.index_pos.p[2])
    #     index_in_palm_T[:3, :3] = M
    #     index_in_palm_T[:3, 3] = p
    #
    #     qg.kdl_calc_fk(self.mid_fk, self.mid_qpos, self.mid_pos)
    #     M[0, 0] = copy.deepcopy(self.mid_pos.M[0, 0])
    #     M[0, 1] = copy.deepcopy(self.mid_pos.M[0, 1])
    #     M[0, 2] = copy.deepcopy(self.mid_pos.M[0, 2])
    #     M[1, 0] = copy.deepcopy(self.mid_pos.M[1, 0])
    #     M[1, 1] = copy.deepcopy(self.mid_pos.M[1, 1])
    #     M[1, 2] = copy.deepcopy(self.mid_pos.M[1, 2])
    #     M[2, 0] = copy.deepcopy(self.mid_pos.M[2, 0])
    #     M[2, 1] = copy.deepcopy(self.mid_pos.M[2, 1])
    #     M[2, 2] = copy.deepcopy(self.mid_pos.M[2, 2])
    #     p[0, 0] = copy.deepcopy(self.mid_pos.p[0])
    #     p[1, 0] = copy.deepcopy(self.mid_pos.p[1])
    #     p[2, 0] = copy.deepcopy(self.mid_pos.p[2])
    #     mid_in_palm_T[:3, :3] = M
    #     mid_in_palm_T[:3, 3] = p
    #
    #     qg.kdl_calc_fk(self.ring_fk, self.ring_qpos, self.ring_pos)
    #     M[0, 0] = copy.deepcopy(self.ring_pos.M[0, 0])
    #     M[0, 1] = copy.deepcopy(self.ring_pos.M[0, 1])
    #     M[0, 2] = copy.deepcopy(self.ring_pos.M[0, 2])
    #     M[1, 0] = copy.deepcopy(self.ring_pos.M[1, 0])
    #     M[1, 1] = copy.deepcopy(self.ring_pos.M[1, 1])
    #     M[1, 2] = copy.deepcopy(self.ring_pos.M[1, 2])
    #     M[2, 0] = copy.deepcopy(self.ring_pos.M[2, 0])
    #     M[2, 1] = copy.deepcopy(self.ring_pos.M[2, 1])
    #     M[2, 2] = copy.deepcopy(self.ring_pos.M[2, 2])
    #     p[0, 0] = copy.deepcopy(self.ring_pos.p[0])
    #     p[1, 0] = copy.deepcopy(self.ring_pos.p[1])
    #     p[2, 0] = copy.deepcopy(self.ring_pos.p[2])
    #     ring_in_palm_T[:3, :3] = M
    #     ring_in_palm_T[:3, 3] = p
    #
    #     qg.kdl_calc_fk(self.thumb_fk, self.thumb_qpos, self.thumb_pos)
    #     M[0, 0] = self.thumb_pos.M[0, 0]
    #     M[0, 1] = self.thumb_pos.M[0, 1]
    #     M[0, 2] = self.thumb_pos.M[0, 2]
    #     M[1, 0] = self.thumb_pos.M[1, 0]
    #     M[1, 1] = self.thumb_pos.M[1, 1]
    #     M[1, 2] = self.thumb_pos.M[1, 2]
    #     M[2, 0] = self.thumb_pos.M[2, 0]
    #     M[2, 1] = self.thumb_pos.M[2, 1]
    #     M[2, 2] = self.thumb_pos.M[2, 2]
    #     p[0, 0] = self.thumb_pos.p[0]
    #     p[1, 0] = self.thumb_pos.p[1]
    #     p[2, 0] = self.thumb_pos.p[2]
    #     thumb_in_palm_T[:3, :3] = M
    #     thumb_in_palm_T[:3, 3] = p
    #
    #     return index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T

    def taxel_callback(self, data):
        """Get contact taxels"""
        _index_taxel = data.index_tip_Value
        _mid_taxel = data.middle_tip_Value
        _ring_taxel = data.ring_tip_Value
        _thumb_taxel = data.thumb_tip_Value
        index_taxel = np.ravel(_index_taxel)
        mid_taxel = np.ravel(_mid_taxel)
        ring_taxel = np.ravel(_ring_taxel)
        thumb_taxel = np.ravel(_thumb_taxel)
        index_ids = np.where(index_taxel > self.threshold)[0]
        mid_ids = np.where(mid_taxel > self.threshold)[0]
        ring_ids = np.where(ring_taxel > self.threshold)[0]
        thumb_ids = np.where(thumb_taxel > self.threshold)[0]
        for i in range(index_ids.shape[0]):
            print(" Taxel id:", index_ids)
            index_name_tmp = qg.id2name_tip(index_ids[i], '0')
            pos0, rpy0 = qg.get_taxel_poseuler(index_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            T_taxel_tip0 = qg.get_T_from_posrpy(pos=pos0, rpy=rpy0)
            posquat_tip0 = qg.T2posquat(T_taxel_tip0)
            # T_taxel_P0 = np.matmul(self.index_in_palm_T, T_taxel_tip0)
            # posquat_P0 = qg.T2posquat(T_taxel_P0)  # Index taxel posquat in {P}
            self.pub_A_marker(posquat_taxel=posquat_tip0, frame_name="link_3.0_tip_tactile",
                              marker_id=index_ids[i])  # Publish
            # self.pub_A_marker(posquat_P0)  # Publish
        for i in range(mid_ids.shape[0]):
            mid_name_tmp = qg.id2name_tip(mid_ids[i], '7')
            pos1, rpy1 = qg.get_taxel_poseuler(mid_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            T_taxel_tip1 = qg.get_T_from_posrpy(pos=pos1, rpy=rpy1)
            posquat_tip1 = qg.T2posquat(T_taxel_tip1)
            # T_taxel_P1 = np.matmul(self.mid_in_palm_T, T_taxel_tip1)
            # posquat_P1 = qg.T2posquat(T_taxel_P1)  # Index taxel posquat in {P}
            self.pub_A_marker(posquat_taxel=posquat_tip1, frame_name="link_7.0_tip_tactile",
                              marker_id=mid_ids[i] + 72)  # Publish
            # self.pub_A_marker(posquat_P1)  # Publish
        for i in range(ring_ids.shape[0]):
            ring_name_tmp = qg.id2name_tip(ring_ids[i], '11')
            pos2, rpy2 = qg.get_taxel_poseuler(ring_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            T_taxel_tip2 = qg.get_T_from_posrpy(pos=pos2, rpy=rpy2)
            posquat_tip2 = qg.T2posquat(T_taxel_tip2)
            # T_taxel_P2 = np.matmul(self.ring_in_palm_T, T_taxel_tip2)
            # posquat_P2 = qg.T2posquat(T_taxel_P2)  # Index taxel posquat in {P}
            self.pub_A_marker(posquat_taxel=posquat_tip2, frame_name="link_11.0_tip_tactile",
                              marker_id=ring_ids[i] + 144)  # Publish
            # self.pub_A_marker(posquat_P2)  # Publish
        for i in range(thumb_ids.shape[0]):
            thumb_name_tmp = qg.id2name_tip(thumb_ids[i], '15')
            pos3, rpy3 = qg.get_taxel_poseuler(thumb_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            T_taxel_tip3 = qg.get_T_from_posrpy(pos=pos3, rpy=rpy3)
            posquat_tip3 = qg.T2posquat(T_taxel_tip3)
            # T_taxel_P3 = np.matmul(self.thumb_in_palm_T, T_taxel_tip3)
            # posquat_P3 = qg.T2posquat(T_taxel_P3)  # Index taxel posquat in {P}
            self.pub_A_marker(posquat_taxel=posquat_tip3, frame_name="link_15.0_tip_tactile",
                              marker_id=thumb_ids[i] + 216)  # Publish
            # self.pub_A_marker(posquat_P3)  # Publish

    # def pub_A_marker(self, posquat_taxel_P):
    #     marker = Marker()
    #     marker.header.frame_id = "palm_link"
    #     marker.type = marker.ARROW
    #     marker.action = marker.ADD
    #     marker.scale.x = 0.01
    #     marker.scale.y = 0.002
    #     marker.scale.z = 0.002
    #     marker.color.a = 1
    #     marker.color.r = 0.5
    #     marker.color.g = 0.8
    #     marker.color.b = 0.9
    # 
    #     marker.pose.position.x = posquat_taxel_P[0]
    #     marker.pose.position.y = posquat_taxel_P[1]
    #     marker.pose.position.z = posquat_taxel_P[2]
    #     marker.pose.orientation.x = posquat_taxel_P[3]
    #     marker.pose.orientation.y = posquat_taxel_P[4]
    #     marker.pose.orientation.z = posquat_taxel_P[5]
    #     marker.pose.orientation.w = posquat_taxel_P[6]
    #     self.marker_taxel.publish_marker(marker)

    def pub_A_marker(self, posquat_taxel, frame_name, marker_id):
        marker = Marker()
        marker.id = marker_id
        marker.header.frame_id = frame_name
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.002
        marker.scale.z = 0.002
        marker.color.a = 1
        marker.color.r = 0.5
        marker.color.g = 0.8
        marker.color.b = 0.9
        marker.lifetime = genpy.Duration(0.2)

        marker.pose.position.x = posquat_taxel[0]
        marker.pose.position.y = posquat_taxel[1]
        marker.pose.position.z = posquat_taxel[2]
        marker.pose.orientation.x = posquat_taxel[3]
        marker.pose.orientation.y = posquat_taxel[4]
        marker.pose.orientation.z = posquat_taxel[5]
        marker.pose.orientation.w = posquat_taxel[6]
        self.marker_taxel.publish_marker(marker)

    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()

    def trans_T_W2palm(self, posquat_W):
        T_object_W = trans_from_pos_quat(posquat_W)
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
        T_object_P_90 = np.matmul(T_y_90, T_object_P)  # Rotation
        posquat_P = trans_compute_pos_quan(T_object_P_90)
        return posquat_P


if __name__ == '__main__':
    rospy.init_node('sub_taxel_marker', anonymous=True, log_level=rospy.WARN)
    my_subs = MySubscriber()
    my_subs.loop()
