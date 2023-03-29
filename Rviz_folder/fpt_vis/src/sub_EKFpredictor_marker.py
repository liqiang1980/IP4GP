#!/usr/bin/env python3
import math
import re
import time
import pathlib

import genpy
import numpy as np
import PyKDL as kdl
import copy
from sensor_msgs.msg import JointState
import rospy
from scipy.spatial.transform import Rotation as R
from allegro_tactile_sensor.msg import tactile_msgs
import scipy
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from mujoco_py import load_model_from_path
import xml.etree.ElementTree as ET
import qgFunc as qg
from qgFunc import if_match, get_poseuler_from_xml
from geometry_msgs.msg import TransformStamped
from publisher import *


class ekfDataClass:
    
    """
    Save params for EKF
    xt: Object State in current round
    xt_pre: Object State in last round
    GD: Ground Truth
    F: State transfer matrix from current round to next round
    P: The covariance matrix of initial state, just give it appropriate large values and it will converge automatically.
    H: Observation matrix. That is, the matrix representation of the apparent measurement h().
    """

    def __init__(self):
        self.xt = np.zeros(6)
        self.xtLong = np.zeros(35)
        self.xtLong[34] = 1
        self.xt_pre = np.zeros(6)
        self.xt_posquat = np.zeros(7)
        self.xt_all = np.zeros(6)
        self.GD_all = np.zeros(6)
        self.F = np.mat(np.zeros([35, 35]))
        self.P = np.mat(np.ones([35, 35]) * 100)
        self.P_pre = np.mat(np.zeros([35, 35]))
        self.H = np.mat(np.zeros([12, 35]))
        self.Q = np.mat(np.eye(35))  # Noise parameter matrix of state xt. Parameter adjustment required
        # self.Q[:3, :3] = np.mat([[1, 15, 2],
        #                          [15, 10, 15],
        #                          [2, 15, 1.5]])
        # self.Q_tmp = np.loadtxt('./saveQR/Q.txt')
        # self.Q[:6, :6] = self.Q_tmp
        self.Q = self.Q * 0.001
        # self.R = np.loadtxt('./saveQR/R.txt')
        # self.R = np.mat(np.eye(12))  # Noise parameter matrix of sensors. Parameter adjustment required
        # self.R[:3, :3] = np.mat([[1, 4, 0],
        #                          [4, 3, -1],
        #                          [0, -1, 1]])
        # self.R[3:6, 3:6] = self.R[:3, :3]
        # self.R[6:9, 6:9] = self.R[:3, :3]
        # self.R[9:12, 9:12] = self.R[:3, :3]
        # self.R = self.R
        self.R = np.mat(np.zeros([12, 12]))
        self.pos_O0_pre = np.mat(np.zeros(3)).T
        self.pos_O1_pre = np.mat(np.zeros(3)).T
        self.pos_O2_pre = np.mat(np.zeros(3)).T
        self.pos_O3_pre = np.mat(np.zeros(3)).T
        self.normal_O0_pre = np.mat(np.ones(3)).T
        self.normal_O1_pre = np.mat(np.ones(3)).T
        self.normal_O2_pre = np.mat(np.ones(3)).T
        self.normal_O3_pre = np.mat(np.ones(3)).T

        # test Variables
        self.J_result_all = np.zeros(24)
        self.G_result_all = np.zeros(24)
        self.joint_vel_all = np.zeros(16)
        self.first_flag = 0
        self.test_pos_all = np.mat(np.zeros(3))
        self.test_pos_all2 = np.mat(np.zeros(3))
        self.test_pos_all3 = np.mat(np.zeros(3))
        self.test_normal_all = np.mat(np.zeros(3))
        self.test_normal_all2 = np.mat(np.zeros(3))

        """Get hand pose in {W}"""
        self.wrist_pose = np.eye(4)
        txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/One_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/One_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/Two_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/Two_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/Four_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/Four_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/largeshaker/Four_fin/3/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/One_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/One_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/One_fin/3/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Two_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Two_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Two_fin/3/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Four_fin/1/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Four_fin/2/"
        # txt_dir = "/home/lqg/robotics_data/qg_ws/src/fpt_vis/data/bottle/Four_fin/3/"
        pose = np.loadtxt(txt_dir + 'hand_pose.txt')  # hand pose(posquat)
        self.T_hnd_W = qg.posquat2T(pose)


class MyClass:
    def __init__(self, dataClass, pass_taxel):
        self._dataClass = dataClass
        self.pass_taxel = pass_taxel
        # self.hand_pose = np.loadtxt(self.this_file_dir + txt_dir + "hand_pose.txt")
        self.hand_description = URDF.from_xml_file(
            "/home/lqg/robotics_data/qg_ws/src/fpt_vis/launch/allegro_hand_tactile_v1.4.urdf")  # for Jacobian
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)  # for Jacobian
        # init kdl_JntArrays for each finger
        self.index_qpos = kdl.JntArray(4)
        self.mid_qpos = kdl.JntArray(4)
        self.ring_qpos = kdl.JntArray(4)
        self.thumb_qpos = kdl.JntArray(4)
        kdl.SetToZero(self.index_qpos)
        kdl.SetToZero(self.mid_qpos)
        kdl.SetToZero(self.ring_qpos)
        kdl.SetToZero(self.thumb_qpos)
        # init joint_vels
        self.index_vel = np.zeros(4)
        self.mid_vel = np.zeros(4)
        self.ring_vel = np.zeros(4)
        self.thumb_vel = np.zeros(4)
        # init kdl_Frames for each finger
        self.index_pos = kdl.Frame()  # Construct an identity frame
        self.mid_pos = kdl.Frame()  # Construct an identity frame
        self.ring_pos = kdl.Frame()  # Construct an identity frame
        self.thumb_pos = kdl.Frame()  # Construct an identity frame
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
        # KDLKinematics
        self.index_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "index_tip_tactile")
        self.mid_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "mid_tip_tactile")
        self.ring_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "ring_tip_tactile")
        self.thumb_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "thumb_tip_tactile")
        # Other params
        # self.T_palm_in_W = np.zeros([4, 4])
        # self.T_palm_in_W = qg.posquat2T(self.hand_pose)  # T: palm in world frame {W}

        self.increasing_pos0 = np.mat(np.zeros(3))
        self.increasing_pos1 = np.mat(np.zeros(3))
        self.increasing_pos2 = np.mat(np.zeros(3))
        self.increasing_pos3 = np.mat(np.zeros(3))
        self.increasing_rpy0 = np.mat(np.zeros(3))
        self.increasing_rpy1 = np.mat(np.zeros(3))
        self.increasing_rpy2 = np.mat(np.zeros(3))
        self.increasing_rpy3 = np.mat(np.zeros(3))
        self.increasing_taxel_name0 = np.empty(0)
        self.increasing_taxel_name1 = np.empty(0)
        self.increasing_taxel_name2 = np.empty(0)
        self.increasing_taxel_name3 = np.empty(0)
        self.threshold = 10
        self.qpos_now = np.zeros(16)
        self.qpos_last = np.zeros(16)
        self.time_now = 0
        self.time_last = 0
        self.qvel = np.zeros(16)
        self.qpos_delta = np.zeros(16)
        self.contact_count = np.zeros(4)  # Numbers of contact taxels on 4 fingers
        self.T_obj_W = np.mat(np.eye(4))
        self.GD_obj_W = np.zeros(6)
        self.posquat_GD_obj_W = np.zeros(7)

        self.marker_ekf = EKFPub()
        rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/vicon/object_qg/object_qg', TransformStamped, self.pose_callback)
        rospy.Subscriber('/allegro_tactile', tactile_msgs, self.taxel_callback)

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
        self.contact_count = np.zeros(4)
        for i in range(index_ids.shape[0]):
            self.contact_count[0] += 1
            index_name_tmp = qg.id2name_tip(index_ids[i], '0')
            pos0, rpy0 = qg.get_taxel_poseuler(index_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            self.increasing_taxel_name0 = np.hstack((self.increasing_taxel_name0, index_name_tmp))
            self.increasing_pos0 = np.vstack((self.increasing_pos0, pos0))
            self.increasing_rpy0 = np.vstack((self.increasing_rpy0, rpy0))
        for i in range(mid_ids.shape[0]):
            self.contact_count[1] += 1
            mid_name_tmp = qg.id2name_tip(mid_ids[i], '7')
            pos1, rpy1 = qg.get_taxel_poseuler(mid_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            self.increasing_taxel_name1 = np.hstack((self.increasing_taxel_name1, mid_name_tmp))
            self.increasing_pos1 = np.vstack((self.increasing_pos1, pos1))
            self.increasing_rpy1 = np.vstack((self.increasing_rpy1, rpy1))
        for i in range(ring_ids.shape[0]):
            self.contact_count[2] += 1
            ring_name_tmp = qg.id2name_tip(ring_ids[i], '11')
            pos2, rpy2 = qg.get_taxel_poseuler(ring_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            self.increasing_taxel_name2 = np.hstack((self.increasing_taxel_name2, ring_name_tmp))
            self.increasing_pos2 = np.vstack((self.increasing_pos2, pos2))
            self.increasing_rpy2 = np.vstack((self.increasing_rpy2, rpy2))
        for i in range(thumb_ids.shape[0]):
            self.contact_count[3] += 1
            thumb_name_tmp = qg.id2name_tip(thumb_ids[i], '15')
            pos3, rpy3 = qg.get_taxel_poseuler(thumb_name_tmp)  # The 'euler' in xml are 'rpy' in fact
            self.increasing_taxel_name3 = np.hstack((self.increasing_taxel_name3, thumb_name_tmp))
            self.increasing_pos3 = np.vstack((self.increasing_pos3, pos3))
            self.increasing_rpy3 = np.vstack((self.increasing_rpy3, rpy3))
        self.increasing_pos0 = self.increasing_pos0[1:]
        self.increasing_pos1 = self.increasing_pos1[1:]
        self.increasing_pos2 = self.increasing_pos2[1:]
        self.increasing_pos3 = self.increasing_pos3[1:]
        self.increasing_rpy0 = self.increasing_rpy0[1:]
        self.increasing_rpy1 = self.increasing_rpy1[1:]
        self.increasing_rpy2 = self.increasing_rpy2[1:]
        self.increasing_rpy3 = self.increasing_rpy3[1:]

        self.ekf_process()

    def joint_callback(self, data):
        _all_joint = data.position
        all_joint = np.array(_all_joint)
        self.qpos_now = all_joint
        self.time_now = time.time()
        self.qpos_delta = self.qpos_now - self.qpos_last
        self.qvel = self.qpos_delta / (self.time_now - self.time_last)
        self.qpos_last = self.qpos_now
        self.time_last = self.time_now

        self.index_qpos[0] = _all_joint[0]
        self.index_qpos[1] = _all_joint[1]
        self.index_qpos[2] = _all_joint[2]
        self.index_qpos[3] = _all_joint[3]
        self.mid_qpos[0] = _all_joint[4]
        self.mid_qpos[1] = _all_joint[5]
        self.mid_qpos[2] = _all_joint[6]
        self.mid_qpos[3] = _all_joint[7]
        self.ring_qpos[0] = _all_joint[8]
        self.ring_qpos[1] = _all_joint[9]
        self.ring_qpos[2] = _all_joint[10]
        self.ring_qpos[3] = _all_joint[11]
        self.thumb_qpos[0] = _all_joint[12]
        self.thumb_qpos[1] = _all_joint[13]
        self.thumb_qpos[2] = _all_joint[14]
        self.thumb_qpos[3] = _all_joint[15]

    def pose_callback(self, data):
        object_W_posquat = np.array([data.transform.translation.x,
                                     data.transform.translation.y,
                                     data.transform.translation.z,
                                     data.transform.rotation.x,
                                     data.transform.rotation.y,
                                     data.transform.rotation.z,
                                     data.transform.rotation.w])
        self.posquat_GD_obj_W = object_W_posquat
        self.T_obj_W = qg.posquat2T(object_W_posquat)
        self.GD_obj_W = qg.posquat2posrpy(object_W_posquat)

    def fk_dealer(self):
        """
        Get T (tips in palm) and J by FK method
        joint positions are updated in main_process()
        """
        M = np.mat(np.zeros((3, 3)))
        p = np.zeros([3, 1])
        index_in_palm_T = np.mat(np.eye(4))
        mid_in_palm_T = np.mat(np.eye(4))
        ring_in_palm_T = np.mat(np.eye(4))
        thumb_in_palm_T = np.mat(np.eye(4))

        # forward kinematics
        qg.kdl_calc_fk(self.index_fk, self.index_qpos, self.index_pos)
        M[0, 0] = copy.deepcopy(self.index_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.index_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.index_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.index_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.index_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.index_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.index_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.index_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.index_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.index_pos.p[0])
        p[1, 0] = copy.deepcopy(self.index_pos.p[1])
        p[2, 0] = copy.deepcopy(self.index_pos.p[2])
        index_in_palm_T[:3, :3] = M
        index_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.mid_fk, self.mid_qpos, self.mid_pos)
        M[0, 0] = copy.deepcopy(self.mid_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.mid_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.mid_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.mid_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.mid_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.mid_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.mid_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.mid_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.mid_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.mid_pos.p[0])
        p[1, 0] = copy.deepcopy(self.mid_pos.p[1])
        p[2, 0] = copy.deepcopy(self.mid_pos.p[2])
        mid_in_palm_T[:3, :3] = M
        mid_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.ring_fk, self.ring_qpos, self.ring_pos)
        M[0, 0] = copy.deepcopy(self.ring_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.ring_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.ring_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.ring_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.ring_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.ring_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.ring_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.ring_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.ring_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.ring_pos.p[0])
        p[1, 0] = copy.deepcopy(self.ring_pos.p[1])
        p[2, 0] = copy.deepcopy(self.ring_pos.p[2])
        ring_in_palm_T[:3, :3] = M
        ring_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.thumb_fk, self.thumb_qpos, self.thumb_pos)
        M[0, 0] = self.thumb_pos.M[0, 0]
        M[0, 1] = self.thumb_pos.M[0, 1]
        M[0, 2] = self.thumb_pos.M[0, 2]
        M[1, 0] = self.thumb_pos.M[1, 0]
        M[1, 1] = self.thumb_pos.M[1, 1]
        M[1, 2] = self.thumb_pos.M[1, 2]
        M[2, 0] = self.thumb_pos.M[2, 0]
        M[2, 1] = self.thumb_pos.M[2, 1]
        M[2, 2] = self.thumb_pos.M[2, 2]
        p[0, 0] = self.thumb_pos.p[0]
        p[1, 0] = self.thumb_pos.p[1]
        p[2, 0] = self.thumb_pos.p[2]
        thumb_in_palm_T[:3, :3] = M
        thumb_in_palm_T[:3, 3] = p

        return index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T

    def Jacobian_dealer(self, contact_count, qpos, index_taxel_name, mid_taxel_name,
                        ring_taxel_name, thumb_taxel_name):
        jstate = qpos
        J_index = np.mat(np.zeros([6, 4]))
        J_mid = np.mat(np.zeros([6, 4]))
        J_ring = np.mat(np.zeros([6, 4]))
        J_thumb = np.mat(np.zeros([6, 4]))
        if contact_count[0] != 0:
            print("\n KDL_chain: palm_link to " + index_taxel_name)
            index_kdl_kin = KDLKinematics(self.hand_description, "palm_link", index_taxel_name)
            J_index = copy.deepcopy(index_kdl_kin.jacobian(q=jstate[0:4]))
            J_index = np.mat(J_index)
            print(" J_index:", J_index, "\n")
        if contact_count[1] != 0:
            print("\n KDL_chain: palm_link to " + mid_taxel_name)
            mid_kdl_kin = KDLKinematics(self.hand_description, "palm_link", mid_taxel_name)
            J_mid = copy.deepcopy(mid_kdl_kin.jacobian(q=jstate[4:8]))
            J_mid = np.mat(J_mid)
            print(" J_mid:", J_mid, "\n")
        if contact_count[2] != 0:
            print("\n KDL_chain: palm_link to " + ring_taxel_name)
            ring_kdl_kin = KDLKinematics(self.hand_description, "palm_link", ring_taxel_name)
            J_ring = copy.deepcopy(ring_kdl_kin.jacobian(q=jstate[8:12]))
            J_ring = np.mat(J_ring)
            print(" J_ring:", J_ring, "\n")
        if contact_count[3] != 0:
            print("\n KDL_chain: palm_link to " + thumb_taxel_name)
            thumb_kdl_kin = KDLKinematics(self.hand_description, "palm_link", thumb_taxel_name)
            J_thumb = copy.deepcopy(thumb_kdl_kin.jacobian(q=jstate[12:16]))
            J_thumb = np.mat(J_thumb)
            print(" J_thumb:", J_thumb, "\n")

        return J_index, J_mid, J_ring, J_thumb

    def ekf_process(self):
        if np.sum(self.contact_count) == 0:  # No contact
            self._dataClass.xt = self.GD_obj_W
            self.pub_A_marker(posquat_ekf_xt=self.posquat_GD_obj_W)
        else:  # If any contact point exists, predict.
            """
            Change frame for GD. {W} to {P}.
            """
            # object_R = R.from_quat(object_posquat[3:]).as_dcm()
            # object_T = np.mat(np.eye(4))
            # object_T[:3, :3] = object_R
            # object_T[:3, 3] = np.mat(object_posquat[:3]).T
            # print(self.T_palm_in_W)
            # object_T_P = np.matmul(np.linalg.inv(self.T_palm_in_W), object_T)
            # object_posrpy[:3] = np.ravel(object_T_P[:3, 3].T)
            # object_posrpy[3:] = R.from_dcm(object_T_P[:3, :3]).as_euler('xyz')

            """
            Exclude tips which is not contact
            """
            for k in range(4):
                if self.contact_count[k] == 0:
                    self.qvel[k * 4:k * 4 + 4] = np.zeros(4)
                    self.qpos_delta[k * 4:k * 4 + 4] = np.zeros(4)

            """
            Choose taxel_name, pos, euler. They are all in the tip frame.
            """
            _mean_pos0 = qg.get_mean_3D(self.increasing_pos0)
            _mean_pos1 = qg.get_mean_3D(self.increasing_pos1)
            _mean_pos2 = qg.get_mean_3D(self.increasing_pos2)
            _mean_pos3 = qg.get_mean_3D(self.increasing_pos3)
            _mean_rpy0 = qg.get_mean_3D(self.increasing_rpy0)
            _mean_rpy1 = qg.get_mean_3D(self.increasing_rpy1)
            _mean_rpy2 = qg.get_mean_3D(self.increasing_rpy2)
            _mean_rpy3 = qg.get_mean_3D(self.increasing_rpy3)

            chosen_pos0 = np.zeros(3)
            chosen_pos1 = np.zeros(3)
            chosen_pos2 = np.zeros(3)
            chosen_pos3 = np.zeros(3)
            chosen_rpy0 = np.zeros(3)
            chosen_rpy1 = np.zeros(3)
            chosen_rpy2 = np.zeros(3)
            chosen_rpy3 = np.zeros(3)
            index_taxel_name = ''
            mid_taxel_name = ''
            ring_taxel_name = ''
            thumb_taxel_name = ''

            if self.increasing_pos0.size != 0:
                chosen_pos0, loc0 = qg.choose_taxel(_mean_pos0, self.increasing_pos0)
                chosen_rpy0 = self.increasing_rpy0[loc0]
                index_taxel_name = self.increasing_taxel_name0[loc0]
            if self.increasing_pos1.size != 0:
                chosen_pos1, loc1 = qg.choose_taxel(_mean_pos1, self.increasing_pos1)
                chosen_rpy1 = self.increasing_rpy1[loc1]
                mid_taxel_name = self.increasing_taxel_name1[loc1]
            if self.increasing_pos2.size != 0:
                chosen_pos2, loc2 = qg.choose_taxel(_mean_pos2, self.increasing_pos2)
                chosen_rpy2 = self.increasing_rpy2[loc2]
                ring_taxel_name = self.increasing_taxel_name2[loc2]
            if self.increasing_pos3.size != 0:
                chosen_pos3, loc3 = qg.choose_taxel(_mean_pos3, self.increasing_pos3)
                chosen_rpy3 = self.increasing_rpy3[loc3]
                thumb_taxel_name = self.increasing_taxel_name3[loc3]

            """
            FK: Get T (tips in palm frame {P})
            """
            index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T = self.fk_dealer()

            """Exclude T of tips which is not contact"""
            if self.contact_count[0] == 0:
                index_in_palm_T = np.mat(np.zeros([4, 4]))
            if self.contact_count[1] == 0:
                mid_in_palm_T = np.mat(np.zeros([4, 4]))
            if self.contact_count[2] == 0:
                ring_in_palm_T = np.mat(np.zeros([4, 4]))
            if self.contact_count[3] == 0:
                thumb_in_palm_T = np.mat(np.zeros([4, 4]))

            T_taxel_in_palm0, T_taxel_in_W0 = self.get_T_taxel(pos=chosen_pos0, rpy=chosen_rpy0,
                                                               T_tip_in_palm=index_in_palm_T)
            T_taxel_in_palm1, T_taxel_in_W1 = self.get_T_taxel(pos=chosen_pos1, rpy=chosen_rpy1,
                                                               T_tip_in_palm=mid_in_palm_T)
            T_taxel_in_palm2, T_taxel_in_W2 = self.get_T_taxel(pos=chosen_pos2, rpy=chosen_rpy2,
                                                               T_tip_in_palm=ring_in_palm_T)
            T_taxel_in_palm3, T_taxel_in_W3 = self.get_T_taxel(pos=chosen_pos3, rpy=chosen_rpy3,
                                                               T_tip_in_palm=thumb_in_palm_T)
            J_index, J_mid, J_ring, J_thumb = self.Jacobian_dealer(self.contact_count, self.qpos_now,
                                                                   index_taxel_name, mid_taxel_name,
                                                                   ring_taxel_name, thumb_taxel_name)

            """
            Get Grasp Matrix G in round i
            """
            big_G = np.mat(np.zeros([6, 24]))
            p = self._dataClass.xt[:3]  # p: The position of object in {W}. Get it from prediction.

            # =======Index Finger:
            R0 = T_taxel_in_W0[:3, :3]  # R: The rotation mat of taxel in world frame {W}
            c0 = np.ravel(T_taxel_in_W0[:3, 3].T)  # ci: The position of taxel in {W}.
            G0, S0 = qg.get_G(R0, p, c0)
            big_G[:, :6] = G0

            # =======Mid Finger:
            R1 = T_taxel_in_W1[:3, :3]  # R: The rotation mat of taxel in world frame {W}
            c1 = np.ravel(T_taxel_in_W1[:3, 3].T)  # ci: The position of taxel in {W}.
            G1, S1 = qg.get_G(R1, p, c1)
            big_G[:, 6:12] = G1

            # =======Ring Finger:
            R2 = T_taxel_in_W2[:3, :3]  # R: The rotation mat of taxel in world frame {W}
            c2 = np.ravel(T_taxel_in_W2[:3, 3].T)  # ci: The position of taxel in {W}.
            G2, S2 = qg.get_G(R2, p, c2)
            big_G[:, 12:18] = G2

            # =======Thumb Finger:
            R3 = T_taxel_in_W3[:3, :3]  # R: The rotation mat of taxel in world frame {W}
            c3 = np.ravel(T_taxel_in_W3[:3, 3].T)  # ci: The position of taxel in {W}.
            G3, S3 = qg.get_G(R3, p, c3)
            big_G[:, 18:] = G3

            """
            Get Jacobian Matrix J in round i
            (Temporarily use J_tips instead of J_taxels)
            """

            """
            Change taxels of tips from {W} to {C}
            """
            test_R0 = np.mat(np.eye(6))
            test_R1 = np.mat(np.eye(6))
            test_R2 = np.mat(np.eye(6))
            test_R3 = np.mat(np.eye(6))
            big_J = np.mat(np.zeros([24, 16]))

            # Test 1:
            # test_R0[:3, :3] = T_taxel_in_palm0[:3, :3]
            # test_R0[3:, 3:] = T_taxel_in_palm0[:3, :3]
            # test_R1[:3, :3] = T_taxel_in_palm1[:3, :3]
            # test_R1[3:, 3:] = T_taxel_in_palm1[:3, :3]
            # test_R2[:3, :3] = T_taxel_in_palm2[:3, :3]
            # test_R2[3:, 3:] = T_taxel_in_palm2[:3, :3]
            # test_R3[:3, :3] = T_taxel_in_palm3[:3, :3]
            # test_R3[3:, 3:] = T_taxel_in_palm3[:3, :3]

            # Test 2:
            # test_R0[:3, :3] = T_taxel_in_W0[:3, :3]
            # test_R0[3:, 3:] = T_taxel_in_W0[:3, :3]
            # test_R1[:3, :3] = T_taxel_in_W1[:3, :3]
            # test_R1[3:, 3:] = T_taxel_in_W1[:3, :3]
            # test_R2[:3, :3] = T_taxel_in_W2[:3, :3]
            # test_R2[3:, 3:] = T_taxel_in_W2[:3, :3]
            # test_R3[:3, :3] = T_taxel_in_W3[:3, :3]
            # test_R3[3:, 3:] = T_taxel_in_W3[:3, :3]

            # Test 3:
            # test_R0[:3, :3] = self.T_palm_in_W[:3, :3]
            # test_R0[3:, 3:] = self.T_palm_in_W[:3, :3]
            # test_R1 = test_R0
            # test_R2 = test_R0
            # test_R3 = test_R0

            big_J[:6, :4] = np.matmul(test_R0.T, J_index)
            big_J[6:12, 4:8] = np.matmul(test_R1.T, J_mid)
            big_J[12:18, 8:12] = np.matmul(test_R2.T, J_ring)
            big_J[18:, 12:] = np.matmul(test_R3.T, J_thumb)

            """
            Prediction
            """
            # F_part = np.dot(np.linalg.pinv(big_G).T, big_J) * delta_time
            F_part = np.dot(np.linalg.pinv(big_G.T), big_J)
            # Prediction = np.dot(F_part, joint_vel)
            Prediction = (F_part * np.mat(self.qpos_delta).T).T
            # self._dataClass.xt = np.ravel(self._dataClass.xt + Prediction)  # x_t: object poseuler in {W}
            self._dataClass.xt = np.ravel(self._dataClass.xt - Prediction)  # x_t: object poseuler in {W}
            # self._dataClass.xtLong[:6] = self._dataClass.xt
            # self._dataClass.xtLong[6:22] = joint_vel

            T_obj_W = qg.posrpy2T(self._dataClass.xt)
            T_obj_P = np.matmul(np.linalg.inv(self._dataClass.T_hnd_W), T_obj_W)
            posquat_ekf_xt = qg.T2posquat(T_obj_P)
            self.pub_A_marker(posquat_ekf_xt=posquat_ekf_xt)

            # self.posterior(T_object_W, T_taxel_in_W0, T_taxel_in_W1, T_taxel_in_W2, T_taxel_in_W3, F_part, tip_flag_outer)

    # def posterior(self, T_O_W, T0, T1, T2, T3, F_part, tip_flag_outer):
    #     """
    #     input: T_O_W, T_object in world frame {W}.
    #     input: T0~T3, T_taxel in world frame {W}.
    #     input: pos0~pos3, taxel positions in world frame {W}.
    #     EKF posterior update.
    #     """
    #     # self._dataClass.F = qg.get_bigF(F_part=F_part)
    #     #
    #     # self._dataClass.P = qg.matmul3mat(self._dataClass.F, self._dataClass.P_pre,
    #     #                                   self._dataClass.F.T) + self._dataClass.Q
    #
    #     """
    #     Use taxels in last round and object pose in current round as h()
    #     Use taxels in current round and object pose in current round as z
    #     calculate error = z - h()
    #     """
    #     R_O_W = T_O_W[:3, :3]  # R_object in {W} in current round
    #     pos_object_W = T_O_W[:3, 3]
    #     T_taxel_O0 = np.dot(np.linalg.inv(T_O_W), T0)  # T_taxel in {O} in current round
    #     T_taxel_O1 = np.dot(np.linalg.inv(T_O_W), T1)  # T_taxel in {O} in current round
    #     T_taxel_O2 = np.dot(np.linalg.inv(T_O_W), T2)  # T_taxel in {O} in current round
    #     T_taxel_O3 = np.dot(np.linalg.inv(T_O_W), T3)  # T_taxel in {O} in current round
    #     pos_taxel_O0 = T_taxel_O0[:3, 3]  # taxel pos in {O} in current round
    #     pos_taxel_O1 = T_taxel_O1[:3, 3]  # taxel pos in {O} in current round
    #     pos_taxel_O2 = T_taxel_O2[:3, 3]  # taxel pos in {O} in current round
    #     pos_taxel_O3 = T_taxel_O3[:3, 3]  # taxel pos in {O} in current round
    #     normal_O0 = qg.get_normal_O(pos_taxel_O0)  # taxel normal in {O} in current round
    #     normal_O1 = qg.get_normal_O(pos_taxel_O1)  # taxel normal in {O} in current round
    #     normal_O2 = qg.get_normal_O(pos_taxel_O2)  # taxel normal in {O} in current round
    #     normal_O3 = qg.get_normal_O(pos_taxel_O3)  # taxel normal in {O} in current round
    #
    #     pos_taxel_W0_h = np.matmul(R_O_W, self._dataClass.pos_O0_pre)  # last taxel in current object, taxel pos in {W}
    #     pos_taxel_W1_h = np.matmul(R_O_W, self._dataClass.pos_O1_pre)  # last taxel in current object, taxel pos in {W}
    #     pos_taxel_W2_h = np.matmul(R_O_W, self._dataClass.pos_O2_pre)  # last taxel in current object, taxel pos in {W}
    #     pos_taxel_W3_h = np.matmul(R_O_W, self._dataClass.pos_O3_pre)  # last taxel in current object, taxel pos in {W}
    #     normal_W0_h = np.matmul(R_O_W,
    #                             self._dataClass.normal_O0_pre)  # last taxel in current object, taxel normal in {W}
    #     normal_W1_h = np.matmul(R_O_W,
    #                             self._dataClass.normal_O1_pre)  # last taxel in current object, taxel normal in {W}
    #     normal_W2_h = np.matmul(R_O_W,
    #                             self._dataClass.normal_O2_pre)  # last taxel in current object, taxel normal in {W}
    #     normal_W3_h = np.matmul(R_O_W,
    #                             self._dataClass.normal_O3_pre)  # last taxel in current object, taxel normal in {W}
    #
    #     # Save pos & normal as last round pos & normal
    #     self._dataClass.pos_O0_pre = pos_taxel_O0
    #     self._dataClass.pos_O1_pre = pos_taxel_O1
    #     self._dataClass.pos_O2_pre = pos_taxel_O2
    #     self._dataClass.pos_O3_pre = pos_taxel_O3
    #     self._dataClass.normal_O0_pre = normal_O0
    #     self._dataClass.normal_O1_pre = normal_O1
    #     self._dataClass.normal_O2_pre = normal_O2
    #     self._dataClass.normal_O3_pre = normal_O3
    #
    #     # Calculate measurement vector z
    #     pos_taxel_W0 = T0[:3, 3]  # taxel pos in {W} in current round
    #     pos_taxel_W1 = T1[:3, 3]  # taxel pos in {W} in current round
    #     pos_taxel_W2 = T2[:3, 3]  # taxel pos in {W} in current round
    #     pos_taxel_W3 = T3[:3, 3]  # taxel pos in {W} in current round
    #     normal_W0_a = np.matmul(R_O_W, normal_O0)  # taxel normal in {W} in current round
    #     normal_W1_a = np.matmul(R_O_W, normal_O1)  # taxel normal in {W} in current round
    #     normal_W2_a = np.matmul(R_O_W, normal_O2)  # taxel normal in {W} in current round
    #     normal_W3_a = np.matmul(R_O_W, normal_O3)  # taxel normal in {W} in current round
    #     """
    #     Make normal_z and normal_h on the same scale
    #     """
    #     normal_W0 = normal_W0_a * math.sqrt(normal_W0_a[0] ** 2 + normal_W0_a[1] ** 2 + normal_W0_a[2] ** 2) \
    #                 / math.sqrt(normal_W0_h[0] ** 2 + normal_W0_h[1] ** 2 + normal_W0_h[2] ** 2)
    #     normal_W1 = normal_W1_a * math.sqrt(normal_W1_a[0] ** 2 + normal_W1_a[1] ** 2 + normal_W1_a[2] ** 2) \
    #                 / math.sqrt(normal_W1_h[0] ** 2 + normal_W1_h[1] ** 2 + normal_W1_h[2] ** 2)
    #     normal_W2 = normal_W2_a * math.sqrt(normal_W2_a[0] ** 2 + normal_W2_a[1] ** 2 + normal_W2_a[2] ** 2) \
    #                 / math.sqrt(normal_W2_h[0] ** 2 + normal_W2_h[1] ** 2 + normal_W2_h[2] ** 2)
    #     normal_W3 = normal_W3_a * math.sqrt(normal_W3_a[0] ** 2 + normal_W3_a[1] ** 2 + normal_W3_a[2] ** 2) \
    #                 / math.sqrt(normal_W3_h[0] ** 2 + normal_W3_h[1] ** 2 + normal_W3_h[2] ** 2)
    #     z0 = qg.get_z_or_h('z0', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3,
    #                        normal_W0_a, normal_W1_a, normal_W2_a, normal_W3_a)
    #     z = qg.get_z_or_h('z', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3,
    #                       normal_W0, normal_W1, normal_W2, normal_W3)
    #     test_z = qg.get_z_or_h2('test_z', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3)
    #
    #     """
    #     Set pos_taxel in {O} just once. Assume that there is not any sliding.
    #     """
    #     if not self._dataClass.first_flag:
    #         self._dataClass.xtLong[22:25] = pos_taxel_O0.T
    #         self._dataClass.xtLong[25:28] = pos_taxel_O1.T
    #         self._dataClass.xtLong[28:31] = pos_taxel_O2.T
    #         self._dataClass.xtLong[31:34] = pos_taxel_O3.T
    #         self._dataClass.first_flag = 1
    #     self._dataClass.H = qg.get_bigH2(R_O_W)  # 12*35
    #     H = self._dataClass.H
    #     # print("     $$ mat_H:", H)
    #
    #     # test H by index finger
    #     # pos_taxel_W, normal_taxel_W = qg.testH(self._dataClass.xtLong, pos_object_W, pos_taxel_W0_h, R_O_W)
    #     # pos_taxel_W = np.ravel(pos_taxel_W)
    #     # normal_taxel_W = np.ravel(normal_taxel_W)
    #     # self._dataClass.test_pos_all = np.vstack((self._dataClass.test_pos_all, pos_taxel_W))
    #     # self._dataClass.test_pos_all3 = np.vstack((self._dataClass.test_pos_all3, np.ravel(pos_taxel_W0_h)))
    #     # self._dataClass.test_normal_all = np.vstack((self._dataClass.test_normal_all, normal_taxel_W))
    #     # self._dataClass.test_normal_all2 = np.vstack((self._dataClass.test_normal_all2, normal_W0_h.T))
    #
    #     test_h = np.matmul(H, self._dataClass.xtLong).T
    #     print("### test: h = H * xt (.T):", test_h.T)
    #     h = qg.get_z_or_h('h', pos_taxel_W0_h, pos_taxel_W1_h, pos_taxel_W2_h, pos_taxel_W3_h,
    #                       normal_W0_h, normal_W1_h, normal_W2_h, normal_W3_h)
    #
    #     error = test_z - test_h
    #     """
    #     Exclude tips which is not contact
    #     """
    #     for k in range(4):
    #         if tip_flag_outer[k] == 0:
    #             error[k * 3:k * 3 + 3, 0] = np.mat(np.zeros(3)).T
    #     print("$$ error:", error.T)
    #
    #     """
    #     The remaining steps of posterior EKF
    #     """
    #     S = qg.matmul3mat(H, self._dataClass.P, H.T) + self._dataClass.R
    #     K = qg.matmul3mat(self._dataClass.P, H.T, np.linalg.inv(S))
    #     print("     $$ mat_S, mat_K:", S.shape, K.shape)
    #     print("     $$ mat_P:", self._dataClass.P)
    #
    #     Update = np.matmul(K, error).T
    #     # print("    >>Posterior Update", Update.shape)
    #     self._dataClass.xtLong = self._dataClass.xtLong - Update
    #     self._dataClass.xtLong = np.ravel(self._dataClass.xtLong)
    #     self._dataClass.xtLong[34] = 1
    #     self._dataClass.xt = self._dataClass.xtLong[:6]
    #     # print("   ??after update???xt:", self._dataClass.xt, self._dataClass.xtLong)
    #     self._dataClass.P_pre = np.matmul((np.mat(np.eye(35)) - np.matmul(K, H)), self._dataClass.P)

    def get_mean_3D(self, input_mat):
        mean_vac = np.zeros(3)
        if input_mat.size != 0:
            mean_vac = np.mean(input_mat, axis=0)
        return mean_vac

    def get_T_taxel(self, pos, rpy, T_tip_in_palm):
        """
        Translate posrpy to T
        """
        R_taxl_in_tip = qg.rpy2R(rpy)  # Taxel in tip
        T_taxel_in_tip = np.mat(np.eye(4))
        T_taxel_in_tip[:3, :3] = R_taxl_in_tip
        T_taxel_in_tip[:3, 3] = np.mat(pos).T
        T_taxel_in_palm = np.dot(T_tip_in_palm, T_taxel_in_tip)
        T_taxel_in_W = np.matmul(self._dataClass.T_hnd_W, T_taxel_in_palm)
        return T_taxel_in_palm, T_taxel_in_W


    def pub_A_marker(self, posquat_ekf_xt):
        marker = Marker()
        # marker.id = 0
        marker.header.frame_id = "palm_link"
        # marker.header.frame_id = "base_link"
        marker.type = marker.MESH_RESOURCE
        marker.mesh_resource = "package://fpt_vis/largeshaker.STL"
        marker.action = marker.ADD
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        """Gray: 0.5 0.5 0.5"""
        marker.color.a = 0.8
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.pose.position.x = posquat_ekf_xt[0]
        marker.pose.position.y = posquat_ekf_xt[1]
        marker.pose.position.z = posquat_ekf_xt[2]
        marker.pose.orientation.x = posquat_ekf_xt[3]
        marker.pose.orientation.y = posquat_ekf_xt[4]
        marker.pose.orientation.z = posquat_ekf_xt[5]
        marker.pose.orientation.w = posquat_ekf_xt[6]
        self.marker_ekf.publish_marker(marker)


    def loop(self):
        rospy.logwarn("Starting Loop...")
        rospy.spin()


if __name__ == '__main__':
    invalid_taxel = np.array(
        ['touch_0_4_9', 'touch_0_4_8', 'touch_0_4_7', 'touch_0_5_6', 'touch_2_1_1', 'touch_9_1_1', 'touch_13_1_1',
         'touch_16_1_1'])
    rospy.init_node('sub_EKF_predictor', anonymous=True, log_level=rospy.WARN)

    ekf_data = ekfDataClass()
    my_class = MyClass(dataClass=ekf_data, pass_taxel=invalid_taxel)
    my_class.loop()
