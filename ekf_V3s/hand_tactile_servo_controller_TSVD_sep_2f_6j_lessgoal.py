#!/usr/bin/env python

# system
from __future__ import print_function
from os import listdir
import pathlib
from numpy import argmin, diag
from scipy.optimize._lsq.least_squares import prepare_bounds
from time import sleep, time
import threading
import sys
import copy


# math
import numpy as np
np.set_printoptions(suppress=True)
import math
from scipy.optimize.nonlin import Jacobian
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from scipy import odr
import pandas
import csv
import matplotlib.pyplot as plt

# ROS
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics


"""
import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list
"""


# self-defined functions
from lhz_func import R2RR, T2pose, find_vec_from_text, wxyz2xyzw, xyzw2wxyz, T2pos_euler, KDLframe2T, average_quaternions, pose_quat2new_ros_pose, ros_pose2new_pose_quat, kdl_calc_fk, kdl_finger_twist2qvel, angular_vel_from_R, angular_vel_from_quats, cross_product_matrix_from_vector3d, quats2delta_angle_degrees, quats2delta_angle, T2AdT, pR2AdT, delta_pose2next_pose, delta_pose2next_T, pose2T, StreamingMovingAverage_array, array_append, average_quaternions, pinv_SVD, pinv_TSVD, invT, unit_orthogonal_vector3d


# self-defined msg
from allegro_ur_ctrl_cmd.msg import allegro_hand_joints_cmd, ur_ee_cmd
from allegro_tactile_sensor.msg import tactile_msgs

def force_error_tobe_minimized(para, x, real_f):
    _k_e = 500.0
    """
    self._k_e is defined as a static float with the class
    px_in_tip    = x[0, 0]
    py_in_tip    = x[0, 1]
    pz_in_tip    = x[0, 2]
    psi_in_tip   = x[0, 3]
    theta_in_tip = x[0, 4]
    phi_in_tip   = x[0, 5]

    depth_in_c = para[0, 0]
    theta_in_c = para[0, 1]
    phi_in_c = para[0, 2]
    """
    # print x[0,0], para[1]
    depth = para[0] + math.cos(para[1])*math.cos(para[2])*x[0,0] + math.cos(para[1])*math.sin(para[2])*x[0,1] - math.sin(para[1])*x[0,2]
    area = math.cos(x[0,4]-para[1])*math.cos(x[0,5]-para[2])
    return real_f - _k_e*depth*area

def detect_contact(tip_tacxel_pose_euler, tip_tactile_array, tac_threhold=5.0):
        x_set = np.zeros((0, 6))
        y_set = np.zeros((0, 1))
        for i, tac_i in enumerate(tip_tactile_array):
                if tac_i > tac_threhold:
                    x_set = np.vstack((x_set, tip_tacxel_pose_euler[i, :])) # don't need to copy() here, vstack() copies the full arrays
                    y_set = np.vstack((y_set, tac_i))
        y_set = np.array(y_set).flatten()
        if np.size(y_set) > 0:
            # print("Tactile values:", y_set)
            return True, x_set, y_set
        else:
            return False, x_set, y_set

def calc_pose_in_tip(x_set, y_set):
    """ Calculate the pose of the contacting surface with respect to the tip (e.t. thumb tip, mid tip, and so on)."""
    pose_in_tip = np.zeros(6) # [x, y, z, rx, ry, rz]
    ls_result = least_squares(force_error_tobe_minimized, np.array([0.,0.,-0.07]), bounds=(-np.pi/6.0, np.pi/6.0), method='dogbox', args=(x_set, y_set)) # original
    pose_in_tip[4] = ls_result.x[1] # rotation angle about the y-axis (surface in tip)
    pose_in_tip[5] = ls_result.x[2] # rotation angle about the z-axis (surface in tip)
    x_in_s = ls_result.x[0] # x_tip_in_surface
    pose_in_tip[0] = x_in_s*np.cos(pose_in_tip[4])*np.cos(pose_in_tip[5])
    pose_in_tip[1] = x_in_s*np.cos(pose_in_tip[4])*np.sin(pose_in_tip[5])
    pose_in_tip[2] = -x_in_s*np.sin(pose_in_tip[4])
    # It is obvious that pose_in_tip[3] is always equal to 0.
    return pose_in_tip

class MyController(object):
    # pose estimation: static variable _k_e
    _k_e = 500.0
    # For subscribe and calculating
    def __init__(self):
        self.this_file_dir = str(pathlib.Path(__file__).parent.absolute())
        print("This script dir:", self.this_file_dir)
        if len(sys.argv) == 2:
            if sys.argv[1] == "sim":
                self.sim_flag = True
                print("Starting calculations for simulation ...")
            else:
                raise ValueError("Unexpected argument! If sim, argument should be \"sim\", else, type nothing")
        elif len(sys.argv) == 1:
            self.sim_flag = False
            print("Starting calculations for real machine ...")
        else:
            raise ValueError("Number of arguments should be 0 or 1.")
        self.index_tacxel_pose = np.loadtxt(self.this_file_dir+"/allegro_tactile_description/index_tacxel_pose_20211220.txt")
        self.index_tacxel_pose_euler = np.zeros((np.size(self.index_tacxel_pose,0),6))
        for i in range(np.size(self.index_tacxel_pose,0)):
            self.index_tacxel_pose_euler[i,:3] = self.index_tacxel_pose[i,:3].copy()
            self.index_tacxel_pose_euler[i,3:] = Rotation.from_quat(self.index_tacxel_pose[i,3:]).as_euler("XYZ", degrees=False)

        # CMD
        self.joint_pos_cmd = np.zeros(16)
        self.ee_pos_cmd = np.zeros(3)
        self.ee_quat_cmd = np.zeros(4)
        # allegro hand joints cmd
        self.allegro_publisher = rospy.Publisher('/allegro_joints_cmd', allegro_hand_joints_cmd, queue_size=10) # Think twice about the queue_size, it should be determined by control rate
        self.jcmd_pub = allegro_hand_joints_cmd()
        self.jcmd_pub.header.seq = 0
        self.jcmd_pub.header.stamp = rospy.Time.now()
        # ur endpoint pose cmd
        self.ur_publisher = rospy.Publisher('/ur_eepose_cmd', ur_ee_cmd, queue_size=10) # Think twice about the queue_size, it should be determined by control rate
        self.urcmd_pub = ur_ee_cmd()
        self.urcmd_pub.header.seq = 0
        self.urcmd_pub.header.stamp = rospy.Time.now()

        # publisher of twist calculated by forward Jacobian:
        # self.index_twist_publisher = rospy.Publisher('/index_twist', ur_ee_cmd, queue_size=10) # Think twice about the queue_size, it should be determined by control rate
        # self.index_twist_pub = ur_ee_cmd()
        # self.thumb_twist_publisher = rospy.Publisher('/thumb_twist', ur_ee_cmd, queue_size=10) # Think twice about the queue_size, it should be determined by control rate
        # self.thumb_twist_pub = ur_ee_cmd()
        self.index_vel_pub = ur_ee_cmd()
        self.index_vel_publisher = rospy.Publisher('/index_vel', ur_ee_cmd, queue_size=10)
        self.index_angvel_pub = ur_ee_cmd()
        self.index_angvel_publisher = rospy.Publisher('/index_angvel', ur_ee_cmd, queue_size=10)
        self.thumb_vel_pub = ur_ee_cmd()
        self.thumb_vel_publisher = rospy.Publisher('/thumb_vel', ur_ee_cmd, queue_size=10)
        self.thumb_angvel_pub = ur_ee_cmd()
        self.thumb_angvel_publisher = rospy.Publisher('/thumb_angvel', ur_ee_cmd, queue_size=10)

        # Allegro joint limits
        self.index_reach_limit = False
        self.mid_reach_limit = False
        self.ring_reach_limit = False
        self.thumb_reach_limit = False
        self.index_weights = np.mat(np.zeros((4,1)))
        self.mid_weights = np.mat(np.zeros((4,1)))
        self.ring_weights = np.mat(np.zeros((4,1)))
        self.thumb_weights = np.mat(np.zeros((4,1)))
        self.j_limits = np.mat(np.loadtxt(self.this_file_dir+"/allegro_tactile_description/allegro_joint_limits_20220225.txt"))
        self.index_limits = self.j_limits[:,0:4]
        self.mid_limits = self.j_limits[:,4:8]
        self.ring_limits = self.j_limits[:,8:12]
        self.thumb_limits = self.j_limits[:,12:16]
        self.index_ranges = np.mat(np.zeros((1,4)))
        self.mid_ranges = np.mat(np.zeros((1,4)))
        self.ring_ranges = np.mat(np.zeros((1,4)))
        self.thumb_ranges = np.mat(np.zeros((1,4)))
        self.index_ranges = self.index_limits[0,:] - self.index_limits[1,:]
        self.mid_ranges = self.mid_limits[0,:] - self.mid_limits[1,:]
        self.ring_ranges = self.ring_limits[0,:] - self.ring_limits[1,:]
        self.thumb_ranges = self.thumb_limits[0,:] - self.thumb_limits[1,:]
        self.index_ranges_sq = self.index_ranges
        self.mid_ranges_sq = self.mid_ranges
        self.ring_ranges_sq = self.ring_ranges
        self.thumb_ranges_sq = self.thumb_ranges
        for i in range(4):
            self.index_ranges_sq[0,i] = (self.index_ranges[0,i]**2)/4.0
            self.mid_ranges_sq[0,i] = (self.mid_ranges[0,i]**2)/4.0
            self.ring_ranges_sq[0,i] = (self.ring_ranges[0,i]**2)/4.0
            self.thumb_ranges_sq[0,i] = (self.thumb_ranges[0,i]**2)/4.0

        # init callback variables
        self.jstate_seq = 0
        self.jstate_secs = 0
        self.jstate_nsecs = 0
        self.eestate_seq = 0
        self.eestate_secs = 0
        self.eestate_nsecs = 0
        self.allegro_jstates = np.zeros(16)
        self.ur_eestates = np.zeros(7)
        self.jstate_dt = 0.0
        self.jstate_t0 = 0.0
        self.jstate_tt = 0.0
        self.urstate_dt = 0.0
        self.urstate_t0 = 0.0
        self.urstate_tt = 0.0
        self.index_tip = np.zeros(72)
        self.mid_tip = np.zeros(72)
        self.ring_tip = np.zeros(72)
        self.thumb_tip = np.zeros(72)
        self.current_time = time()

        # Timing and cmd, states init
        while not rospy.is_shutdown():
            print("Waiting for the first message from simulation...")
            jmsg = rospy.wait_for_message('/allegro_joints_state', allegro_hand_joints_cmd)
            eemsg = rospy.wait_for_message('/ur_eepose_state', ur_ee_cmd)
            if (jmsg.header.stamp != None) and (eemsg.pose != None):
                
                print("Received the first msg from \'/allegro_joints_state\'.")
                print("The message is\n", jmsg,"\n")
                print("Received the first msg from \'/ur_eepose_state\'.")
                print("The message is\n", eemsg,"\n")
                # self.joint_pos_cmd = jmsg.position
                # self.ee_pos_cmd[0] = eemsg.pose.position.x
                # self.ee_pos_cmd[1] = eemsg.pose.position.y
                # self.ee_pos_cmd[2] = eemsg.pose.position.z
                # self.ee_quat_cmd[0] = eemsg.pose.orientation.x
                # self.ee_quat_cmd[1] = eemsg.pose.orientation.y
                # self.ee_quat_cmd[2] = eemsg.pose.orientation.z
                # self.ee_quat_cmd[3] = eemsg.pose.orientation.w
                self.jcmd_pub = copy.deepcopy(jmsg)
                self.urcmd_pub = copy.deepcopy(eemsg)
                break
            rospy.sleep(0.01)
            # '/allegro_joints_state', allegro_hand_joints_cmd
        if self.sim_flag:
            self.last_time = float(jmsg.header.stamp.secs) + float(jmsg.header.stamp.nsecs)*1e-9
            self.current_time = float(jmsg.header.stamp.secs) + float(jmsg.header.stamp.nsecs)*1e-9
        else:
            self.last_time = time()
        self.allegro_jstates = np.array(copy.deepcopy(jmsg.position))
        self.ur_eestates = ros_pose2new_pose_quat(eemsg.pose)
        self.delta_time = 0.0

        # Save the first states, for safety use:
        self.init_allegro_jstates = copy.deepcopy(self.allegro_jstates)
        self.init_ur_eestates = copy.deepcopy(self.ur_eestates)
        print("self.ur_eestates:", self.ur_eestates)
        print("self.init_ur_eestates:", self.init_ur_eestates)
        print("self.init_allegro_jstates:", self.init_allegro_jstates)

        self.ry_hat_in_tip = np.zeros(4)
        self.rz_hat_in_tip = np.zeros(4)
        
        
        # urdf_parser and KDL
        # import URDF
        self.hand_description = URDF.from_xml_file(self.this_file_dir+"/allegro_tactile_description/allegro_hand_description_right_tactile_end.urdf")
        # create kdl_tree
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)
        # init kdl_Frames for each finger
        self.index_pos = kdl.Frame() # Construct an identity frame
        self.index_pose = Pose()
        self.mid_pos = kdl.Frame() # Construct an identity frame
        self.ring_pos = kdl.Frame() # Construct an identity frame
        self.thumb_pos = kdl.Frame() # Construct an identity frame
        # init kdl_JntArrays for each finger
        self.index_qpos = kdl.JntArray(4)
        self.mid_qpos = kdl.JntArray(4)
        self.ring_qpos = kdl.JntArray(4)
        self.thumb_qpos = kdl.JntArray(4)
        kdl.SetToZero(self.index_qpos)
        kdl.SetToZero(self.mid_qpos)
        kdl.SetToZero(self.ring_qpos)
        kdl.SetToZero(self.thumb_qpos)
        # chain
        self.index_chain = self.hand_tree.getChain("palm_link", "index_tip_tactile")
        self.mid_chain = self.hand_tree.getChain("palm_link", "mid_tip_tactile")
        self.ring_chain = self.hand_tree.getChain("palm_link", "ring_tip_tactile")
        self.thumb_chain = self.hand_tree.getChain("palm_link", "thumb_tip_tactile")
        # forward kinematics
        self.index_fk = kdl.ChainFkSolverPos_recursive(self.index_chain)
        self.mid_fk = kdl.ChainFkSolverPos_recursive(self.mid_chain)
        self.ring_fk = kdl.ChainFkSolverPos_recursive(self.ring_chain)
        self.thumb_fk = kdl.ChainFkSolverPos_recursive(self.thumb_chain)
        # KDLKinematics
        self.index_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "index_tip_tactile")
        self.mid_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "mid_tip_tactile")
        self.ring_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "ring_tip_tactile")
        self.thumb_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "thumb_tip_tactile")

        # finger angles IK weights
        self.finger_joint_w = np.ones(16)
        # self.thumb2mid_kdl_kin = KDLKinematics(self.hand_description, "thumb_tip_tactile", "mid_tip_tactile")

        # pose estimation

        # moveit
        """
            moveit_commander.roscpp_initialize(sys.argv)
            self.hand = moveit_commander.RobotCommander()
            self.hand_scene = moveit_commander.PlanningSceneInterface()
            group_name = "right_index"
            self.hand_group = moveit_commander.MoveGroupCommander(group_name)
            self.planning_frame = self.hand_group.get_planning_frame()
            print("============ Planning frame: %s" % self.planning_frame)
            self.eef_link = self.hand_group.get_end_effector_link()
            print("============ End effector link: %s" % self.eef_link)
            group_names = self.hand.get_group_names()
            print ("============ Available Planning Groups:", self.hand.get_group_names())
            # Sometimes for debugging it is useful to print the entire state of the
            # robot:
            print ("============ Printing robot state")
            print (self.hand.get_current_state())
            print ("")
        """

        rospy.Subscriber('/allegro_joints_state', allegro_hand_joints_cmd, self.allegro_jstates_callback)
        rospy.Subscriber('/ur_eepose_state', ur_ee_cmd, self.ur_eestates_callback)
        rospy.Subscriber('allegro_tactile', tactile_msgs, self.allegro_tacstates_callback)

        

        # Threading
        # self.mutex = threading.Lock()
        self.ros_sub_thread = threading.Thread(target=self.loop)
        self.ros_sub_thread.setDaemon(True)
        self.key_break = False
        self.wait_key_thread = threading.Thread(target=self.wait_key_break)
        self.wait_key_thread.setDaemon(True)
        self.ros_sub_thread.start()
        self.wait_key_thread.start()
        # self.main_process_thread.join()

    def wait_key_break(self):
        a = input()
        self.key_break = True

    def get_time(self):
        if self.sim_flag:
            self.current_time = float(self.jstate_secs + self.eestate_secs)*0.5 + float(self.jstate_nsecs + self.eestate_nsecs)*0.5*1e-9
        else:
            self.current_time = time()
        self.delta_time = self.current_time - self.last_time
        # print "tt=", self.current_time, "t0=", self.last_time, "dt=", self.delta_time
    def save_time(self):
        if self.sim_flag:
            self.last_time = self.current_time
        else:
            self.last_time = self.current_time

    def allegro_jstates_callback(self, msg):
        if msg.position != None:      
            self.allegro_jstates = np.array(msg.position)
            self.index_qpos[0] = self.allegro_jstates[0]
            self.index_qpos[1] = self.allegro_jstates[1]
            self.index_qpos[2] = self.allegro_jstates[2]
            self.index_qpos[3] = self.allegro_jstates[3]
            # print("index_qpos:", self.index_qpos)
            self.mid_qpos[0] = self.allegro_jstates[4]
            self.mid_qpos[1] = self.allegro_jstates[5]
            self.mid_qpos[2] = self.allegro_jstates[6]
            self.mid_qpos[3] = self.allegro_jstates[7]
            self.ring_qpos[0] = self.allegro_jstates[8]
            self.ring_qpos[1] = self.allegro_jstates[9]
            self.ring_qpos[2] = self.allegro_jstates[10]
            self.ring_qpos[3] = self.allegro_jstates[11]
            self.thumb_qpos[0] = self.allegro_jstates[12]
            self.thumb_qpos[1] = self.allegro_jstates[13]
            self.thumb_qpos[2] = self.allegro_jstates[14]
            self.thumb_qpos[3] = self.allegro_jstates[15]
            # index
            for i_joint in range(4):
                if self.index_qpos[i_joint] < self.index_limits[1,i_joint] + 0.0175: # about 1 degree
                    self.index_reach_limit = True
                elif self.index_qpos[i_joint] > self.index_limits[0,i_joint] - 0.0175:
                    self.index_reach_limit = True
            # thumb
            for i_joint in range(4):
                if self.thumb_qpos[i_joint] < self.thumb_limits[1,i_joint] + 0.0175: # about 1 degree
                    self.thumb_reach_limit = True
                elif self.thumb_qpos[i_joint] > self.thumb_limits[0,i_joint] - 0.0175:
                    self.thumb_reach_limit = True
            for iq in range(4):
                # index
                self.index_weights[iq,0] = 13.44 - 12.0*(self.index_qpos[iq]-self.index_limits[1,iq])*(self.index_limits[0,iq]-self.index_qpos[iq])/self.index_ranges_sq[0,iq]
                # mid
                self.mid_weights[iq,0] = 13.44 - 12.0*(self.mid_qpos[iq]-self.mid_limits[1,iq])*(self.mid_limits[0,iq]-self.mid_qpos[iq])/self.mid_ranges_sq[0,iq]
                # ring
                self.ring_weights[iq,0] = 13.44 - 12.0*(self.ring_qpos[iq]-self.ring_limits[1,iq])*(self.ring_limits[0,iq]-self.ring_qpos[iq])/self.ring_ranges_sq[0,iq]
                # thumb
                self.thumb_weights[iq,0] = 13.44 - 12.0*(self.thumb_qpos[iq]-self.thumb_limits[1,iq])*(self.thumb_limits[0,iq]-self.thumb_qpos[iq])/self.thumb_ranges_sq[0,iq]
            # print("thumb_qpos:", self.thumb_qpos)
        if msg.header != None:
            if msg.header.seq != None:
                self.jstate_seq = msg.header.seq
            if msg.header.stamp != None:
                if msg.header.stamp.secs != None:
                    self.jstate_secs = msg.header.stamp.secs
                else:
                    self.jstate_secs = 0
                if msg.header.stamp.nsecs != None:
                    self.jstate_nsecs = msg.header.stamp.nsecs
                else:
                    self.jstate_nsecs = 0

    def allegro_tacstates_callback(self, msg):
        self.index_tip = np.array(msg.index_tip_Value)
        self.mid_tip = np.array(msg.middle_tip_Value)
        self.ring_tip = np.array(msg.ring_tip_Value)
        self.thumb_tip = np.array(msg.thumb_tip_Value)
        self.index_tip = self.index_tip.astype(float)
        self.mid_tip = self.mid_tip.astype(float)
        self.ring_tip = self.ring_tip.astype(float)
        self.thumb_tip = self.thumb_tip.astype(float)
        # Call main_process in the lowest-frequency callback function

    def ur_eestates_callback(self, msg):
        self.ur_eestates[0] = msg.pose.position.x
        self.ur_eestates[1] = msg.pose.position.y
        self.ur_eestates[2] = msg.pose.position.z
        self.ur_eestates[3] = msg.pose.orientation.x
        self.ur_eestates[4] = msg.pose.orientation.y
        self.ur_eestates[5] = msg.pose.orientation.z
        self.ur_eestates[6] = msg.pose.orientation.w
        if msg.header != None:
            if msg.header.seq != None:
                self.eestate_seq = msg.header.seq
            if msg.header.stamp != None:
                if msg.header.stamp.secs != None:
                    self.eestate_secs = msg.header.stamp.secs
                else:
                    self.eestate_secs = 0
                if msg.header.stamp.nsecs != None:
                    self.eestate_nsecs = msg.header.stamp.nsecs
                else:
                    self.eestate_nsecs = 0

    def loop(self):
        rospy.logwarn("Starting MySbuscriber Loop...")
        rospy.spin()

    # Some functions that simplify the main_process() below
    def write_publish_ee_cmd(self, pos_t, quat_t):
        error_p = self.ur_eestates[:3] - pos_t
        norm_error_p = np.linalg.norm(error_p)
        if norm_error_p > 0.015:
            pos_t = pos_t + error_p/norm_error_p*0.015
        error_angle = np.zeros(3)
        error_angle[0],error_angle[1],error_angle[2] = angular_vel_from_quats(quat_t,self.ur_eestates[3:],1.0)
        error_angle_norm = np.linalg.norm(error_angle)
        if error_angle_norm > 0.2:
            delta_pose = np.zeros(6)
            delta_pose[3:] = error_angle/error_angle_norm*0.2
            quat_t = delta_pose2next_pose(delta_pose,self.ur_eestates)[3:]
        self.urcmd_pub.pose = pose_quat2new_ros_pose(np.hstack((pos_t, quat_t)))
        self.publish_ee_cmd()
        return pos_t, quat_t

    def write_publish_allegro_cmd(self, jcmd):
        angle_error_max = 0.3
        jstates = copy.deepcopy(self.allegro_jstates)
        for i in range(np.size(jstates)):
            delta_i = jcmd[i] - jstates[i]
            if delta_i > angle_error_max:
                jcmd[i] = jstates[i] + angle_error_max
            elif delta_i < -angle_error_max:
                jcmd[i] = jstates[i] - angle_error_max
            self.finger_joint_w[i] = 1.0 + 9.0/angle_error_max/angle_error_max*(delta_i**2)
        self.jcmd_pub.position = jcmd
        self.publish_allegro_cmd()
        return jcmd

    def publish_ee_cmd(self):
        """
        Publish self.urcmd_pub
        In this function, you only need to specify urcmd_pub.pose,\n
        urcmd_pub.header is auto filled, and then urcmd_pub is published.\n
        If cmd is too large, it will be reset to current ee_state.
        """
        # next position cannot be too far from the initial position when the procedure began
        if np.linalg.norm(self.init_ur_eestates[:3]-ros_pose2new_pose_quat(self.urcmd_pub.pose)[:3]) < 0.9:
            # next position cannot be too far from last position
            if np.linalg.norm(self.ur_eestates[:3]-ros_pose2new_pose_quat(self.urcmd_pub.pose)[:3]) < 0.15:
                self.urcmd_pub.header.seq = self.urcmd_pub.header.seq + 1
                self.urcmd_pub.header.stamp = rospy.Time.now()
                self.ur_publisher.publish(self.urcmd_pub)
            else:
                self.urcmd_pub.pose = pose_quat2new_ros_pose(self.ur_eestates)
                print("Control signal too far from current position, not pulishing")
        else:
            self.urcmd_pub.pose = pose_quat2new_ros_pose(self.ur_eestates)
            print("Control signal too far from initial position, not pulishing")

    def publish_allegro_cmd(self):
        """
        In this function, you only need to specify jcmd_pub.position,\n
        jcmd_pub.header is auto filled, and then jcmd_pub is published
        """
        self.jcmd_pub.header.seq = self.jcmd_pub.header.seq + 1
        self.jcmd_pub.header.stamp = rospy.Time.now()
        jcmd = np.array(self.jcmd_pub.position)
        for i_joint in range(16):
            if jcmd[i_joint] < self.j_limits[1,i_joint]:
                jcmd[i_joint] = self.j_limits[1,i_joint]
            elif jcmd[i_joint] > self.j_limits[0,i_joint]:
                jcmd[i_joint] = self.j_limits[0,i_joint]
        # # -0.47 0.47
        # if jcmd[0] < -0.47:
        #     jcmd[0] = -0.47
        #     print("index0")
        # elif jcmd[0] > 0.47:
        #     jcmd[0] = 0.47
        # # -0.196 1.61
        # if jcmd[1] < -0.196:
        #     jcmd[1] = -0.196
        # elif jcmd[1] > 1.61:
        #     jcmd[1] = 1.61
        # # "-0.174 1.709"
        # if jcmd[2] < -0.174:
        #     jcmd[2] = -0.174
        # elif jcmd[2] > 1.709:
        #     jcmd[2] = 1.709
        # # "-0.227 1.618"
        # if jcmd[3] < -0.227:
        #     jcmd[3] = -0.227
        # elif jcmd[3] > 1.618:
        #     jcmd[3] = 1.618
        # # -0.47 0.47
        # if jcmd[4] < -0.47:
        #     jcmd[4] = -0.47
        # elif jcmd[4] > 0.47:
        #     jcmd[4] = 0.47
        # # -0.196 1.61
        # if jcmd[5] < -0.196:
        #     jcmd[5] = -0.196
        # elif jcmd[5] > 1.61:
        #     jcmd[5] = 1.61
        # # "-0.174 1.709"
        # if jcmd[6] < -0.174:
        #     jcmd[6] = -0.174
        # elif jcmd[6] > 1.709:
        #     jcmd[6] = 1.709
        # # "-0.227 1.618"
        # if jcmd[7] < -0.227:
        #     jcmd[7] = -0.227
        # elif jcmd[7] > 1.618:
        #     jcmd[7] = 1.618
        # # -0.47 0.47
        # if jcmd[8] < -0.47:
        #     jcmd[8] = -0.47
        # elif jcmd[8] > 0.47:
        #     jcmd[8] = 0.47
        # # -0.196 1.61
        # if jcmd[9] < -0.196:
        #     jcmd[9] = -0.196
        # elif jcmd[9] > 1.61:
        #     jcmd[9] = 1.61
        # # "-0.174 1.709"
        # if jcmd[10] < -0.174:
        #     jcmd[10] = -0.174
        # elif jcmd[10] > 1.709:
        #     jcmd[10] = 1.709
        # # "-0.227 1.618"
        # if jcmd[11] < -0.227:
        #     jcmd[11] = -0.227
        # elif jcmd[11] > 1.618:
        #     jcmd[11] = 1.618
        # # "0.263 1.396"
        # if jcmd[12] < 0.263:
        #     jcmd[12] = 0.263
        # elif jcmd[12] > 1.396:
        #     jcmd[12] = 1.396
        # # "-0.105 1.163"
        # if jcmd[13] < -0.105:
        #     jcmd[13] = -0.105
        # elif jcmd[13] > 1.163:
        #     jcmd[13] = 1.163
        # # "-0.189 1.644"
        # if jcmd[14] < -0.189:
        #     jcmd[14] = -0.189
        # elif jcmd[14] > 1.644:
        #     jcmd[14] = 1.644
        # # "-0.162 1.719"
        # if jcmd[15] < -0.162:
        #     jcmd[15] = -0.162
        # elif jcmd[15] > 1.719:
        #     jcmd[15] = 1.719
        # print("jcmd:", jcmd)
        self.jcmd_pub.position = jcmd
        self.allegro_publisher.publish(self.jcmd_pub)

    def plan_grasp(self,obj_pos=np.array([0, 0, 0.785]),obj_euler=np.array([0, 0, 1.57]),hand_pos=np.array([0, -0.4, 0.85]),obj_in_hand_pos=np.array([0.09, -0.16, -0.0]), obj_in_hand_euler=np.array([-1.57, 0, 0])):
        """
        Plan a grasp here (just planning, not moving):\n
        object pose:\n
        obj_pos = np.array([0, 0, 0.785])\n
        obj_euler = np.array([0, 0, 1.57])
        """
        obj_dcm = Rotation.from_euler("xyz",obj_euler).as_dcm()
        # initial hand pose, orientation is not important:
        # hand_pos = np.array([0, -0.4, 0.85])
        hand2obj = obj_pos - hand_pos # Make this vector close to obj's x-axis
        obj_y_specified = np.cross(obj_dcm[:3,2].flatten(),hand2obj)
        obj_y_specified = obj_y_specified/np.linalg.norm(obj_y_specified)
        obj_x_specified = np.cross(obj_y_specified, obj_dcm[:3,2].flatten())
        obj_x_specified = obj_x_specified/np.linalg.norm(obj_x_specified)
        obj_dcm[:3,0] = obj_x_specified
        obj_dcm[:3,1] = obj_y_specified
        obj_T = np.eye(4)
        obj_T[:3,:3] = obj_dcm
        obj_T[:3,3] = obj_pos
        obj_in_hand_T = np.eye(4)
        obj_in_hand_T[:3,3] = obj_in_hand_pos
        obj_in_hand_T[:3,:3] = Rotation.from_euler("xyz",obj_in_hand_euler).as_dcm()
        hand_pregrasp_T = np.dot(obj_T, np.linalg.inv(obj_in_hand_T))
        # pregrasping hand pose:
        hand_pregrasp_pos = hand_pregrasp_T[:3,3].flatten()
        hand_pregrasp_quat = Rotation.from_dcm(hand_pregrasp_T[:3,:3]).as_quat()
        return hand_pregrasp_pos, hand_pregrasp_quat

    def move2posquat(self, pos, quat, vel=0.01, angle_vel_degrees=3.0, error_p=0.01, error_angle_degrees=3.0):
        """
        error_p is the tolerance of position errors (default = 0.01),
        error_angle_degrees is the tolerance of orientation errors (in degrees) (default = 3.0),
        """
        self.get_time()
        p_distance = pos - self.ur_eestates[:3]
        duration_p = np.linalg.norm(p_distance)/vel
        v = p_distance/duration_p
        duration_o = quats2delta_angle_degrees(quat, self.ur_eestates[3:])/angle_vel_degrees
        w = angular_vel_from_quats(quat,self.ur_eestates[3:],dt=duration_o)
        while not rospy.is_shutdown():
            self.get_time()
            pos_t = ros_pose2new_pose_quat(self.urcmd_pub.pose)[:3]
            quat_t = ros_pose2new_pose_quat(self.urcmd_pub.pose)[3:]
            I3d = np.eye(3)
            delta_pos = np.linalg.norm(pos-self.ur_eestates[:3])
            delta_angle_d = quats2delta_angle_degrees(quat, self.ur_eestates[3:])
            if delta_pos > error_p:
                pos_t = self.delta_time*v + ros_pose2new_pose_quat(self.urcmd_pub.pose)[:3] # Use real current pos or current cmd pos?
            if delta_angle_d > error_angle_degrees:
                R_0 = Rotation.from_quat(ros_pose2new_pose_quat(self.urcmd_pub.pose)[3:]).as_dcm()
                R_t = np.dot((self.delta_time*cross_product_matrix_from_vector3d(w)+I3d), R_0)
                quat_t = Rotation.from_dcm(R_t).as_quat()
            if (delta_pos < error_p) and (delta_angle_d < error_angle_degrees):
                print("Palm stop approaching: distance between target pose and current pose is below tolerance.")
                break
            self.write_publish_ee_cmd(pos_t, quat_t)
            rospy.sleep(0.001)
            self.save_time()

    def main_process(self):
        """ FK: palm and finger tips' relationship """
        # Finger tips in palm
        T_palm2index = np.mat(np.eye(4))
        T_palm2mid = np.mat(np.eye(4))
        T_palm2ring = np.mat(np.eye(4))
        T_palm2thumb = np.mat(np.eye(4))
        # Palm in finger tips
        T_index2palm = np.mat(np.eye(4))
        T_mid2palm = np.mat(np.eye(4))
        T_ring2palm = np.mat(np.eye(4))
        T_thumb2palm = np.mat(np.eye(4))
        # Temporarily save the rotation matrixes and positions of the tips w.r.t. palm
        M = np.mat(np.zeros((3,3)))
        p = np.mat(np.zeros((3,1)))
        # self.save_time()
        # g_dir = "/home/lhz/Documents/tactile_servo_config/1_3f/"
        # g_dir = "/home/lhz/Documents/tactile_servo_config/0218/" # kexing_largeshaker
        # g_dir = self.this_file_dir + "/tactile_servo_config/largeshaker/0218/" # kexing_largeshaker
        # g_dir = "/home/lhz/Documents/tactile_servo_config/cube_2f/" # cube
        # g_dir = self.this_file_dir + "/tactile_servo_config/spherecylinder/spherecylinder_2f/" # spherecylinder
        g_dir = self.this_file_dir + "/tactile_servo_config/sphere/sphere_2f_1/" # sphere
        print("g_dir:", g_dir)
        file_list = listdir(g_dir)
        quats_palm2obj = np.mat(np.zeros((0,4)))
        positions_palm2obj = np.zeros(3)
        index_values = np.zeros(72)
        mid_values = np.zeros(72)
        ring_values = np.zeros(72)
        thumb_values = np.zeros(72)
        target_pose = np.zeros(16)
        load_amount = 0
        for fname in file_list:
            # fpart = copy.deepcopy(fname)
            # fpart.replace(".txt", "")
            if "T_palm2obj" in fname:
                load_amount = load_amount + 1
                q_tmp = Rotation.from_dcm(np.loadtxt(g_dir+fname)[:3,:3]).as_quat()
                quats_palm2obj = np.vstack((quats_palm2obj, q_tmp))
                positions_palm2obj = positions_palm2obj + np.reshape(np.array(np.loadtxt(g_dir+fname)[:3,3]), 3)
            if "index_tactile" in fname:
                index_values = index_values + np.reshape(np.array(np.loadtxt(g_dir+fname)), 72)
            if "mid_tactile" in fname:
                mid_values = mid_values + np.reshape(np.array(np.loadtxt(g_dir+fname)), 72)
            if "ring_tactile" in fname:
                ring_values = ring_values + np.reshape(np.array(np.loadtxt(g_dir+fname)), 72)
            if "thumb_tactile" in fname:
                thumb_values = thumb_values + np.reshape(np.array(np.loadtxt(g_dir+fname)), 72)
            if "joint_state" in fname:
                target_pose = target_pose + np.reshape(np.array(np.loadtxt(g_dir+fname)), 16)
        weights_quats = np.ones(load_amount)
        target_quat = average_quaternions(quats_palm2obj, weights_quats)
        target_pos = positions_palm2obj/float(load_amount)
        print("Desired obj pose:", target_pos, target_quat)
        target_pose = target_pose/float(load_amount)
        target_pose = np.mat(target_pose).T
        print("Desired Allegro Jpose:", target_pose.T)
        # index tactile feature
        target_indexV = index_values/float(load_amount)
        print("target_indexV:", target_indexV)
        index_tf_d = np.mat(np.zeros((3,1)))
        index_tf_d[0,0] = 125
        index_tf_d[1,0] = np.matmul(self.index_tacxel_pose_euler[:,1].T, np.mat(target_indexV).T)[0,0]/np.sum(target_indexV)
        index_tf_d[2,0] = np.matmul(self.index_tacxel_pose_euler[:,2].T, np.mat(target_indexV).T)[0,0]/np.sum(target_indexV)
        print("index_tf_d:", index_tf_d.T)
        # mid tactile feature
        target_midV = mid_values/float(load_amount)
        print("target_midV:", target_midV)
        mid_tf_d = np.mat(np.zeros((3,1)))
        mid_tf_d[0,0] = 125
        mid_tf_d[1,0] = np.matmul(self.index_tacxel_pose_euler[:,1].T, np.mat(target_midV).T)[0,0]/np.sum(target_midV)
        mid_tf_d[2,0] = np.matmul(self.index_tacxel_pose_euler[:,2].T, np.mat(target_midV).T)[0,0]/np.sum(target_midV)
        # ring tactile feature
        target_ringV = ring_values/float(load_amount)
        print("target_ringV:", target_ringV)
        ring_tf_d = np.mat(np.zeros((3,1)))
        ring_tf_d[0,0] = 125
        ring_tf_d[1,0] = np.matmul(self.index_tacxel_pose_euler[:,1].T, np.mat(target_ringV).T)[0,0]/np.sum(target_ringV)
        ring_tf_d[2,0] = np.matmul(self.index_tacxel_pose_euler[:,2].T, np.mat(target_ringV).T)[0,0]/np.sum(target_ringV)
        # thumb tactile feature
        target_thumbV = thumb_values/float(load_amount)
        print("target_thumbV:", target_thumbV)
        thumb_tf_d = np.mat(np.zeros((3,1)))
        thumb_tf_d[0,0] = 125
        thumb_tf_d[1,0] = np.matmul(self.index_tacxel_pose_euler[:,1].T, np.mat(target_thumbV).T)[0,0]/np.sum(target_thumbV)
        thumb_tf_d[2,0] = np.matmul(self.index_tacxel_pose_euler[:,2].T, np.mat(target_thumbV).T)[0,0]/np.sum(target_thumbV)
        """ Calculate the index in thumb pose with FK """
        qpos_tmp = kdl.JntArray(4)
        # index
        qpos_tmp[0] = target_pose[0,0]
        qpos_tmp[1] = target_pose[1,0]
        qpos_tmp[2] = target_pose[2,0]
        qpos_tmp[3] = target_pose[3,0]
        kdl_calc_fk(self.index_fk, qpos_tmp, self.index_pos)
        Td_palm2index = KDLframe2T(self.index_pos)
        # thumb
        qpos_tmp[0] = target_pose[12,0]
        qpos_tmp[1] = target_pose[13,0]
        qpos_tmp[2] = target_pose[14,0]
        qpos_tmp[3] = target_pose[15,0]
        kdl_calc_fk(self.thumb_fk, qpos_tmp, self.thumb_pos)
        Td_palm2thumb = KDLframe2T(self.thumb_pos)
        Td_thumb2index = np.matmul(np.linalg.inv(Td_palm2thumb), Td_palm2index)
        print("Desired index in thumb: Td_thumb2index;\n", Td_thumb2index)

        # target_pose = np.loadtxt(g_dir)
        print("Hand joints target pose:\n", target_pose)

        index_tf = np.mat(np.zeros((3,1)))
        index_tf_last = np.mat(np.zeros((3,1)))
        # index_tf_d = np.mat([800, 0, 0.0055]).T

        mid_tf = np.mat(np.zeros((3,1)))
        mid_tf_last = np.mat(np.zeros((3,1)))
        # mid_tf_d = np.mat([100, 0, 0]).T

        ring_tf = np.mat(np.zeros((3,1)))
        ring_tf_last = np.mat(np.zeros((3,1)))
        # ring_tf_d = np.mat([100, 0, 0]).T

        thumb_tf = np.mat(np.zeros((3,1)))
        # thumb_tf_d = np.mat([800, 0, -0.012]).T
        thumb_tf_last = np.mat(np.zeros((3,1)))
        
        print("Desired tactile features:")
        print("Thumb tip:\n", thumb_tf_d)
        print("Mid tip:\n", mid_tf_d)

        self.get_time()
        self.save_time()
        # Plan a simple grasp:
        obj_pos = np.array([0, 0, 0.785])
        # obj_in_hand_pos=np.array([0.15, -0.11, -0.0])
        hand_pregrasp_pos, hand_pregrasp_quat = self.plan_grasp(obj_pos,obj_euler=np.array([0, 0, 1.57]),hand_pos=np.array([0, -0.4, 0.6]),obj_in_hand_pos=np.array([0.105, -0.12, 0.01]), obj_in_hand_euler=np.array([-1.57, 0, -0.2]))
        print("Obj at", obj_pos)
        # print("Target position:", hand_pregrasp_pos, "; target orientation (quat):", hand_pregrasp_quat)
        I3d = np.eye(3)
        print("Approaching object ...")
        print("Initial ur ee state:", self.ur_eestates)
        """ Open hand """
        t_open = 0.0
        while not rospy.is_shutdown():
            self.get_time()
            # with self.mutex:

            jcmd = np.zeros(16)
            jcmd[0] = target_pose[0,0]
            jcmd[4] = target_pose[4,0]
            jcmd[8] = target_pose[8,0]
            jcmd[12] = target_pose[12,0]
            
            self.jcmd_pub.position = jcmd
            self.publish_allegro_cmd()
            if (t_open > 3.0):
                break
            t_open = t_open + self.delta_time
            rospy.sleep(0.004)
            self.save_time()
        """ Move to obj """
        # self.move2posquat(hand_pregrasp_pos, hand_pregrasp_quat, vel=0.25, angle_vel_degrees=20.0, error_p=0.008)
        print("Closing hand...")
        print("Moving middle and thumb fingers...")
        """ Close hand """
        jcmd = np.zeros(16)
        jcmd[0] = target_pose[0,0]
        jcmd[4] = target_pose[4,0]
        jcmd[8] = target_pose[8,0]
        jcmd[12] = target_pose[12,0]
        while not rospy.is_shutdown():
            self.get_time()
            # with self.mutex:
            # if self.delta_time > 0.02:
            ang_vel = 0.1
            if sum(self.index_tip) < 150.0: # index finger
                jcmd[1] = jcmd[1] + ang_vel*self.delta_time
                jcmd[2] = jcmd[2] + ang_vel*self.delta_time
            if sum(self.thumb_tip) < 150.0: # Thumb finger
                jcmd[14] = jcmd[14] + ang_vel*self.delta_time
                jcmd[15] = jcmd[15] + ang_vel*self.delta_time
            self.jcmd_pub.position = jcmd
            self.publish_allegro_cmd()
            if (sum(self.index_tip)>150.0) and (sum(self.thumb_tip) > 150.0):
                break
            rospy.sleep(0.004)
            # print("delta_time:", self.delta_time, "current_time:", self.current_time, "last_time:", self.last_time)
            self.save_time()
        print("Middle and thumb finger tips stop.")
        print("---------------------------------------Finish tipping---------------------------------------")

        print("Tactile servoing begins ...")
        # target_pose = np.mat([0.0235, 1.003339493829417783e+00, 6.470923439160897184e-01, 9.107568419408678118e-01,
        #                         -8.330980837360064950e-02, 1.003339493829417783e+00, 6.470923439160897184e-01, 9.107568419408678118e-01,
        #                         -0.24, 0.544, 0.749, 0.881,
        #                         1.288869808247061322e+00, 1.282233491253784441e-01, 2.071146641358613261e-01, 7.838402103670218946e-01]).T
        # target_pose = np.mat([0.0235, 0.544, 0.749, 0.989,
        #                         -8.330980837360064950e-02, 1.003339493829417783e+00, 6.470923439160897184e-01, 9.107568419408678118e-01,
        #                         -0.24, 0.544, 0.749, 0.881,
        #                         1.288869808247061322e+00, 1.282233491253784441e-01, 2.071146641358613261e-01, 7.838402103670218946e-01]).T

        
        # tactile_jacobian
        # feature = [force, y, z]
        # J_s_+ = np.mat([[-3e-6,  0,  0],
        #                 [ 0, -0.4,  0],
        #                 [ 0,  0, -0.75],
        #                 [ 0,  0,  0],w_diag
        #                 [ 0,  0,  25],
        #                 [ 0, -60,  0]])

        # J_s * V_tip = V_tf
        # twist of tip ----> rate of tactile feature
        # Version 1:
        """
        These are weights for control signals
        """
        # w_palm_diag = np.ones(6)
        # w_palm_diag[0:3] = w_palm_diag[0:3]*1.0
        # w_palm_diag[3:6] = w_palm_diag[3:6]*1.0
        # w_fjoint_diag = np.ones(6)
        # w_fjoint_diag = w_fjoint_diag*1.0
        # w_fjoint_diag[0:3] = self.finger_joint_w[1:4]
        # w_fjoint_diag[3:6] = self.finger_joint_w[13:16]
        # w_diag = np.concatenate((w_palm_diag, w_fjoint_diag))
        # # w_diag = np.reshape(w_diag, 12)
        # w_diag_inv_sqr = np.sqrt(1/w_diag)
        # W_inv_sqr = np.diag(w_diag_inv_sqr)
        # W_sqr = np.linalg.inv(W_inv_sqr)
        w_palm = np.ones(6)
        w_palm = w_palm*1.0
        w_palm[0:3] = w_palm[0:3]*0.8
        w_palm[3:5] = w_palm[3:5]*0.8
        w_palm[5] = w_palm[5]*8.0
        # w_palm_r = np.loadtxt(self.this_file_dir + "/tactile_servo_config/")

        w_index = np.ones(3)
        w_thumb = np.ones(3)
        w_diag = np.concatenate((w_palm, w_index))
        w_diag = np.concatenate((w_diag, w_thumb))
        """ Jacobian: tip twist ----> tip tactile feature rate """
        # J_si = np.mat([[4.0e4,  0,  0, 0,   0,    0],
        #               [   0, -0,  0, 0,   0, -1.6e-2],
        #               [   0,  0, -1, 0,   0,    0]])
        # define some functions for J_s switch:
        def make_J_s_abcd(sy, sz):
            rz = 0 if sy==1 else 1
            ry = 0 if sz==1 else 1
            # J_s_abcd = np.mat([[9e4,  0,  0, 0,   0,    0],
            #                 [   0,  -1.5*sy, -0.0, 0,   0, -3.2e-2*rz],
            #                 [   0,  0, -1.25*sz, 0,   3.2e-2*ry,    0]])*10.
            # J_s_abcd = np.mat([[9e4,  0,  0, 0,   0,    0],
            #                 [   0,  -3.2e-1*sy, -0.0, 0,   0, -3.2e-2*rz],
            #                 [   0,  0, -3.2e-1*sz, 0,   3.2e-2*ry,    0]])*10.
            J_s_abcd = np.mat([[9e5,  0,  0, 0,   0,    0],
                            [   0,  -1.6e-1*sy, -0.0, 0,   0, -3.2e-2*rz],
                            [   0,  0, -3.2e-1*sz, 0,   3.2e-2*ry,    0]])*10.
            return J_s_abcd
        
        def servo_to_T(curTnext, delta_tf, delta_t_tf):
            poseul_next = T2pos_euler(curTnext)
            # print('poseul_next', poseul_next)

            js_abcd = np.array((0, 1, 2, 3), np.matrix)
            js_abcd[0] = make_J_s_abcd(0, 0)
            js_abcd[1] = make_J_s_abcd(1, 0)
            js_abcd[2] = make_J_s_abcd(0, 1)
            js_abcd[3] = make_J_s_abcd(1, 1)

            curTnext_js = np.array((0, 1, 2, 3), np.matrix)
            for i in range(4):
                js = np.matrix(js_abcd[i])
                # print("delta_tf:", delta_tf.T,"delta_t_tf:", delta_t_tf)
                twist_vel = np.matmul(pinv_SVD(js), delta_tf*delta_t_tf)
                # print("twist_vel:", twist_vel.T)
                posxyzw_next2curr = delta_pose2next_pose(twist_vel, [0, 0, 0, 0, 0, 0, 1])
                # print("posxyzw_next2curr:", posxyzw_next2curr)
                curTnext_js[i] = pose2T(posxyzw_next2curr)

            for i in range(4):
                poseul_js = T2pos_euler(curTnext_js[i])
                # print('poseul_js', i, poseul_js)

            for i in range(4):
                poseul_js = T2pos_euler(curTnext_js[i])
                s = np.linalg.norm(poseul_next[1:3] - poseul_js[1:3])
                r = np.linalg.norm(poseul_next[4:6] - poseul_js[4:6])
                sr = np.linalg.norm([s,r])
                # print(i, 's', s.round(3), 'r', r.round(3), 'sr', sr.round(6),
                #     poseul_next[1]*poseul_js[1]>=0,
                #     poseul_next[2]*poseul_js[2]>=0,
                #     poseul_next[4]*poseul_js[4]>=0,
                #     poseul_next[5]*poseul_js[5]>=0)

            sr = np.zeros(4) + 1000
            for i in range(4):
                poseul_js = T2pos_euler(curTnext_js[i])
                if (poseul_next[1]*poseul_js[1]>=0  and
                    poseul_next[2]*poseul_js[2]>=0  and
                    poseul_next[4]*poseul_js[4]>=0  and
                    poseul_next[5]*poseul_js[5]>=0 ):
                    s = np.linalg.norm(poseul_next[1:3] - poseul_js[1:3])
                    r = np.linalg.norm(poseul_next[4:6] - poseul_js[4:6])
                    sr[i] = np.linalg.norm([s,r])
            i_min = np.argmin(sr)
            sr_min = sr[i_min]

            if sr_min == 1000:
                # print('no good Js !!!!!!!!!!!!')
                e = np.mat(np.eye(4))
                e[0, 3] = 0.27*delta_t_tf
                e[1, 3] = poseul_next[1]*10.0*delta_t_tf
                e[2, 3] = poseul_next[2]*10.0*delta_t_tf
                return e
            else:
                # print('best js is', i_min, sr_min)
                return curTnext_js[i_min]

        def servo_to_jpose(J_pinv, null_J, dU_d, dPose_index_in_index, delta_tf, delta_t_tf):
            js_abcd = np.array((0, 1, 2, 3), np.matrix)
            js_abcd[0] = make_J_s_abcd(0, 0)
            js_abcd[1] = make_J_s_abcd(1, 0)
            js_abcd[2] = make_J_s_abcd(0, 1)
            js_abcd[3] = make_J_s_abcd(1, 1)
            dU_1_js = np.array((0, 1, 2, 3), np.matrix)
            dU_2_js = np.matmul(null_J, (dU_d))
            dU_js = np.array((0, 1, 2, 3), np.matrix)
            norm_ddU = np.zeros(4)
            dPose_thumb_in_thumb_js = np.array((0, 1, 2, 3), np.matrix)
            for i in range(4):
                js = np.matrix(js_abcd[i])
                dPose_thumb_in_thumb = np.matmul(pinv_SVD(js), delta_tf*delta_t_tf)
                dPose_thumb_in_thumb = np.vstack((dPose_thumb_in_thumb[0:3,:], dPose_thumb_in_thumb[4:6,:]))
                dPose_thumb_in_thumb_js[i] = dPose_thumb_in_thumb
                dPose = np.vstack((dPose_index_in_index, dPose_thumb_in_thumb))
                dU_1_js[i] = np.matmul(J_pinv, dPose)
                dU_js[i] = dU_1_js[i] + dU_2_js
                norm_ddU[i] = np.linalg.norm((dU_js[i])[6:,:] - dU_d[6:,:])
            i_min = argmin(norm_ddU)
            return dU_1_js[i_min], dU_2_js, dPose_thumb_in_thumb_js[i_min]
        """
        Jacobian: tips twists in its own frames ----> tactile features in each tip
        No!!! I tend to use a resizable J_s, to be defined below.
        """
        # J_s = np.mat(np.zeros((6,12))) # 2*3 X 2*6 (2 fingers)
        # J_s[0:3,0:6] = J_si
        # J_s[3:6,6:12] = J_si
        """
        R_x2y: Rotation matrix: {y} in {x}
        T_x2y: Transformation matrix: {y} in {x}
        RR_x2y: Rotation matrixes X 2: {y} in {x}:
        [[R_x2y  0  ]
         [  0  R_x2y]]
        """
        """
        Rotation matrixes (6 X 6) of tips w.r.t. contacting surfaces (Convert twist between two --relatively static-- frames):
        R_x2y * twist_in_y = twist_in_x
        """
        RR_sur2index = np.mat(np.eye(6))
        RR_sur2mid = np.mat(np.eye(6))
        RR_sur2ring = np.mat(np.eye(6))
        RR_sur2thumb = np.mat(np.eye(6))
        # Kinematics Jacobian (palm twist + fingers joints velocities ----> tips twists)
        """
        To avoid joints limits, joint thumb 0 is fixed, so J_kin_thumb is 6 X 3

        """
        J_kin = np.mat(np.zeros((12,12))) # 2 fingers: thumb + index
        # J_kin = np.mat(np.zeros((18,17))) # 3 fingers: thumb + index + mid
        # J_kin = np.mat(np.zeros((24,21))) # 2 fingers: thumb + index + mid + ring
        """
        Identity matrixes, predefined here (dimension corresponding to J_kin defined above)
        Used for calculating null-space ( null(J) = I - J_pinv * J )
        """
        I_12 = np.mat(np.eye(12))
        I_13 = np.mat(np.eye(13))
        I_17 = np.mat(np.eye(17))
        I_21 = np.mat(np.eye(21))
        """ Jacobian: palm twist + finger joints velocities ----> tip tactile feature rate """
        # J_U2tf = np.mat(np.zeros((6,12))) # 2 fingers: thumb + index
        # J_U2tf = np.mat(np.zeros((6,17))) # 3 fingers: thumb + index + mid
        # J_U2tf = np.mat(np.zeros((6,21))) # 4 fingers: thumb + index + mid + ring
        """ Jacobian: palm twist ----> tip twist """
        J_palmV2indexV = np.mat(np.eye(6))
        J_palmV2midV = np.mat(np.eye(6))
        J_palmV2ringV = np.mat(np.eye(6))
        J_palmV2thumbV = np.mat(np.eye(6))

        """
        Zero-like matrixes for SVD and calculating generalized inverses
        SVD: U * D * V^T = J
        """
        D_kin = np.mat(np.zeros((12,12))) # 2 fingers: thumb + index
        # D_kin = np.mat(np.zeros((18,17))) # 3 fingers: thumb + index + mid
        # D_kin = np.mat(np.zeros((24,21))) # 4 fingers: thumb + index + mid + ring
        D_kin_pinv = np.mat(np.zeros((12,12))) # 2 fingers: thumb + index
        # D_kin_pinv = np.mat(np.zeros((17,18))) # 3 fingers: thumb + index + mid
        # D_kin_pinv = np.mat(np.zeros((21,24))) # 4 fingers: thumb + index + mid + ring

        error_tf = np.mat(np.zeros((6,1)))
        delta_u_d = np.mat(np.zeros((12,1)))

        
        # moving average filter ma
        ma_filter = StreamingMovingAverage_array(24,12)
        tf_stack = np.mat(np.zeros((12,1))) # stacking tactile features

        # timer:
        t_t0 = 0.0
        t_tt = 0.0
        t_tt_last = 0.0
        t_tt_delta = 0.0

        # Plotting:
        t_plot = []
        fxd_index_list = []
        fx_index_list = []
        fyd_index_list = []
        fy_index_list = []
        fzd_index_list = []
        fz_index_list = []
        fx_index_ma_list = []
        fy_index_ma_list = []
        fz_index_ma_list = []
        
        fxd_mid_list = []
        fx_mid_list = []
        fyd_mid_list = []
        fy_mid_list = []
        fzd_mid_list = []
        fz_mid_list = []
        fx_mid_ma_list = []
        fy_mid_ma_list = []
        fz_mid_ma_list = []

        fxd_ring_list = []
        fx_ring_list = []
        fyd_ring_list = []
        fy_ring_list = []
        fzd_ring_list = []
        fz_ring_list = []
        fx_ring_ma_list = []
        fy_ring_ma_list = []
        fz_ring_ma_list = []

        fxd_thumb_list = []
        fx_thumb_list = []
        fyd_thumb_list = []
        fy_thumb_list = []
        fzd_thumb_list = []
        fz_thumb_list = []
        fx_thumb_ma_list = []
        fy_thumb_ma_list = []
        fz_thumb_ma_list = []
        twist_index_r = array_append(6)
        twist_thumb_r = array_append(6)
        twist_index_tservo = array_append(5)
        twist_thumb_tservo = array_append(5)
        twist_palm_list = array_append(6)
        # comparing the IK outpur twist vs the real twist (calculated with diver)
        t_plot_for_twist = []
        twist_index_real_list = array_append(6)
        twist_thumb_real_list = array_append(6)
        twist_index_ma_filter = StreamingMovingAverage_array(4,6)
        twist_index_real_ma_list = array_append(6)
        twist_thumb_ma_filter = StreamingMovingAverage_array(4,6)
        twist_thumb_real_ma_list = array_append(6)
        twist_index_IK_list = array_append(6)
        twist_thumb_IK_list = array_append(6)

        joints_list = np.zeros((0,16))
        joints_d_list = np.zeros((0,16))

        pos_thumb2index_list = np.zeros((0,3))
        posd_thumb2index_list = np.zeros((0,3))
        error_axis_thumb2index = np.zeros((0,2))
        # initial palm pose
        ground_T0_palm = np.mat(np.eye(4))
        ee_state = np.mat(copy.deepcopy(self.ur_eestates)).reshape(7,1)
        ground_T0_palm = pose2T(ee_state)

        # twist presentation
        ground_T_vel_index = np.mat(np.eye(4))
        ground_T_angvel_index = np.mat(np.eye(4))
        ground_T_vel_thumb = np.mat(np.eye(4))
        ground_T_angvel_thumb = np.mat(np.eye(4))
        ground_T_index_last = np.mat(np.eye(4))
        ground_T_thumb_last = np.mat(np.eye(4))
        while not rospy.is_shutdown():
            ## Calc
            # Timing
            self.get_time()
            # get last states
            jstate = np.mat(copy.deepcopy(self.allegro_jstates)).reshape((16,1))
            ee_state = np.mat(copy.deepcopy(self.ur_eestates)).reshape(7,1)
            T_palm = pose2T(ee_state)
            # get last cmds
            jcmd = np.array(self.jcmd_pub.position)
            ee_cmd = ros_pose2new_pose_quat(self.urcmd_pub.pose)
            # print("Time:", self.current_time, "error_tf_mid:", error_tf[0:3,0].T, "error_tf_thumb:", error_tf[3:6,0].T)
            """ Real joint angles FK """
            # index
            kdl_calc_fk(self.index_fk, self.index_qpos, self.index_pos)
            T_palm2index = KDLframe2T(self.index_pos)
            J_palmV2indexV[0:3,3:6] = -cross_product_matrix_from_vector3d(T_palm2index[:3,3])
            RR_palm2index = T2AdT(T_palm2index)
            T_index2palm[:3,:3] = T_palm2index[:3,:3].T
            T_index2palm[:3,3] = -np.matmul(T_index2palm[:3,:3], T_palm2index[:3,3])
            RR_index2palm = T2AdT(T_index2palm)

            # mid
            kdl_calc_fk(self.mid_fk, self.mid_qpos, self.mid_pos)
            T_palm2mid = KDLframe2T(self.mid_pos)
            J_palmV2midV[0:3,3:6] = -cross_product_matrix_from_vector3d(T_palm2mid[:3,3])
            RR_palm2mid = T2AdT(T_palm2mid)
            T_mid2palm[:3,:3] = T_palm2mid[:3,:3].T
            T_mid2palm[:3,3] = -np.matmul(T_mid2palm[:3,:3], T_palm2mid[:3,3])
            RR_mid2palm = T2AdT(T_mid2palm)

            # ring
            kdl_calc_fk(self.ring_fk, self.ring_qpos, self.ring_pos)
            T_palm2ring = KDLframe2T(self.ring_pos)
            J_palmV2ringV[0:3,3:6] = -cross_product_matrix_from_vector3d(T_palm2ring[:3,3])
            RR_palm2ring = T2AdT(T_palm2ring)
            T_ring2palm[:3,:3] = T_palm2ring[:3,:3].T
            T_ring2palm[:3,3] = -np.matmul(T_ring2palm[:3,:3], T_palm2ring[:3,3])
            RR_ring2palm = T2AdT(T_ring2palm)

            # thumb
            kdl_calc_fk(self.thumb_fk, self.thumb_qpos, self.thumb_pos)
            T_palm2thumb = KDLframe2T(self.thumb_pos)
            J_palmV2thumbV[0:3,3:6] = -cross_product_matrix_from_vector3d(T_palm2thumb[:3,3])
            RR_palm2thumb = T2AdT(T_palm2thumb)
            T_thumb2palm[:3,:3] = T_palm2thumb[:3,:3].T
            T_thumb2palm[:3,3] = -np.matmul(T_thumb2palm[:3,:3], T_palm2thumb[:3,3])
            RR_thumb2palm = T2AdT(T_thumb2palm)
            """ Tactile feature errors: """
            # index
            index_contact, index_x_set, index_y_set = detect_contact(self.index_tacxel_pose_euler, self.index_tip)
            if index_contact:
                tac_values = np.reshape(np.mat(copy.deepcopy(index_y_set)), (1,-1))
                tac_sum = np.sum(tac_values)
                index_tf[0,0] = tac_sum
                # index_tf[1,0] = np.average(index_x_set[:,1])*1e2
                # index_tf[2,0] = np.average(index_x_set[:,2])*1e2
                index_tf[1,0] = np.matmul(tac_values, index_x_set[:,1])[0,0]/tac_sum
                index_tf[2,0] = np.matmul(tac_values, index_x_set[:,2])[0,0]/tac_sum
                pose_in_index = calc_pose_in_tip(index_x_set, index_y_set)
                R_sur2index = Rotation.from_euler("xyz", pose_in_index[3:6]).as_dcm().T
                RR_sur2index[0:3,0:3] = R_sur2index
                RR_sur2index[3:6,3:6] = R_sur2index
            else: # if not contact, expected vel_x_in_surface is set as default
                # print(self.current_time, ": index tip loses contact...")
                index_tf[0,0] = -50.0
                # index_tf[1,0] = 0.0
                # index_tf[2,0] = 8.5e-4
            

            # mid
            mid_contact, mid_x_set, mid_y_set = detect_contact(self.index_tacxel_pose_euler, self.mid_tip)
            if mid_contact:
                tac_values = np.reshape(np.mat(copy.deepcopy(mid_y_set)), (1,-1))
                tac_sum = np.sum(tac_values)
                mid_tf[0,0] = tac_sum
                # mid_tf[1,0] = np.average(mid_x_set[:,1])*1e2
                # mid_tf[2,0] = np.average(mid_x_set[:,2])*1e2
                mid_tf[1,0] = np.matmul(tac_values, mid_x_set[:,1])[0,0]/tac_sum
                mid_tf[2,0] = np.matmul(tac_values, mid_x_set[:,2])[0,0]/tac_sum
                pose_in_mid = calc_pose_in_tip(mid_x_set, mid_y_set)
                R_sur2mid = Rotation.from_euler("xyz", pose_in_mid[3:6]).as_dcm().T
                RR_sur2mid[0:3,0:3] = R_sur2mid
                RR_sur2mid[3:6,3:6] = R_sur2mid
                # print("RR_sur2mid:\n", RR_sur2mid)
            else: # if not contact, expected vel_x_in_surface is set as default
                # print(self.current_time, ": Mid tip loses contact...")
                mid_tf[0,0] = -50.0
                # mid_tf[1,0] = 0.0
                # mid_tf[2,0] = 8.5e-4
            

            # ring
            ring_contact, ring_x_set, ring_y_set = detect_contact(self.index_tacxel_pose_euler, self.ring_tip)
            if ring_contact:
                tac_values = np.reshape(np.mat(copy.deepcopy(ring_y_set)), (1,-1))
                tac_sum = np.sum(tac_values)
                ring_tf[0,0] = tac_sum
                # ring_tf[1,0] = np.average(ring_x_set[:,1])*1e2
                # ring_tf[2,0] = np.average(ring_x_set[:,2])*1e2
                ring_tf[1,0] = np.matmul(tac_values, ring_x_set[:,1])[0,0]/tac_sum
                ring_tf[2,0] = np.matmul(tac_values, ring_x_set[:,2])[0,0]/tac_sum
                pose_in_ring = calc_pose_in_tip(ring_x_set, ring_y_set)
                R_sur2ring = Rotation.from_euler("xyz", pose_in_ring[3:6]).as_dcm().T
                RR_sur2ring[0:3,0:3] = R_sur2ring
                RR_sur2ring[3:6,3:6] = R_sur2ring
                # print("RR_sur2ring:\n", RR_sur2ring)
            else: # if not contact, expected vel_x_in_surface is set as default
                # print(self.current_time, ": ring tip loses contact...")
                ring_tf[0,0] = -50.0
                # ring_tf[1,0] = 0.0
                # ring_tf[2,0] = 8.5e-4
            
            
            # thumb:
            thumb_contact, thumb_x_set, thumb_y_set = detect_contact(self.index_tacxel_pose_euler, self.thumb_tip)
            if thumb_contact:
                tac_values = np.reshape(np.mat(copy.deepcopy(thumb_y_set)), (1,-1))
                tac_sum = np.sum(tac_values)
                thumb_tf[0,0] = np.sum(thumb_y_set)
                # thumb_tf[1,0] = np.average(thumb_x_set[:,1])*1e2
                # thumb_tf[2,0] = np.average(thumb_x_set[:,2])*1e2
                thumb_tf[1,0] = np.matmul(tac_values, thumb_x_set[:,1])[0,0]/tac_sum
                thumb_tf[2,0] = np.matmul(tac_values, thumb_x_set[:,2])[0,0]/tac_sum
                pose_in_thumb = calc_pose_in_tip(thumb_x_set, thumb_y_set)
                R_sur2thumb = Rotation.from_euler("xyz", pose_in_thumb[3:6]).as_dcm().T
                RR_sur2thumb[0:3,0:3] = R_sur2thumb
                RR_sur2thumb[3:6,3:6] = R_sur2thumb
                # print("RR_sur2thumb:\n", RR_sur2thumb)
            else: # if not contact, expected vel_x_in_surface is set as default
                #   of cause we should note that pose_in_thumb is set to zeros if not contact
                # print(self.current_time, ": Thumb tip loses contact...")
                thumb_tf[0,0] = -50.0
                # thumb_tf[1,0] = 0.0
                # thumb_tf[2,0] = 8.5e-4
            

            """MA (moving average) process"""
            tf_stack[0:3,0] = index_tf[0:3,0]
            tf_stack[3:6,0] = mid_tf[0:3,0]
            tf_stack[6:9,0] = ring_tf[0:3,0]
            tf_stack[9:12,0] = thumb_tf[0:3,0]
            tf_ma_stack = ma_filter.process(tf_stack)

            # error_tactile_feature
            # (1) Raw feature:
            # # error_tf[0:3,0] = (index_tf_d - index_tf)
            # error_tf[0:3,0] = (mid_tf_d - mid_tf)
            # # error_tf[6:9,0] = (ring_tf_d - ring_tf)
            # error_tf[3:6,0] = (thumb_tf_d - thumb_tf)
            # (2) Feature after moving average filter
            error_tf[0:3,0] = index_tf_d - tf_ma_stack[0:3,0]
            error_tf[3:6,0] = thumb_tf_d - tf_ma_stack[9:12,0]
            # error_tf[0,0] = 0.0
            # error_tf[1,0] = 0.0
            # error_tf[2,0] = 0.0
            # error_tf[3,0] = 0.0
            # error_tf[4,0] = 0.0
            # error_tf[5,0] = 0.0

            """ Calculate desired twists with tactile features errors """
            
            T_index2indexd = np.matmul(np.linalg.inv(T_palm2index), np.matmul(T_palm2thumb, Td_thumb2index))
            T_index2indexd_js = servo_to_T(T_index2indexd, error_tf[0:3,0], self.delta_time)
            # T_index2indexd_js = servo_to_T(T_index2indexd, error_tf[0:3,0], 0.02)
            # print("T_index2indexd_js[0:3,3]:\n", T_index2indexd_js[0:3,3])
            dtheta_x, dtheta_y, dtheta_z = angular_vel_from_R(T_index2indexd_js[:3,:3], 1.0)
            dOri_index_in_index = np.mat([[dtheta_x, dtheta_y, dtheta_z]]).T
            # print("dOri_index_in_index:\n", dOri_index_in_index)
            dPose_index_in_index = np.vstack((T_index2indexd_js[0:3,3].copy(), dOri_index_in_index))
            dPose_index_in_index = np.vstack((dPose_index_in_index[0:3,:], dPose_index_in_index[4:6,:]))

            if dPose_index_in_index[0,0] > 5e-4:
                dPose_index_in_index[0,0] = 5e-4
                print("Index Max X")
            elif dPose_index_in_index[0,0] < -5e-4:
                dPose_index_in_index[0,0] = -5e-4
                print("Index Min X")

            if dPose_index_in_index[1,0] > 1e-4:
                dPose_index_in_index[1,0] = 1e-4
                print("Index Max Y")
            elif dPose_index_in_index[1,0] < -1e-4:
                dPose_index_in_index[1,0] = -1e-4
                print("Index Min Y")

            if dPose_index_in_index[2,0] > 1e-4:
                dPose_index_in_index[2,0] = 1e-4
                print("Index Max Z")
            elif dPose_index_in_index[2,0] < -1e-4:
                dPose_index_in_index[2,0] = -1e-4
                print("Index Min Z")

            # RR_palm2index = np.mat(np.eye(6))
            # RR_palm2index[0:3,0:3] = T_palm2index[:3,:3].copy()
            # RR_palm2index[3:6,3:6] = T_palm2index[:3,:3].copy()
            # dPose_index_in_palm = np.matmul(RR_palm2index, dPose_index_in_index)

            
            
            # Finger kinematics Jacobians
            # index_jac = self.index_kdl_kin.jacobian(jstate[0:4])
            # index
            J_kin_index = copy.deepcopy(self.index_kdl_kin.jacobian(jstate[0:4]))
            J_kin_index = np.mat(J_kin_index)
            J_kin_index = J_kin_index[:,1:4]
            # mid
            J_mid = copy.deepcopy(self.mid_kdl_kin.jacobian(jstate[4:8]))
            J_mid = np.mat(J_mid)
            # ring
            J_ring = copy.deepcopy(self.ring_kdl_kin.jacobian(jstate[8:12]))
            J_ring = np.mat(J_ring)
            # thumb
            J_kin_thumb = copy.deepcopy(self.thumb_kdl_kin.jacobian(jstate[12:16]))
            J_kin_thumb = np.mat(J_kin_thumb) # Do not move the joint 0 of thumb
            J_kin_thumb = J_kin_thumb[:,1:4]

            """In each loop, calculate the kinematics Jacobian for palm twist + finger joints velocities ----> tips twists in {palm} frame """
            J_kin[0:6,0:6] = J_palmV2indexV
            J_kin[6:12,0:6] = J_palmV2thumbV
            J_kin[0:6,6:9] = J_kin_index
            J_kin[6:12,9:12] = J_kin_thumb
            """ 
            Now we have kinematics Jacobian for palm twist + finger joints velocities ----> tips twists, all twists in {palm} frame
            However, we want finger tips twists in each {tip}'s frame:
            """
            J_kin[0:6,0:6] = np.matmul(RR_index2palm, J_kin[0:6,0:6])
            J_kin[0:6,6:9] = np.matmul(RR_index2palm, J_kin[0:6,6:9])
            J_kin[6:12,0:6] = np.matmul(RR_thumb2palm, J_kin[6:12,0:6])
            J_kin[6:12,9:12] = np.matmul(RR_thumb2palm, J_kin[6:12,9:12])

            """ take out 2 rows """
            J_kin_t = np.vstack((J_kin[0:3,:],J_kin[4:9,:]))
            J_kin_t = np.vstack((J_kin_t, J_kin[10:12,:]))

            """ Update the weights of the controlled variables """
            k_sq = 1.0
            half_k_sq = k_sq*0.5
            delta_index = np.abs(target_pose[1:4] - jstate[1:4])
            w_index = np.reshape(np.array(delta_index/self.index_ranges[0,1:4].T*k_sq), 3) + np.ones(3)*half_k_sq
            # w_index[2] = w_index[2] + 0.35 # trick for largeshaker
            w_index[2] = w_index[2] + 0.0 # trick for 
            delta_thumb = np.abs(target_pose[13:16] - jstate[13:16])
            w_thumb = np.reshape(np.array(delta_thumb/self.thumb_ranges[0,1:4].T*k_sq), 3) + np.ones(3)*half_k_sq
            # w_thumb[0] = w_thumb[0] +0.35 # trick for largeshaker
            # w_thumb[2] = w_thumb[2] +0.35 # trick for largeshaker
            w_thumb[0] = w_thumb[0] +0.0 # trick for 
            w_thumb[2] = w_thumb[2] +0.0 # trick for 
            # print("w_palm:", w_palm)
            # print("w_index:", w_index)
            # print("w_thumb:", w_thumb)
            w_diag = np.concatenate((w_palm, w_index))
            w_diag = np.concatenate((w_diag, w_thumb))
            w_diag = np.ones_like(w_diag) # Cancel weights
            print("w_diag:", w_diag)
            # W_diag = diag(w_diag)
            w_diag_inv_sqr = np.sqrt(1/w_diag)
            W_inv_sqr = np.diag(w_diag_inv_sqr)
            J_kin_w = np.matmul(J_kin_t, W_inv_sqr)
            """
            My special IK (Inverse Tactile feature):
            (1) q_vel = alpha*J^T*e + beta*(I_14x14 - J_pinv*J)*(q^* - q)
            alpha = 
            (2) q_vel = J_pinv*e + beta*(I_14x14 - J_pinv*J)*(q^* - q)
            How to get J_pinv?
                1) singular value decomposition of J: J = U * D * V^T
                2) Let D_pinv be the M-P-pseudoinverse of D, then the entries of D_pinv are:
                    d_pinv_ii = 1/d_ii (if d_ii != 0) or 0 (if d_ii == 0)
                3) then, J_pinv = V * D_pinv * U^T
            """
            k_pos = 0.0
            k_ori = 10.0
            T0_in_T = np.matmul(invT(T_palm), ground_T0_palm)
            delta_u_d[0:3,:] = k_pos*(ground_T0_palm[:3,3]-T_palm[:3,3])
            wx, wy, wz = angular_vel_from_R(T0_in_T[:3,:3],dt=1.0/k_ori)
            delta_u_d[3:6,:] = np.mat([wx, wy, wz]).T
            

            delta_u_d[0:3,:] = np.matmul(T_palm[:3,:3].T, delta_u_d[0:3,:])
            delta_u_d[0:5,:] = delta_u_d[0:5,:]*0.0
            delta_u_d[5,:] = delta_u_d[5,:]
            delta_u_d[6:9,0] = (target_pose[1:4,0] - jstate[1:4,0])*2.0
            delta_u_d[9:12,0] = (target_pose[13:16,0] - jstate[13:16,0])*2.0

            
            # delta_u_d[9,0] = 1.25*delta_u_d[8,0] # trick
            # delta_u_d[9,0] = 1.25*delta_u_d[9,0] # trick
            # delta_u_d[11,0] = 1.25*delta_u_d[11,0] # trick
            # print("delta_u_d:", delta_u_d.T)
            # alpha = 3.5 # largeshaker
            # beta = 0.5*self.delta_time # largeshaker

            alpha = 0.2
            beta = 0.02*self.delta_time
            # beta = 0.02
            J_kin_w_pinv, J_kin_w = pinv_TSVD(J_kin_w,1000.)
            J_kin_t_pinv, J_kin_t = pinv_TSVD(J_kin_t, 1000.)
            # null-space 1:
            # null_J = I_12 - np.matmul(J_kin_t_pinv, J_kin_t)
            # null-space 2:
            null_J = I_12 - np.matmul(J_kin_w_pinv, J_kin_w)
            null_J = np.matmul(W_inv_sqr, null_J)
            delta_u_1, delta_u_2, dPose_thumb_in_thumb = servo_to_jpose(J_kin_w_pinv, null_J, delta_u_d, dPose_index_in_index, error_tf[3:6,:], self.delta_time)
            # null-space 3: (This method is wrong, the 2 methods above is right)
            # null_J = I_12 - np.matmul(J_kin_w_pinv, J_kin_w)
            # delta_u_1 = np.matmul(J_kin_w_pinv, np.vstack((dPose_index_in_index, dPose_thumb_in_thumb)))
            delta_u_1 = np.matmul(W_inv_sqr, delta_u_1)
            delta_u_1 = alpha*delta_u_1
            delta_u_2 = beta*delta_u_2
            delta_u = delta_u_1 + delta_u_2
            print("Rate of u1/u2:", (delta_u_1/delta_u_2).T)
            print("delta_u_1:", delta_u_1.T)
            print("delta_u_2:", delta_u_2.T)
            print("delta_u:", delta_u.T)
            # if wz < -0.1:
            #     if delta_u[5,0] > 0:
            #         delta_u[5,0] = 0
            # delta_u = np.matmul(J_kin_t_pinv, np.vstack((dPose_index_in_index, dPose_thumb_in_thumb)))
            # print("delta_u:", delta_u.T)
            # print("delta_u[5,0]:", delta_u[5,0])
            # print("J_U2tf_pinv:\n", J_U2tf_pinv)
            
            # print("isnan:", np.isnan(np.sum(J_U2tf_pinv)))
            # print("e:", e.T)
            # print("delta_u:", delta_u.T)
            # print("delta_u:", delta_u.T)
            # delta_u = np.dot(alpha, np.matmul(J_U2tf_pinv, e)) # (3)
            # delta_u = np.matmul(W_inv_sqr, delta_u)
            delta_q = np.mat(np.zeros((16,1)))
            delta_q[1:4,0] = delta_u[6:9,0]
            delta_q[13:16,0] = delta_u[9:12,0]
            delta_q = np.array(delta_q).T.flatten()
            jcmd_0 = jcmd
            jcmd = jcmd + delta_q
            jcmd[0] = target_pose[0,0]
            jcmd[12] = target_pose[12,0]
            jcmd = self.write_publish_allegro_cmd(jcmd)
            """ Check IK output twist """
            delta_q_real = np.mat(copy.deepcopy(jcmd - jcmd_0)).T
            delta_u_real = delta_u.copy()
            delta_u_real[6:9,:] = delta_q_real[1:4,:]
            delta_u_real[9:12,:] = delta_q_real[13:16,:]
            twist_real = np.matmul(J_kin, delta_u_real)
            T_below_x = pose2T([-0.05,0,0,0,0,0,1])
            # index
            twist_index_real = (twist_real.copy())[0:6,:]
            # twist_index_wx = twist_real[3,0]
            ground_T_index = np.matmul(T_palm, T_palm2index)
            # ground_T_twist_index = np.matmul(ground_T_index, T_below_x)
            norm_vel_index = np.linalg.norm(twist_index_real[0:3,:])
            if norm_vel_index > 0:
                ground_T_vel_index[:3,0] = np.matmul(ground_T_index[:3,:3], twist_index_real[0:3,:])/norm_vel_index
                ground_T_vel_index[:3,1] = np.mat(unit_orthogonal_vector3d(ground_T_vel_index[:3,0])).T
                ground_T_vel_index[:3,2] = np.cross(ground_T_vel_index[:3,0].T, ground_T_vel_index[:3,1].T).T
                ground_T_vel_index[:3,3] = ground_T_index[:3,3].copy()
                pose_vel_index = T2pose(ground_T_vel_index)
                self.index_vel_pub.pose = pose_quat2new_ros_pose(pose_vel_index)
                self.index_vel_publisher.publish(self.index_vel_pub)
            norm_angvel_index = np.linalg.norm(twist_index_real[3:6,:])
            if norm_angvel_index > 0:
                ground_T_angvel_index[:3,0] = np.matmul(ground_T_index[:3,:3], twist_index_real[3:6,:])/norm_angvel_index
                ground_T_angvel_index[:3,1] = np.mat(unit_orthogonal_vector3d(ground_T_angvel_index[:3,0])).T
                ground_T_angvel_index[:3,2] = np.cross(ground_T_angvel_index[:3,0].T, ground_T_angvel_index[:3,1].T).T
                ground_T_index_b = np.matmul(ground_T_index, T_below_x)
                ground_T_angvel_index[:3,3] = ground_T_index_b[:3,3].copy()
                # ground_T_angvel_index = np.matmul(ground_T_angvel_index, T_below_x)
                pose_angvel_index = T2pose(ground_T_angvel_index)
                self.index_angvel_pub.pose = pose_quat2new_ros_pose(pose_angvel_index)
                self.index_angvel_publisher.publish(self.index_angvel_pub)

            # thumb
            twist_thumb_real = (twist_real.copy())[6:12,:]
            # twist_thumb_wx = twist_real[3,0]
            ground_T_thumb = np.matmul(T_palm, T_palm2thumb)
            # ground_T_twist_thumb = np.matmul(ground_T_thumb, T_below_x)
            norm_vel_thumb = np.linalg.norm(twist_thumb_real[0:3,:])
            if norm_vel_thumb > 0:
                ground_T_vel_thumb[:3,0] = np.matmul(ground_T_thumb[:3,:3], twist_thumb_real[0:3,:])/norm_vel_thumb
                ground_T_vel_thumb[:3,1] = np.mat(unit_orthogonal_vector3d(ground_T_vel_thumb[:3,0])).T
                ground_T_vel_thumb[:3,2] = np.cross(ground_T_vel_thumb[:3,0].T, ground_T_vel_thumb[:3,1].T).T
                ground_T_vel_thumb[:3,3] = ground_T_thumb[:3,3].copy()
                pose_vel_thumb = T2pose(ground_T_vel_thumb)
                self.thumb_vel_pub.pose = pose_quat2new_ros_pose(pose_vel_thumb)
                self.thumb_vel_publisher.publish(self.thumb_vel_pub)
            norm_angvel_thumb = np.linalg.norm(twist_thumb_real[3:6,:])
            if norm_angvel_thumb > 0:
                ground_T_angvel_thumb[:3,0] = np.matmul(ground_T_thumb[:3,:3], twist_thumb_real[3:6,:])/norm_angvel_thumb
                ground_T_angvel_thumb[:3,1] = np.mat(unit_orthogonal_vector3d(ground_T_angvel_thumb[:3,0])).T
                ground_T_angvel_thumb[:3,2] = np.cross(ground_T_angvel_thumb[:3,0].T, ground_T_angvel_thumb[:3,1].T).T
                ground_T_thumb_b = np.matmul(ground_T_thumb, T_below_x)
                ground_T_angvel_thumb[:3,3] = ground_T_thumb_b[:3,3].copy()
                # ground_T_angvel_thumb = np.matmul(ground_T_angvel_thumb, T_below_x)
                pose_angvel_thumb = T2pose(ground_T_angvel_thumb)
                self.thumb_angvel_pub.pose = pose_quat2new_ros_pose(pose_angvel_thumb)
                self.thumb_angvel_publisher.publish(self.thumb_angvel_pub)

            # thumb

            # twist_thumb_wx = twist_real[9,0]
            # ground_T_thumb = np.matmul(T_palm, T_palm2thumb)
            # ground_T_twist_thumb = np.matmul(ground_T_thumb, T_below_x)
            # if twist_thumb_wx < 0:
            #     ground_T_twist_thumb[:3,:2] = -ground_T_twist_thumb[:3,:2]

            # publish the twist of wx:
            # index
            
            
            # thumb
            # pose_twist_thumb = T2pose(ground_T_twist_thumb)
            # self.thumb_twist_pub.pose = pose_quat2new_ros_pose(pose_twist_thumb)
            # self.thumb_twist_publisher.publish(self.thumb_twist_pub)

            """ calc, collect, and then print tip twists """
            twist_index = np.matmul(J_palmV2indexV, delta_u[0:6,0]) + np.matmul(J_kin_index, delta_u[6:9,0])
            twist_index = np.matmul(RR_index2palm, twist_index)
            print("dPose_index_in_index*10000 (dPose by tactile servoing):", 1e4*alpha*dPose_index_in_index.T)
            print("twist_index*10000 (dPose by pinv_Jacobian):",1e4*twist_index.T)
            # print("twist_index_real*10000 (dPose considering joint state vs cmd max error):", 1e4*twist_real[0:6,:].T)
            print("solution_error:\n\n\n", (dPose_index_in_index[0:3,:].T-twist_index[0:3,:].T), (dPose_index_in_index[3:5,:].T-twist_index[4:6,:].T))
            
            
            # delta_tf_index = np.matmul(J_si, twist_index)
            twist_thumb = np.matmul(J_palmV2thumbV, delta_u[0:6,0]) + np.matmul(J_kin_thumb, delta_u[9:12,0])
            twist_thumb = np.matmul(RR_thumb2palm, twist_thumb)
            print("dPose_thumb_in_thumb*10000 (dPose by tactile servoing):", 1e4*alpha*dPose_thumb_in_thumb.T)
            print("twist_thumb*10000 (dPose by pinv_Jacobian):",1e4*twist_thumb.T)
            # print("twist_thumb_real*10000 (dPose considering joint state vs cmd max error):", 1e4*twist_real[6:12,:].T)
            print("solution_error:\n\n\n", (dPose_thumb_in_thumb[0:3,:].T-twist_thumb[0:3,:].T), (dPose_thumb_in_thumb[3:5,:].T-twist_thumb[4:6,:].T))

            # delta_tf_thumb = np.matmul(J_si, twist_thumb)
            T_ee = pose2T(ee_cmd)
            palm_dpose_in_world = np.matmul(T2AdT(T_ee), delta_u[0:6,0])
            # ee_state is now an np.mat of 7x1
            # Watch out here ...
            ee_cmd = delta_pose2next_pose(np.array(palm_dpose_in_world).T.flatten(), ee_cmd)
            ee_cmd[:3], ee_cmd[3:] = self.write_publish_ee_cmd(ee_cmd[:3], ee_cmd[3:])
            # print delta_tf caused by delta_u
            if np.linalg.norm(error_tf[1:3]) + np.linalg.norm(error_tf[4:6]) < 0.01:
                print("threshold:", np.linalg.norm(error_tf[1:3]) + np.linalg.norm(error_tf[4:6]))
                print("index qpose:",jstate[0:4,0].T)
                print("index qpose error:", target_pose[0:4,:].T - jstate[0:4,:].T)
                print("thumb qpose:",jstate[12:16,0].T)
                print("thumb qpose error:", target_pose[12:16,:].T - jstate[12:16,:].T)
            
            # if t_tt - t_t0 > 60.0:
            if (np.linalg.norm(error_tf[1:3]) + np.linalg.norm(error_tf[4:6]) < 0.0056) or (t_tt - t_t0 > 15.0) or (self.index_reach_limit == True) or (self.thumb_reach_limit == True):
                print("At", t_tt - t_t0, "Plot...")
                print("threshold:", np.linalg.norm(error_tf[1:3]) + np.linalg.norm(error_tf[4:6]))
                print("index qpose:",jstate[0:4,0].T)
                print("index qpose error:", target_pose[0:4,:].T - jstate[0:4,:].T)
                print("thumb qpose:",jstate[12:16,0].T)
                print("thumb qpose error:", target_pose[12:16,:].T - jstate[12:16,:].T)
                # index
                fig1, axs1 = plt.subplots(3)
                # fx:
                axs1[0].plot(t_plot, fx_index_list, label='$fx_{real}$')
                axs1[0].plot(t_plot, fxd_index_list, label='$fx_{desired}$')
                axs1[0].plot(t_plot, fx_index_ma_list, label='$fx_{movingaverage}$')
                axs1[0].legend()
                axs1[0].set_xlabel("t / s")
                axs1[0].set_ylabel("$f_x$")
                axs1[0].set_ylim([-50, 1600])
                fig1.suptitle('Index Tactile Feature')
                # fy:
                # axs1[1].plot(t_plot, fy_index_list, label='$fy_{real}$')
                axs1[1].plot(t_plot, fyd_index_list, label='$fy_{desired}$')
                axs1[1].plot(t_plot, fy_index_ma_list, label='$fy_{ma}$')
                axs1[1].legend()
                axs1[1].set_xlabel("t / s")
                axs1[1].set_ylabel("$f_y$ / m")
                # axs1[1].set_ylim([min([min(fy_index_list),min(fy_index_ma_list)]), max([max(fy_index_list),max(fy_index_ma_list)])])
                # fz:
                # axs1[2].plot(t_plot, fz_index_list, label='$fz_{real}$')
                axs1[2].plot(t_plot, fzd_index_list, label='$fz_{desired}$')
                axs1[2].plot(t_plot, fz_index_ma_list, label='$fz_{movingaverage}$')
                axs1[2].legend()
                axs1[2].set_xlabel("t / s")
                axs1[2].set_ylabel("$f_z$ / m")
                plt.savefig(self.this_file_dir+"/datas/Index_Tactile_Feature.png")
                # axs1[2].set_ylim([min([min(fz_index_list),min(fzd_index_list)]), max([max(fz_index_list),max(fzd_index_list)])])
                # thumb
                fig2, axs2 = plt.subplots(3)
                fig2.suptitle('Thumb Tactile Feature')
                # fx:
                axs2[0].plot(t_plot, fx_thumb_list, label='$fx_{real}$')
                axs2[0].plot(t_plot, fxd_thumb_list, label='$fx_{desired}$')
                axs2[0].plot(t_plot, fx_thumb_ma_list, label='$fx_{movingaverage}$')
                axs2[0].legend()
                axs2[0].set_xlabel("t / s")
                axs2[0].set_ylabel("$f_x$ / m")
                axs2[0].set_ylim([-50, 1600])
                # fy:
                # axs2[1].plot(t_plot, fy_thumb_list, label='$fy_{real}$')
                axs2[1].plot(t_plot, fyd_thumb_list, label='$fy_{desired}$')
                axs2[1].plot(t_plot, fy_thumb_ma_list, label='$fy_{ma}$')
                axs2[1].legend()
                axs2[1].set_xlabel("t / s")
                axs2[1].set_ylabel("$f_y$ / m")
                # axs2[1].set_ylim([min([min(fy_thumb_list),min(fyd_thumb_list)]), max([max(fy_thumb_list),max(fyd_thumb_list)])])
                # fz:
                # axs2[2].plot(t_plot, fz_thumb_list, label='$fz_{real}$')
                axs2[2].plot(t_plot, fzd_thumb_list, label='$fz_{desired}$')
                axs2[2].plot(t_plot, fz_thumb_ma_list, label='$fz_{movingaverage}$')
                axs2[2].legend()
                axs2[2].set_xlabel("t / s")
                axs2[2].set_ylabel("$f_z$ / m")
                # axs2[2].set_ylim([min([min(fz_thumb_list),min(fzd_thumb_list)]), max([max(fz_thumb_list),max(fzd_thumb_list)])])
                plt.savefig(self.this_file_dir+"/datas/Thumb_Tactile_Feature.png")
                # twist
                # index
                fig3, axs3 = plt.subplots(6)
                fig3.suptitle("Index Twist")
                axs3[0].plot(t_plot, twist_index_r.datas[0,:].T, label='$\\Delta t\\bullet$$velX_{IK-solution}$')
                axs3[0].plot(t_plot, twist_index_tservo.datas[0,:].T, 'r--', label='$\\Delta t\\bullet$$velX_{tac-servo}$')
                axs3[0].legend()
                axs3[1].plot(t_plot, twist_index_r.datas[1,:].T, label='$\\Delta t\\bullet$$velY_{IK-solution}$')
                axs3[1].plot(t_plot, twist_index_tservo.datas[1,:].T, 'r--', label='$\\Delta t\\bullet$$velY_{tac-servo}$')
                axs3[1].legend()
                axs3[2].plot(t_plot, twist_index_r.datas[2,:].T, label='$\\Delta t\\bullet$$velZ_{IK-solution}$')
                axs3[2].plot(t_plot, twist_index_tservo.datas[2,:].T, 'r--', label='$\\Delta t\\bullet$$velZ_{tac-servo}$')
                axs3[2].legend()
                axs3[3].plot(t_plot, twist_index_r.datas[3,:].T, label='$\\Delta t\\bullet$$angularVelX_{IK-solution}$')
                # axs3[3].plot(t_plot, twist_index_tservo.datas[3,:].T, label='$\\Delta t\\bullet$$angularVelX_{tac-servo}$')
                axs3[3].legend()
                axs3[4].plot(t_plot, twist_index_r.datas[4,:].T, label='$\\Delta t\\bullet$$angularVelY_{IK-solution}$')
                axs3[4].plot(t_plot, twist_index_tservo.datas[3,:].T, 'r--', label='$\\Delta t\\bullet$$angularVelY_{tac-servo}$')
                axs3[4].legend()
                axs3[5].plot(t_plot, twist_index_r.datas[5,:].T, label='$\\Delta t\\bullet$$angularVelZ_{IK-solution}$')
                axs3[5].plot(t_plot, twist_index_tservo.datas[4,:].T, 'r--', label='$\\Delta t\\bullet$$angularVelZ_{tac-servo}$')
                axs3[5].legend()
                plt.savefig(self.this_file_dir+"/datas/Index_Twist.png")
                # thumb
                fig4, axs4 = plt.subplots(6)
                fig4.suptitle("Thumb Twist")
                axs4[0].plot(t_plot, twist_thumb_r.datas[0,:].T, label='$\\Delta t\\bullet$$velX_{IK-solution}$')
                axs4[0].plot(t_plot, twist_thumb_tservo.datas[0,:].T, 'r--', label='$\\Delta t\\bullet$$velX_{tac-servo}$')
                axs4[0].legend()
                axs4[1].plot(t_plot, twist_thumb_r.datas[1,:].T, label='$\\Delta t\\bullet$$velY_{IK-solution}$')
                axs4[1].plot(t_plot, twist_thumb_tservo.datas[1,:].T, 'r--', label='$\\Delta t\\bullet$$velY_{tac-servo}$')
                axs4[1].legend()
                axs4[2].plot(t_plot, twist_thumb_r.datas[2,:].T, label='$\\Delta t\\bullet$$velZ_{IK-solution}$')
                axs4[2].plot(t_plot, twist_thumb_tservo.datas[2,:].T, 'r--', label='$\\Delta t\\bullet$$velZ_{tac-servo}$')
                axs4[2].legend()
                axs4[3].plot(t_plot, twist_thumb_r.datas[3,:].T, label='$\\Delta t\\bullet$$angularVelX_{IK-solution}$')
                # axs4[3].plot(t_plot, twist_thumb_tservo.datas[3,:].T, label='$\\Delta t\\bullet$$angularVelX_{tac-servo}$')
                axs4[3].legend()
                axs4[4].plot(t_plot, twist_thumb_r.datas[4,:].T, label='$\\Delta t\\bullet$$angularVelY_{IK-solution}$')
                axs4[4].plot(t_plot, twist_thumb_tservo.datas[3,:].T, 'r--', label='$\\Delta t\\bullet$$angularVelY_{tac-servo}$')
                axs4[4].legend()
                axs4[5].plot(t_plot, twist_thumb_r.datas[5,:].T, label='$\\Delta t\\bullet$$angularVelZ_{IK-solution}$')
                axs4[5].plot(t_plot, twist_thumb_tservo.datas[4,:].T, 'r--', label='$\\Delta t\\bullet$$angularVelZ_{tac-servo}$')
                axs4[5].legend()
                plt.savefig(self.this_file_dir+"/datas/Thumb_Twist.png")
                # index joints angles
                fig5, axs5 = plt.subplots(4)
                fig5.suptitle("Index Joints States")
                axs5[0].plot(t_plot, joints_list[:,0], label='$index0_{real}$')
                axs5[0].plot(t_plot, joints_d_list[:,0], label='$index0_{desired}$')
                axs5[0].set_ylim([self.index_limits[1,0], self.index_limits[0,0]])
                axs5[0].legend()
                axs5[1].plot(t_plot, joints_list[:,1], label='$index1_{real}$')
                axs5[1].plot(t_plot, joints_d_list[:,1], label='$index1_{desired}$')
                axs5[1].set_ylim([self.index_limits[1,1], self.index_limits[0,1]])
                axs5[1].legend()
                axs5[2].plot(t_plot, joints_list[:,2], label='$index2_{real}$')
                axs5[2].plot(t_plot, joints_d_list[:,2], label='$index2_{desired}$')
                axs5[2].set_ylim([self.index_limits[1,2], self.index_limits[0,2]])
                axs5[2].legend()
                axs5[3].plot(t_plot, joints_list[:,3], label='$index3_{real}$')
                axs5[3].plot(t_plot, joints_d_list[:,3], label='$index3_{desired}$')
                axs5[3].set_ylim([self.index_limits[1,3], self.index_limits[0,3]])
                axs5[3].legend()
                plt.savefig(self.this_file_dir+"/datas/Index_Joint_Angles.png")
                # thumb joints angles
                fig6, axs6 = plt.subplots(4)
                fig6.suptitle("Thumb Joints States")
                axs6[0].plot(t_plot, joints_list[:,12], label='$thumb0_{real}$')
                axs6[0].plot(t_plot, joints_d_list[:,12], label='$thumb0_{desired}$')
                axs6[0].set_ylim([self.thumb_limits[1,0], self.thumb_limits[0,0]])
                axs6[0].legend()
                axs6[1].plot(t_plot, joints_list[:,13], label='$thumb1_{real}$')
                axs6[1].plot(t_plot, joints_d_list[:,13], label='$thumb1_{desired}$')
                axs6[1].set_ylim([self.thumb_limits[1,1], self.thumb_limits[0,1]])
                axs6[1].legend()
                axs6[2].plot(t_plot, joints_list[:,14], label='$thumb2_{real}$')
                axs6[2].plot(t_plot, joints_d_list[:,14], label='$thumb2_{desired}$')
                axs6[2].set_ylim([self.thumb_limits[1,2], self.thumb_limits[0,2]])
                axs6[2].legend()
                axs6[3].plot(t_plot, joints_list[:,15], label='$thumb3_{real}$')
                axs6[3].plot(t_plot, joints_d_list[:,15], label='$thumb3_{desired}$')
                axs6[3].set_ylim([self.thumb_limits[1,3], self.thumb_limits[0,3]])
                axs6[3].legend()
                plt.savefig(self.this_file_dir+"/datas/Thumb_Joint_Angles.png")
                # index in thumb pose
                fig7, axs7 = plt.subplots(4)
                fig7.suptitle("Index-in-Thumb Pose")
                axs7[0].plot(t_plot, pos_thumb2index_list[:,0], label='$^{thumb}X_{index_{real}}$')
                axs7[0].plot(t_plot, posd_thumb2index_list[:,0], label='$^{thumb}X_{index_{desired}}$')
                axs7[0].legend()
                axs7[1].plot(t_plot, pos_thumb2index_list[:,1], label='$^{thumb}Y_{index_{real}}$')
                axs7[1].plot(t_plot, posd_thumb2index_list[:,1], label='$^{thumb}Y_{index_{desired}}$')
                axs7[1].legend()
                axs7[2].plot(t_plot, pos_thumb2index_list[:,2], label='$^{thumb}Z_{index_{real}}$')
                axs7[2].plot(t_plot, posd_thumb2index_list[:,2], label='$^{thumb}Z_{index_{desired}}$')
                axs7[2].legend()
                axs7[3].plot(t_plot, error_axis_thumb2index[:,0], label='$\\theta_{X-axis}$')
                axs7[3].plot(t_plot, error_axis_thumb2index[:,1], label='$\\theta_{Z-axis}$')
                axs7[3].legend()
                plt.savefig(self.this_file_dir+"/datas/Index_in_Thumb.png")

                # twist of palm
                fig8, axs8 = plt.subplots(2)
                fig8.suptitle("IK output palm twist in palm")
                axs8[0].plot(t_plot, twist_palm_list.datas[0,:].T, label='$v_x \\bullet \\Delta t$')
                axs8[0].plot(t_plot, twist_palm_list.datas[1,:].T, label='$v_y \\bullet \\Delta t$')
                axs8[0].plot(t_plot, twist_palm_list.datas[2,:].T, label='$v_z \\bullet \\Delta t$')
                axs8[0].legend()
                axs8[1].plot(t_plot, twist_palm_list.datas[3,:].T, label='$\\omega_x \\bullet \\Delta t$')
                axs8[1].plot(t_plot, twist_palm_list.datas[4,:].T, label='$\\omega_y \\bullet \\Delta t$')
                axs8[1].plot(t_plot, twist_palm_list.datas[5,:].T, label='$\\omega_z \\bullet \\Delta t$')
                axs8[1].legend()
                plt.savefig(self.this_file_dir+"/datas/IK_output_V_palm_in_palm.png")

                """ Tip twist comparision: Real vs Output of IK (considering joint limits) """
                # index
                fig9, axs9 = plt.subplots(6)
                fig9.suptitle("Index real twist ($\\frac{d}{dt} T \\bullet \\Delta t$) vs output of IK (considering joint limits)")
                axs9[0].plot(t_plot, twist_index_real_ma_list.datas[0,:].T, label='$v_{xreal} \\bullet \\Delta t$')
                axs9[0].plot(t_plot, twist_index_IK_list.datas[0,:].T, label='$v_{xIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[0,:])
                IK_std = np.std(twist_index_IK_list.datas[0,:])
                # axs9[0].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[0].legend()
                axs9[1].plot(t_plot, twist_index_real_ma_list.datas[1,:].T, label='$v_{yreal} \\bullet \\Delta t$')
                axs9[1].plot(t_plot, twist_index_IK_list.datas[1,:].T, label='$v_{yIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[1,:])
                IK_std = np.std(twist_index_IK_list.datas[1,:])
                # axs9[1].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[1].legend()
                axs9[2].plot(t_plot, twist_index_real_ma_list.datas[2,:].T, label='$v_{zreal} \\bullet \\Delta t$')
                axs9[2].plot(t_plot, twist_index_IK_list.datas[2,:].T, label='$v_{zIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[2,:])
                IK_std = np.std(twist_index_IK_list.datas[2,:])
                # axs9[2].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[2].legend()
                axs9[3].plot(t_plot, twist_index_real_ma_list.datas[3,:].T, label='$\\omega_{xreal} \\bullet \\Delta t$')
                axs9[3].plot(t_plot, twist_index_IK_list.datas[3,:].T, label='$\\omega_{xIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[3,:])
                IK_std = np.std(twist_index_IK_list.datas[3,:])
                # axs9[3].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[3].legend()
                axs9[4].plot(t_plot, twist_index_real_ma_list.datas[4,:].T, label='$\\omega_{yreal} \\bullet \\Delta t$')
                axs9[4].plot(t_plot, twist_index_IK_list.datas[4,:].T, label='$\\omega_{yIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[4,:])
                IK_std = np.std(twist_index_IK_list.datas[4,:])
                # axs9[4].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[4].legend()
                axs9[5].plot(t_plot, twist_index_real_ma_list.datas[5,:].T, label='$\\omega_{zreal} \\bullet \\Delta t$')
                axs9[5].plot(t_plot, twist_index_IK_list.datas[5,:].T, label='$\\omega_{zIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_index_IK_list.datas[5,:])
                IK_std = np.std(twist_index_IK_list.datas[5,:])
                # axs9[5].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs9[5].legend()
                plt.savefig(self.this_file_dir+"/datas/dPose_index_in_ground_real_vs_IK.png")
                # thumb
                fig10, axs10 = plt.subplots(6)
                fig10.suptitle("Thumb real twist ($\\frac{d}{dt} T \\bullet \\Delta t$) vs output of IK (considering joint limits)")
                axs10[0].plot(t_plot, twist_thumb_real_ma_list.datas[0,:].T, label='$v_{xreal} \\bullet \\Delta t$')
                axs10[0].plot(t_plot, twist_thumb_IK_list.datas[0,:].T, label='$v_{xIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[0,:])
                IK_std = np.std(twist_thumb_IK_list.datas[0,:])
                # axs10[0].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[0].legend()
                axs10[1].plot(t_plot, twist_thumb_real_ma_list.datas[1,:].T, label='$v_{yreal} \\bullet \\Delta t$')
                axs10[1].plot(t_plot, twist_thumb_IK_list.datas[1,:].T, label='$v_{yIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[1,:])
                IK_std = np.std(twist_thumb_IK_list.datas[1,:])
                # axs10[1].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[1].legend()
                axs10[2].plot(t_plot, twist_thumb_real_ma_list.datas[2,:].T, label='$v_{zreal} \\bullet \\Delta t$')
                axs10[2].plot(t_plot, twist_thumb_IK_list.datas[2,:].T, label='$v_{zIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[2,:])
                IK_std = np.std(twist_thumb_IK_list.datas[2,:])
                # axs10[2].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[2].legend()
                axs10[3].plot(t_plot, twist_thumb_real_ma_list.datas[3,:].T, label='$\\omega_{xreal} \\bullet \\Delta t$')
                axs10[3].plot(t_plot, twist_thumb_IK_list.datas[3,:].T, label='$\\omega_{xIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[3,:])
                IK_std = np.std(twist_thumb_IK_list.datas[3,:])
                # axs10[3].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[3].legend()
                axs10[4].plot(t_plot, twist_thumb_real_ma_list.datas[4,:].T, label='$\\omega_{yreal} \\bullet \\Delta t$')
                axs10[4].plot(t_plot, twist_thumb_IK_list.datas[4,:].T, label='$\\omega_{yIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[4,:])
                IK_std = np.std(twist_thumb_IK_list.datas[4,:])
                # axs10[4].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[4].legend()
                axs10[5].plot(t_plot, twist_thumb_real_ma_list.datas[5,:].T, label='$\\omega_{zreal} \\bullet \\Delta t$')
                axs10[5].plot(t_plot, twist_thumb_IK_list.datas[5,:].T, label='$\\omega_{zIK} \\bullet \\Delta t$')
                IK_mean = np.mean(twist_thumb_IK_list.datas[5,:])
                IK_std = np.std(twist_thumb_IK_list.datas[5,:])
                # axs10[5].set_ylim([IK_mean-IK_std*2.0,IK_mean+IK_std*2.0])
                axs10[5].legend()
                plt.savefig(self.this_file_dir+"/datas/dPose_thumb_in_ground_real_vs_IK.png")

                plt.show()
                # self.main_process_thread.join()
                # sys.exit()
                break
            else:
                # t_plot:
                t_plot.append(t_tt - t_t0)
                # index: fx:
                fx_index_list.append(index_tf[0,0])
                fxd_index_list.append(index_tf_d[0,0])
                fx_index_ma_list.append(tf_ma_stack[0,0])
                # fy:
                fy_index_list.append(index_tf[1,0])
                fyd_index_list.append(index_tf_d[1,0])
                fy_index_ma_list.append(tf_ma_stack[1,0])
                # fz:
                fz_index_list.append(index_tf[2,0])
                fzd_index_list.append(index_tf_d[2,0])
                fz_index_ma_list.append(tf_ma_stack[2,0])
                # thumb: fx:
                fx_thumb_list.append(thumb_tf[0,0])
                fxd_thumb_list.append(thumb_tf_d[0,0])
                fx_thumb_ma_list.append(tf_ma_stack[9,0])
                # fy:
                fy_thumb_list.append(thumb_tf[1,0])
                fyd_thumb_list.append(thumb_tf_d[1,0])
                fy_thumb_ma_list.append(tf_ma_stack[10,0])
                # fz:
                fz_thumb_list.append(thumb_tf[2,0])
                fzd_thumb_list.append(thumb_tf_d[2,0])
                fz_thumb_ma_list.append(tf_ma_stack[11,0])
                # twist
                # index
                twist_index_r.add_data(twist_index)
                twist_index_tservo.add_data(dPose_index_in_index*alpha)
                # thumb
                twist_thumb_r.add_data(twist_thumb)
                twist_thumb_tservo.add_data(dPose_thumb_in_thumb*alpha)
                """ save real twist in array """
                """
                Why do I not use twist but use dPose?
                ---- Because in my cycle loop, I control the dPose every loop, and it is unsafe to devided by dt (it may equal to 0) to obtain the twist.
                """
                # index_dPose 
                real_index_dPose = np.mat(np.zeros((6,1)))
                real_index_dPose[0:3,0] = ground_T_index[0:3,3] - ground_T_index_last[0:3,3]
                wxt, wyt, wzt = angular_vel_from_R(np.matmul(ground_T_index_last[:3,:3].T, ground_T_index[:3,:3]), dt=1.0)
                real_index_dPose[3:6,0] = np.mat([[wxt], [wyt], [wzt]])
                real_index_dPose[3:6,0] = np.matmul(ground_T_index_last[:3,:3], real_index_dPose[3:6,0])
                twist_index_real_ma_list.add_data(twist_index_ma_filter.process(real_index_dPose))
                dPose_index_IK = twist_real.copy()[0:6,0]
                dPose_index_IK = np.matmul(R2RR(ground_T_index[:3,:3]), dPose_index_IK)
                # thumb_dPose 
                real_thumb_dPose = np.mat(np.zeros((6,1)))
                real_thumb_dPose[0:3,0] = ground_T_thumb[0:3,3] - ground_T_thumb_last[0:3,3]
                wxt, wyt, wzt = angular_vel_from_R(np.matmul(ground_T_thumb_last[:3,:3].T, ground_T_thumb[:3,:3]), dt=1.0)
                real_thumb_dPose[3:6,0] = np.mat([[wxt], [wyt], [wzt]])
                real_thumb_dPose[3:6,0] = np.matmul(ground_T_thumb_last[:3,:3], real_thumb_dPose[3:6,0])
                twist_thumb_real_ma_list.add_data(twist_thumb_ma_filter.process(real_thumb_dPose))
                dPose_thumb_IK = twist_real.copy()[6:12,0]
                dPose_thumb_IK = np.matmul(R2RR(ground_T_thumb[:3,:3]), dPose_thumb_IK)

                # save datas for plotting
                t_plot_for_twist.append(t_tt)
                twist_index_real_list.add_data(real_index_dPose)
                twist_index_IK_list.add_data(dPose_index_IK)
                twist_thumb_real_list.add_data(real_thumb_dPose)
                twist_thumb_IK_list.add_data(dPose_thumb_IK)
                # save last states
                ground_T_index_last = ground_T_index.copy()
                ground_T_thumb_last = ground_T_thumb.copy()
                # if t_tt - t_tt_last > 0.019:
                #     # Delta time
                #     delta_t_tt = t_tt - t_tt_last
                #     """
                #     Why do I not use twist but use dPose?
                #     ---- Because in my cycle loop, I control the dPose every loop, and it is unsafe to devided by dt (it may equal to 0) to obtain the twist.
                #     """
                #     # index_dPose 
                #     real_index_dPose = np.mat(np.zeros((6,1)))
                #     real_index_dPose[0:3,0] = ground_T_index[0:3,3] - ground_T_index_last[0:3,3]
                #     wxt, wyt, wzt = angular_vel_from_R(np.matmul(ground_T_index_last[:3,:3].T, ground_T_index[:3,:3]), dt=1.0)
                #     real_index_dPose[3:6,0] = np.mat([[wxt], [wyt], [wzt]])
                    


                #     # save datas for plotting
                #     t_plot_for_twist.append(t_tt)
                #     twist_index_real_list.add_data(real_index_dPose)
                #     twist_thumb_real_list.add_data(real_thumb_dPose)
                #     # save last states
                #     ground_T_index_last = ground_T_index.copy()
                #     ground_T_thumb_last = ground_T_thumb.copy()
                #     t_tt_last = t_tt
                # joints angles append
                joints_list = np.vstack((joints_list, jstate.T))
                joints_d_list = np.vstack((joints_d_list, target_pose.T))
                # index in thumb pose
                T_thumb2index = np.matmul(invT(T_palm2thumb), T_palm2index)
                pos_thumb2index_list = np.vstack((pos_thumb2index_list, T_thumb2index[:3,3].T))
                posd_thumb2index_list = np.vstack((posd_thumb2index_list, Td_thumb2index[:3,3].T))
                cosX = np.matmul(T_thumb2index[:3,0].T, Td_thumb2index[:3,0])[0,0]
                if cosX > 1:
                    cosX = 1.0
                elif cosX < -1:
                    cosX = -1.0
                thetaX = math.acos(cosX)
                cosZ = np.matmul(T_thumb2index[:3,2].T, Td_thumb2index[:3,2])[0,0]
                if cosZ > 1:
                    cosZ = 1.0
                elif cosZ < -1:
                    cosZ = -1.0
                thetaZ = math.acos(cosZ)
                error_axis_thumb2index = np.vstack((error_axis_thumb2index, np.mat([thetaX, thetaZ])))
                # twist_palm
                twist_palm_list.add_data(delta_u[0:6,0])
            
            t_tt = t_tt + self.delta_time
            rospy.sleep(0.02)
            self.save_time()
        rospy.signal_shutdown("Exit...")

if __name__ == '__main__':
    # Init node (In one python script, node can only be inited once.)
    rospy.init_node('sub_calc_pub_node', anonymous=True, log_level=rospy.WARN)
    my_controller = MyController()
    my_controller.main_process()
