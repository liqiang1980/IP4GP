#!/usr/bin/env python

# system
from __future__ import print_function
import copy
import pathlib
from scipy.optimize._lsq.least_squares import prepare_bounds
from time import sleep, time
import threading
import sys
import xml.etree.ElementTree as ET
import csv
import re
# from pympler import muppy, summary, asizeof

# plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# math
import numpy as np
import math
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.optimize import minimize, least_squares, curve_fit
from scipy import odr
import pandas as pd

# ROS
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

# self-defined msg
# from allegro_ur_ctrl_cmd.msg import allegro_hand_joints_cmd, ur_ee_cmd
from allegro_tactile_sensor.msg import tactile_msgs

def KDLframe2T(KDLframe):
    """ Input a kdl.Frame()\n
    Output a T(ransformation) matrix T
    """
    T_from_PyKDL_frame = np.mat(np.eye(4))
    KDLframe_copy = copy.deepcopy(KDLframe)
    T_from_PyKDL_frame[0,0] = KDLframe_copy.M[0,0]
    T_from_PyKDL_frame[0,1] = KDLframe_copy.M[0,1]
    T_from_PyKDL_frame[0,2] = KDLframe_copy.M[0,2]
    T_from_PyKDL_frame[1,0] = KDLframe_copy.M[1,0]
    T_from_PyKDL_frame[1,1] = KDLframe_copy.M[1,1]
    T_from_PyKDL_frame[1,2] = KDLframe_copy.M[1,2]
    T_from_PyKDL_frame[2,0] = KDLframe_copy.M[2,0]
    T_from_PyKDL_frame[2,1] = KDLframe_copy.M[2,1]
    T_from_PyKDL_frame[2,2] = KDLframe_copy.M[2,2]
    T_from_PyKDL_frame[0,3] = KDLframe_copy.p[0]
    T_from_PyKDL_frame[1,3] = KDLframe_copy.p[1]
    T_from_PyKDL_frame[2,3] = KDLframe_copy.p[2]
    return T_from_PyKDL_frame

def pose_quat2new_ros_pose(pose_quat):
    """
    Input an array or list of 7-D (px, py, pz, qx, qy, qz, qw), mapping to a geometry_msgs.msg.Pose
    Point: position and Quaternion: orientation are all geometry_msgs.msg.Pose gets, it has no header
    """
    pose_quat = np.reshape(np.array(pose_quat), 7)
    ros_pose = Pose()
    ros_pose.position.x = pose_quat[0]
    ros_pose.position.y = pose_quat[1]
    ros_pose.position.z = pose_quat[2]
    ros_pose.orientation.x = pose_quat[3]
    ros_pose.orientation.y = pose_quat[4]
    ros_pose.orientation.z = pose_quat[5]
    ros_pose.orientation.w = pose_quat[6]
    return ros_pose

def ros_pose2new_pose_quat(ros_pose):
    """
    Input an array or list of 7-D (px, py, pz, qx, qy, qz, qw), mapping to a geometry_msgs.msg.Pose
    Point: position and Quaternion: orientation are all geometry_msgs.msg.Pose gets, it has no header
    """
    pose_quat = np.zeros(7)
    pose_quat[0] = ros_pose.position.x
    pose_quat[1] = ros_pose.position.y
    pose_quat[2] = ros_pose.position.z
    pose_quat[3] = ros_pose.orientation.x
    pose_quat[4] = ros_pose.orientation.y
    pose_quat[5] = ros_pose.orientation.z
    pose_quat[6] = ros_pose.orientation.w
    return pose_quat

def wxyz2xyzw(wxyz0):
    """ Change quaternion order: from wxyz to xyzw """
    wxyz = np.array(copy.deepcopy(wxyz0)).flatten().tolist()
    if np.size(wxyz) == 4:
        xyzw = np.zeros(4)
        xyzw[:3] = wxyz[1:]
        w = wxyz[0]
        xyzw[3] = w
        return xyzw
    else:
        print("wxyz size != 4")

def xyzw2wxyz(xyzw0):
    """ Change quaternion order: from xyzw to wxyz """
    xyzw = np.array(copy.deepcopy(xyzw0)).flatten().tolist()
    if np.size(xyzw) == 4:
        wxyz = np.zeros(4)
        wxyz[1:] = xyzw[:3]
        w = xyzw[3]
        wxyz[0] = w
        return wxyz
    else:
        print("xyzw size != 4")

def mujoco_posquat2T(xyzqwqxqyqz0):
    xyzqwqxqyqz = copy.deepcopy(xyzqwqxqyqz0)
    T_1 = np.mat(np.eye(4))
    T_1[:3, 3] = np.mat(xyzqwqxqyqz[:3].copy()).T
    T_1[:3, :3] = Rotation.from_quat(wxyz2xyzw(xyzqwqxqyqz[3:])).as_matrix()
    return T_1

def T2mujoco_posquat(T_10):
    T_1 = T_10.copy()
    pos = np.array(T_1[:3, 3].copy().T).flatten().tolist()
    quat = Rotation.from_matrix(T_1[:3, :3].copy()).as_quat().flatten().tolist()
    quat = xyzw2wxyz(quat)
    pos.append(quat)
    return pos

def find_vec_from_text(text1):
    """ Extract all numbers as a vector from a text """
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    vector_list = rx.findall(text1)
    vector_list = np.array(vector_list)
    vector_list = vector_list.astype(np.float)
    return(vector_list)


def average_quaternions_normal(quats):
    """ quats is a Nx4 numpy matrix and contains the quaternions to average in the rows. """
    A = np.zeros((4,4))
    quats_num = np.size(quats, 0)
    for i in range(quats_num):
        this_quat = quats[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(this_quat,this_quat) + A
    # scale
    A = (1.0/quats_num)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    quat_bar = np.real(eigenVectors[:,0])
    return quat_bar

# def quats_NSlerp(curr_quat, des_quat, duration):
#     # Nslerp or Slerp for quaternions, depending on the angle between quaternions (using MuJoCo's form: qw,qx,qy,qz)
#     # First turn into angle-axis presentation
#     theta = np.arccos(curr_quat[0]*des_quat[0])

def joint_kdl_to_list(q):
    if q == None:
        return None
    return [q[i] for i in range(q.rows())]

def kdl_finger_twist2qvel(kdl_kin, kdl_chain, qpos_current, twist):
    """
    Derived from Chaojie Yan's func.py
    For 4-joints finger Cartesian motion planning.
    Unit: length: meter, angle: rad, time: second.
    """
    _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)
    vel_twist = kdl.Twist()
    vel_twist.vel = kdl.Vector(twist[0], twist[1], twist[2])
    vel_twist.rot = kdl.Vector(twist[3], twist[4], twist[5])
    
    num_joints = len(kdl_kin.get_joint_names())
    qvel_out = kdl.JntArray(num_joints)
    q_pos_input = kdl.JntArray(num_joints)
    for i, q_i in enumerate(qpos_current):
        q_pos_input[i] = q_i

    _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)
    matrix_weight  = np.eye(4)
    # matrix_weight[0][0] = 0.1
    # matrix_weight[1][1] = 0.5
    # matrix_weight[2][2] = 0.3
    # matrix_weight[3][3] = 0.2
    # _ik_v_kdl.setWeightJS(matrix_weight)
    _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, qvel_out)
    qvel_out = np.array(joint_kdl_to_list(qvel_out))
    return qvel_out

def kdl_calc_fk(fk, q, pos):
    fk_flag = fk.JntToCart(q, pos)

def kdl_finger_fk(kdl_kin, kdl_chain, qpos_current, twist):
    """
    forward kinematics
    """
    _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)
    vel_twist = kdl.Twist()
    vel_twist.vel = kdl.Vector(twist[0], twist[1], twist[2])
    vel_twist.rot = kdl.Vector(twist[3], twist[4], twist[5])
    
    num_joints = len(kdl_kin.get_joint_names())
    qvel_out = kdl.JntArray(num_joints)
    q_pos_input = kdl.JntArray(num_joints)
    for i, q_i in enumerate(qpos_current):
        q_pos_input[i] = q_i

    _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)
    matrix_weight  = np.eye(4)
    # matrix_weight[0][0] = 0.1
    # matrix_weight[1][1] = 0.5
    # matrix_weight[2][2] = 0.3
    # matrix_weight[3][3] = 0.2
    _ik_v_kdl.setWeightJS(matrix_weight)
    _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, qvel_out)
    qvel_out = np.array(joint_kdl_to_list(qvel_out))
    return qvel_out

def kdl_finger_cartpln_jac():
    pass

def pose2T(pose):
    """
    Transform pose (7,) (x,y,z,qx,qy,qz,qw) to T (transformation matrix)
    """
    pose = np.reshape((np.array(copy.deepcopy(pose))).copy(),7)
    T = np.mat(np.eye(4))
    T[:3,:3] = Rotation.from_quat(pose[3:]).as_dcm()
    T[:3,3] = np.reshape(pose[:3].copy(), (3,1))
    return T

def T2pose(T00):
    """
    Transform T (transformation matrix) to pose (7,) (x,y,z,qx,qy,qz,qw)
    """
    T0 = T00.copy()
    pose = np.zeros(7)
    pose[:3] = np.reshape(np.array(T0[:3,3]),3)
    pose[3:] = Rotation.from_dcm(T0[:3,:3]).as_quat()
    return pose

def T2pos_euler(T00):
    """ From transformation matrix T to position + euler (extrin) """
    poseul = np.zeros(6)
    T0 = T00.copy()
    poseul[:3] = T0[:3, 3].T
    poseul[3:] = Rotation.from_dcm(T0[:3,:3]).as_euler('zyx')
    return poseul.round(10)

def R2RR(R00):
    """ stack 3X3 rotation matrix to construct a 6X6 RR matrix """
    R0 = R00.copy()
    RR0 = np.mat(np.eye(6))
    RR0[:3,:3] = R0
    RR0[3:,3:] = R0
    return RR0

def angular_vel_from_quats(q_new, q, dt):
    '''
    :param q_new: new quaternion [x,y,z,w]
    :param q: last quaternion [x,y,z,w]
    :param dt: duration time
    :return: angular velocity [wx, wy, wz]
    R(t+T) = exp(integral_from_t_to_t+T(w))*R(t)
    '''
    q_new = np.reshape(np.array(q_new), 4)
    # print("q_new:",q_new)
    q = np.reshape(np.array(q), 4)
    # print("q:",q)
    R_new = Rotation.from_quat(q_new).as_dcm()
    R_0 = Rotation.from_quat(q).as_dcm()
    R_delta = np.matmul(R_new, R_0.T)
    temp = (np.trace(R_delta)-1.0)*0.5
    if temp > 1.0:
        temp = 1.0
    elif temp < -1.0:
        temp = -1.0
    theta = math.acos(temp)
    vel = theta/dt
    s_theta = math.sin(theta)
    if s_theta == 0.0:
        wx = 0.0
        wy = 0.0
        wz = 0.0
    else:
        wx = vel*(R_delta[2,1] - R_delta[1,2])/(2.0*s_theta)
        wy = vel*(R_delta[0,2] - R_delta[2,0])/(2.0*s_theta)
        wz = vel*(R_delta[1,0] - R_delta[0,1])/(2.0*s_theta)

    # wx = angle_axis[0]
    # wy = angle_axis[1]
    # wz = angle_axis[2]
    return wx, wy, wz

def angular_vel_from_R(R_delta, dt):
    '''
    :param q_new: new quaternion [x,y,z,w]
    :param q: last quaternion [x,y,z,w]
    :param dt: time step
    :return: angular velocity [wx, wy, wz]
    R(t+T) = exp(integral_from_t_to_t+T(w))*R(t)
    '''
    R_delta = R_delta
    tmp = (np.trace(R_delta)-1.0)*0.5
    if tmp > 1.0:
        tmp = 1.0
    elif tmp < -1.0:
        tmp = -1.0
    theta = math.acos(tmp)
    vel = theta/dt
    s_theta = math.sin(theta)
    if s_theta == 0.0:
        wx = 0.0
        wy = 0.0
        wz = 0.0
    else:
        wx = vel*(R_delta[2,1] - R_delta[1,2])/(2.0*s_theta)
        wy = vel*(R_delta[0,2] - R_delta[2,0])/(2.0*s_theta)
        wz = vel*(R_delta[1,0] - R_delta[0,1])/(2.0*s_theta)
    # wx = angle_axis[0]
    # wy = angle_axis[1]
    # wz = angle_axis[2]
    # print(np.isnan(wx), np.isnan(wy), np.isnan(wz))
    return wx, wy, wz

def cross_product_matrix_from_vector3d(vector3d00):
    """
    cross product matrix (skew symmetric matrix) from 3d-vector
    """
    vector3d = np.reshape(np.array(copy.deepcopy(vector3d00)), 3)
    cpm = np.mat(np.zeros((3,3)))
    cpm[0,1] = -vector3d[2]
    cpm[0,2] = vector3d[1]
    cpm[1,0] = vector3d[2]
    cpm[1,2] = -vector3d[0]
    cpm[2,0] = -vector3d[1]
    cpm[2,1] = vector3d[0]
    return cpm

def quats2delta_angle(quat_0, quat_t):
    """ From quaternions to angle (rads) """
    quat_0 = np.reshape(np.array(copy.deepcopy(quat_0)), 4)
    quat_t = np.reshape(np.array(copy.deepcopy(quat_t)), 4)
    return 2.0*math.acos(np.dot(quat_0, quat_t))

def R2delta_angle(R_0, R_t):
    """ From rotation matrices to angle (rads) """
    d_R = np.matmul(R_0, R_t.T)
    return math.acos((np.trace(d_R)-1.0)/2.0)

def quats2delta_angle_degrees(quat_0, quat_t):
    """ From quaternions to angle (degrees) """
    quat_0 = np.reshape(np.array(copy.deepcopy(quat_0)), 4)
    quat_t = np.reshape(np.array(copy.deepcopy(quat_t)), 4)
    return 360.0*math.acos(np.dot(quat_0, quat_t))/math.pi

def T2AdT(T):
    """
    Function that transform T matrix to adjoint map matrix
    """
    return pR2AdT(T[:3,3], T[:3,:3])

def pR2AdT(p_vector, R_matrix):
    """
    Function that transform position (vector) + Rotation (matrix) to adjoint map matrix
    Ad(T(s in b))*twist(in s) = twist(in b), twist := [vel, angular vel]^T
    """
    AdT = np.mat(np.zeros((6,6)))
    AdT[:3,:3] = R_matrix
    AdT[3:6,3:6] = R_matrix
    # AdT[0:3,3:6] = np.matmul(cross_product_matrix_from_vector3d(p_vector), R_matrix)
    return AdT

def delta_pose2next_T(d_pose, T_00):
    """
    d_pose or delta_pose means twist*duration
    """
    T_0 = T_00.copy()
    d_pose = np.reshape(np.array(copy.deepcopy(d_pose)),6)
    T_t = np.mat(np.eye(4))

    delta_angles_norm = np.linalg.norm(d_pose[3:])
    if delta_angles_norm == 0:
        T_t[:3,:3] = T_0[:3,:3]
    else:
        omega = d_pose[3:]/delta_angles_norm
        theta = delta_angles_norm
        omega_hat = cross_product_matrix_from_vector3d(omega)
        d_R = np.mat(np.eye(3)) + omega_hat*math.sin(theta) + np.matmul(omega_hat, omega_hat)*(1.0 - math.cos(theta))
        T_t[:3,:3] = np.matmul(d_R, T_0[:3,:3])
    T_t[:3,3] = T_0[:3,3] + d_pose[0:3].T
    return T_t




def delta_pose2next_pose(d_pose, pose_0):
    """
    d_pose or delta_pose means twist*duration
    """
    d_pose = np.reshape(np.array(copy.deepcopy(d_pose)),6)
    pose_t = np.zeros(7)
    pose_00 = np.reshape(np.array(copy.deepcopy(pose_0)), 7)
    # T_t = np.mat(np.eye(4))
    R_0 = Rotation.from_quat(pose_00[3:]).as_dcm()

    delta_angles_norm = np.linalg.norm(d_pose[3:])
    if delta_angles_norm == 0:
        pose_t[3:] = pose_00[3:]
    else:
        omega = d_pose[3:]/delta_angles_norm
        theta = delta_angles_norm
        omega_hat = cross_product_matrix_from_vector3d(omega)
        d_R = np.mat(np.eye(3)) + omega_hat*math.sin(theta) + np.matmul(omega_hat, omega_hat)*(1.0 - math.cos(theta))
        quat_t = Rotation.from_dcm(np.matmul(d_R, R_0)).as_quat()
        pose_t[3:] = quat_t
    p_t = pose_00[:3] + d_pose[:3]
    pose_t[:3] = p_t
    return pose_t

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Calculate rotation matrix from 2 given vectors
    """
    vec1 = np.reshape(np.array(copy.deepcopy(vec1)), 3)
    vec2 = np.reshape(np.array(copy.deepcopy(vec2)), 3)
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    v = np.cross(a, b)
    if np.linalg.norm(v) == 0:
        if np.dot(a, b) > 0:
        # print("v = ", v)
            return np.eye(3)
        else:
            nx = [1., 0., 0.]
            if np.cross(a, nx):
                c = np.cross(a, nx)
            else:
                c = np.cross(a, [0., 1., 0.])
            R_rot = Rotation.from_rotvec(180.0 * c, degrees=True).as_dcm()
            return R_rot
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    # print("rotation_matrix:\n", rotation_matrix)
    return rotation_matrix

def weighted_average_quaternions(quats, weights):
    """
    Quaternions are stacked in row, e.t.,\n
    [[qx_1, qy_1, qz_1, qw_1],
     [qx_2, qy_2, qz_2, qw_2],
     ...
     [qx_n, qy_n, qz_n, qw_n]]
    ()
    """
    quats00 = copy.deepcopy(quats)
    M = np.mat(np.zeros((4,4)))
    n = np.size(quats00, 0) # number of quaternions
    wSum = 0
    for i in range(n):
        q = quats00[i,:]
        w_i = weights[i]
        M = M + w_i*np.matmul(q.T, q)
        wSum = wSum + w_i
    # Scaling
    M = M/wSum
    eig_values, eig_vectors = np.linalg.eigh(M)
    return np.reshape(np.array(eig_vectors[:,3]), 4)
    # eigh guarantees you that the eigenvalues are sorted and uses a faster algorithm that takes advantage of the fact that the matrix is symmetric
    # The eigenvalues in ascending order, each repeated according to its multiplicity.

def pinv_SVD(matrix_0):
    """
    Generalized inverse with SVD, return the generalized inverse
    """
    # D = np.mat(np.zeros_like(matrix_0))
    D_T = np.mat(np.zeros_like(matrix_0.T))
    # I_n = np.mat(np.eye(np.size(matrix_0, 1)))
    U, diag_D, V_T = np.linalg.svd(matrix_0, full_matrices=True, compute_uv=True)
    for i_diag in range(np.size(diag_D)):
        if diag_D[i_diag] != 0:
            D_T[i_diag,i_diag] = 1.0/diag_D[i_diag]
        else:
            break
    matrix_pinv = np.matmul(np.matmul(V_T.T, D_T), U.T)
    return matrix_pinv

def pinv_TSVD(matrix_0, thres_ratio=1e3):
    """
    Generalized inverse with Truncated-SVD, return the generalized inverse and the Truncated-SVD-matrix_0
    """
    thres_den = 1.0/thres_ratio
    D = np.mat(np.zeros_like(matrix_0))
    D_T = np.mat(np.zeros_like(matrix_0.T))
    # I_n = np.mat(np.eye(np.size(matrix_0, 1)))
    U, diag_D, V_T = np.linalg.svd(matrix_0, full_matrices=True, compute_uv=True)
    for i_diag in range(np.size(diag_D)):
        if diag_D[i_diag] > thres_den*diag_D[0]:
            D_T[i_diag,i_diag] = 1.0/diag_D[i_diag]
            D[i_diag,i_diag] = diag_D[i_diag]
        else:
            break
    matrix_TSVD = np.matmul(np.matmul(U, D), V_T)
    matrix_pinv = np.matmul(np.matmul(V_T.T, D_T), U.T)
    return matrix_pinv, matrix_TSVD

def invT(TransMat0):
    """ Inverse the transformation matrix in SE(3) """
    TransMat = TransMat0.copy()
    TransMat_inv = np.mat(np.eye(4))
    TransMat_inv[:3,:3] = TransMat[:3,:3].T
    TransMat_inv[:3,3] = np.matmul(TransMat_inv[:3,:3], -TransMat[:3,3])
    return TransMat_inv
    

























class StreamingMovingAverage_list:
    """Moving average filter, inited with window_size, buffer will increase until it reached the window_size"""
    def __init__(self, window_size):
        self.window_size1 = float(window_size)
        self.values = []
        self.sum = 0.0

    def process(self, value):
        self.values.append(value)
        self.sum = self.sum + value
        current_size = len(self.values)
        if current_size > self.window_size1:
            self.sum = self.sum - self.values.pop(0)
            return self.sum / self.window_size1
        else:
            return self.sum / float(current_size)

class StreamingMovingAverage_array:
    """Moving average filter, inited with dimension and window_size, buffer will shift"""
    def __init__(self, window_size, dimension):
        self.values = np.mat((np.zeros((dimension, window_size))))
        self.sum = np.mat((np.zeros((dimension, 1))))
        self.if_full = False
        self.window_size_int = window_size
        self.window_size_float = float(window_size)
        self.current_size = 0

    def process(self, value):
        """ 
        Input: value: column vector\n 
        Output: Averages of each dimensions: column vector
        """
        
        self.sum = self.sum + value
        if self.if_full:
            self.sum = self.sum - self.values[:,self.window_size_int-1]
            self.values[:,1:] = self.values[:,:self.window_size_int-1]
            self.values[:,0] = value
            return self.sum/self.window_size_float
        else:
            self.values[:,1:] = self.values[:,:self.window_size_int-1]
            self.values[:,0] = value
            self.current_size = self.current_size + 1
            if self.current_size > self.window_size_int:
                self.if_full = True
                self.current_size = self.current_size - 1
                return self.sum/self.window_size_float
            else:
                return self.sum/float(self.current_size)

class array_append:
    def __init__(self, dimension):
        self.datas = np.mat(np.zeros((dimension, 0)))
    def add_data(self, data):
        """ Input data is treated as column vector """
        self.datas = np.hstack((self.datas, data))

