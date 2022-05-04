import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics.rotations import quat2euler, euler2quat, mat2euler, quat_mul, quat_conjugate
import os
import random
#import torch
# from mjremote import mjremote
import time
import matplotlib.pyplot as plt
from mujoco_py import functions
from scipy.spatial.transform import Rotation as R
import math

#导入的库
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

import PyKDL as kdl

from PID import pid

robot = URDF.from_xml_file('../UR5/allegro_hand_tactile_right.urdf')
kdl_tree = kdl_tree_from_urdf_model(robot)
# kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
kdl_chain = kdl_tree.getChain("palm_link", "link_3.0_tip")

def get_geom_posquat(sim, name):
    rot = sim.data.get_geom_xmat(name)
    tform = np.eye(4)
    tform[0:3, 0:3] = rot
    tform[0:3, 3] = sim.data.get_geom_xpos(name).transpose()
    posquat = trans2posquat(tform)
    return posquat

#获得body的位置姿态四元数
def get_body_posquat(sim, name):
    pos = sim.data.get_body_xpos(name)
    quat = sim.data.get_body_xquat(name)
    posquat = np.hstack((pos, quat))
    return posquat

def get_relative_posquat_geom(sim, frame, object):
    posquat_frame = get_geom_posquat(sim, frame)
    # trans_frame = posquat2trans(sim, posquat_frame)
    trans_frame = posquat2trans(posquat_frame)
    posquat_object = get_body_posquat(sim, object)
    trans_object = posquat2trans(posquat_object)
    frameHobj = np.matmul(np.linalg.inv(trans_frame), trans_object)
    return trans2posquat(frameHobj)

#获得相对的位置姿态的四元数
def get_relative_posquat(sim, frame, object):
    posquat_frame = get_body_posquat(sim, frame)
    # trans_frame = posquat2trans(sim, posquat_frame)
    trans_frame = posquat2trans(posquat_frame)
    posquat_object = get_body_posquat(sim, object)
    trans_object = posquat2trans(posquat_object)
    frameHobj = np.matmul(np.linalg.inv(trans_frame), trans_object)
    return trans2posquat(frameHobj)

def get_prepose_posequat(wHo, oHg):
    # wHg = wHo * oHg
    wHg = np.matmul(wHo, oHg)
    posquat = trans2posquat(wHg)
    return posquat

def jac_geom(sim, geom_name):
        jacp = sim.data.get_geom_jacp(geom_name)
        jacr = sim.data.get_geom_jacr(geom_name)
        jacp = jacp.reshape(3, -1)
        jacr = jacr.reshape(3, -1)
        # print(np.vstack((jacp, jacr)))
        return np.vstack((jacp[:, :7], jacr[:, :7]))

#转换成四元数
def trans2posquat(tform):
    pos = (tform[0:3, 3]).transpose()
    # a = R.identity(3)
    # R.from_rotvec()
    # r = R.from_matrix(tform[0:3, 0:3])
    # quat = r.as_quat()
    # quat = np.hstack((quat[3], quat[0:3]))
    # print('--------------')
    # print(quat)
    quat = from_matrix(tform[0:3, 0:3])
    quat = np.hstack((quat[3], quat[0:3]))
    # print(quat)
    return np.hstack((pos, quat))


def conj_quat(quat):
    # w x y z
    quat = np.array(quat)
    res = np.array([1., 0., 0., 0.])
    functions.mju_negQuat(res, quat)
    return res

def mul_quat(quat1, quat2):
    quat1 = np.array(quat1)
    quat2 = np.array(quat2)
    res = np.array([1., 0., 0., 0.])
    functions.mju_mulQuat(res, quat1, quat2)
    return res

def quat2vel(quat, dt):
    quat = np.array(quat)
    res = np.array([0., 0., 0.])
    functions.mju_quat2Vel(res, quat, dt)
    return res

#四元数转换为旋转矩阵为位置的表示
def posquat2trans(posquat):
    # scipy quaternion is x y z w
    # quat = np.array(posquat[-4:])
    quat = np.array(posquat[3:])
    pos = np.array(posquat[:3])
    quat = np.hstack((quat[1:], quat[0]))
    r = R.from_quat(quat)
    rot = r.as_matrix()
    # rot = as_matrix(np.hstack((quat[1:], quat[0])))
    tform = np.eye(4)
    tform[0:3, 0:3] = rot
    tform[0:3, 3] = pos.transpose()
    return tform


def qpos_equal(qpos1, qpos2):
    qpos1 = np.array(qpos1)
    qpos2 = np.array(qpos2)
    delta = 0.005

    if np.shape(qpos1)[0] is not np.shape(qpos2)[0]:
        print('qpos have different sizes, exit the program')
        exit()
    else:
        N = np.shape(qpos1)[0]
        for i in range(0, N):
            if (np.linalg.norm(qpos1[i]-qpos2[i]) > delta):
                return False
        return True

def posquat_equal(posquat1, posquat2, delta=0.04):
    posquat1 = np.array(posquat1)
    posquat2 = np.array(posquat2)
    # delta = 0.04

    # if (np.shape(posquat1)[0] is not np.shape(posquat2)[0]) and (np.shape(posquat2)[0] is not 7):
    if (np.shape(posquat1)[0] is not np.shape(posquat2)[0]):
        print('ee have different sizes, exit the program')
        exit()
    else:
        for i in range(3):
            if (np.linalg.norm(posquat1[i] - posquat2[i]) > delta):
                return False

        quat1_conj = conj_quat(posquat1[-4:])
        quat_res = mul_quat(quat1_conj, posquat2[-4:])
        quat_res = quat_res - np.array([1, 0, 0, 0])
        # print('quat_res', quat_res)

        for i in range(4):
            if (np.linalg.norm(quat_res) > delta*5):
                return False
        return True


def from_matrix(matrix):

    is_single = False
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected `matrix` to have shape (3, 3) or "
                         "(N, 3, 3), got {}".format(matrix.shape))

    # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
    # set _single to True so that we can return appropriate objects in
    # the `to_...` methods
    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
        is_single = True

    num_rotations = matrix.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]

    if is_single:
        return quat[0]
        # return cls(quat[0], normalize=False, copy=False)
    else:
        # return cls(quat, normalize=False, copy=False)
        return quat

def as_matrix(quat):

    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = np.empty((1, 3, 3))

    matrix[:, 0, 0] = x2 - y2 - z2 + w2
    matrix[:, 1, 0] = 2 * (xy + zw)
    matrix[:, 2, 0] = 2 * (xz - yw)

    matrix[:, 0, 1] = 2 * (xy - zw)
    matrix[:, 1, 1] = - x2 + y2 - z2 + w2
    matrix[:, 2, 1] = 2 * (yz + xw)

    matrix[:, 0, 2] = 2 * (xz + yw)
    matrix[:, 1, 2] = 2 * (yz - xw)
    matrix[:, 2, 2] = - x2 - y2 + z2 + w2
    return matrix[0]

#author: ycj
def  quat2euler_XYZ(Rq):
    r = R.from_quat(Rq)
    euler0 = r.as_euler('xyz', degrees=True)
    return euler0

def pos_quat2pos_XYZ_RPY(pos_quat):
    #mujoco的四元数需要调换位置最前面的放在最后面
    #欧拉角转四元数的函数有问题 弧度和角度之间的转换
    PI = 3.141592654
    quat = np.hstack((pos_quat[4:], pos_quat[3]))
    # quat = np.hstack((pos_quat[1:], pos_quat[0]))
    r = R.from_quat(quat)
    euler0 = r.as_rotvec()
    pos_XYZ_angle = np.zeros(6, dtype = np.double)
    pos_XYZ_angle[0:3] = pos_quat[0:3]
    pos_XYZ_angle[-3:] = euler0
    # pos_euler_XYZ = np.hstack([pos_quat[0:3], euler0])
    return pos_XYZ_angle

# RPY角和POS之间的转换
def RPY2Rot_mat(pos_rpy):
    #mujoco的四元数需要调换位置最前面的放在最后面
    #欧拉角转四元数的函数有问题 弧度和角度之间的转换
    print("pos_rpy[-3:]:", pos_rpy[0, -3:])
    r = R.from_rotvec(pos_rpy[0, -3:])
    rot = r.as_matrix()

    return rot

def move_ik(sim, ee_tget_posquat, gripper_action=0.04, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_geom_posquat(sim, "center_hand_position")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)) :
            break
        try:
            ee_jac = jac_geom(sim, "center_hand_position")
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[3:], conj_quat(ee_curr_posquat[3:])), 1)))
            qvel = np.matmul(np.linalg.pinv(ee_jac), vel.transpose())
            sim.data.ctrl[:7] = sim.data.qpos[:7] + qvel
            sim.data.ctrl[7] = gripper_action
            sim.step()
            # viewer.render()

            ee_curr_posquat = get_geom_posquat(sim, "center_hand_position")
        except Exception as e:
            return 0
    return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))


def move_ik_finger(sim, ee_tget_posquat, gripper_action=0.04, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)) :
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp  = np.array(q_pos_test[13:17])
            print("q_pos_temp:", q_pos_temp)
            ee_jac = kdl_kin.jacobian(q_pos_temp)
            # if i == 0:
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) , quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

            qvel = np.matmul(np.linalg.pinv(ee_jac), vel.transpose())
            print("qvel:\n", qvel)
            sim.data.ctrl[6:10] = sim.data.qpos[13:17] + qvel
            # print("sim.data.ctrl[6:10]:\n", sim.data.ctrl[6:10] )
            # sim.data.ctrl[7] = gripper_action
            sim.step()
            viewer.render()

            ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        except Exception as e:
            return 0
    return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

def move_ik_kdl_finger_pinv(sim, ee_tget_posquat, gripper_action=0.04, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)) :
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp  = np.array(q_pos_test[13:17])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) , quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

            print("vel:", vel)
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)

            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input =  kdl.JntArray(num_joints)
            for i, q_i in enumerate(q_pos_temp):
                q_pos_input[i] = q_i

            succ = _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
            print("succ:", succ)
            q_out = np.array(joint_kdl_to_list(q_out))

            sim.data.ctrl[6:10] = sim.data.qpos[13:17] + q_out
            sim.step()
            viewer.render()

            ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        except Exception as e:
            return 0
    return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

def move_ik_kdl_finger_wdls_middle(sim, ee_tget_posquat, gripper_action=0.04, viewer=None):
    # ee_target is in world frame
    # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)) :
            break
        try:
            q_pos_test = sim.data.qpos
            # q_pos_temp  = np.array(q_pos_test[13:17])
            q_pos_temp  = np.array(q_pos_test[17:21])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]), quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

            #转化速度到twist形式
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_chain)
            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input =  kdl.JntArray(num_joints)
            for i, q_i in enumerate(q_pos_temp):
                q_pos_input[i] = q_i

            matrix_weight  = np.eye(4)
            matrix_weight[0][0] = 0.1
            matrix_weight[1][1] = 0.5
            matrix_weight[2][2] = 0.3
            matrix_weight[3][3] = 0.2
            _ik_v_kdl.setWeightJS(matrix_weight)

            _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
            q_out = np.array(joint_kdl_to_list(q_out))

            # 这里添加了限幅值
            # if(q_out[1]>0.05):
            #     q_out[1] = 0.05
            # if(q_out[1]<-0.05):
            #     q_out[1] = -0.05
            # print("q_out:", q_out)
            # q_out[q_out>1] = 1
            # sim.data.ctrl[6:10]   = sim.data.qpos[13:17] + q_out
            # sim.data.qpos[6:10] = sim.data.qpos[13:17] + q_out
            sim.data.ctrl[10:14] = sim.data.qpos[17:21] + q_out
            sim.step()
            viewer.render()

            # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
            ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
        except Exception as e:
            return 0
    return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

def move_ik_kdl_finger_wdls_king(sim, ee_tget_posquat, gripper_action=0.04, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)) :
            print("**********************************************************************")
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp = np.array(q_pos_test[13:17])
            # q_pos_temp  = np.array(q_pos_test[17:21])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]), quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            print("vel:", vel)
            #转化速度到twist形式
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_chain)
            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input =  kdl.JntArray(num_joints)
            for i, q_i in enumerate(q_pos_temp):
                q_pos_input[i] = q_i

            _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
            q_out = np.array(joint_kdl_to_list(q_out))
            sim.data.ctrl[6:10] = sim.data.qpos[13:17] + q_out
            # sim.data.ctrl[10:14] = sim.data.qpos[17:21] + q_out
            sim.step()
            viewer.render()

            ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        except Exception as e:
            return 0
    return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

# def move_interperate_point(sim, desire_pos_quat, curr_posquat, viewer=None):
#     # curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
#     delta_k = 5
#     # X = np.arange(0, 1, 1)
#     # Y = [curr_posquat, desire_pos_quat]
#     interpolate_point = []
#     for i in range(1,delta_k+1):
#         interpolate_point.append(curr_posquat + (desire_pos_quat-curr_posquat)/delta_k*i)
#
#     count_execute = 0;
#     for k,inter in enumerate(interpolate_point):
#         done_execute = False
#         # move_ik_kdl_finger_wdls(sim, inter)
#         print("inter:", inter)
#         while(count_execute < 200):
#             done_execute = move_ik_kdl_finger_wdls(sim, inter)
#             count_execute += 1
#             sim.step()
#             viewer.render()
#         count_execute = 0

#对这里进行修正，进行修改即可
# def force_control(sim, force_set, cur_force):
# 	kp, ki, kd = 0.0, 0.3, 0.0
# 	pid = pid.PID(kp, ki, kd)
# 	transfom_factor = 0.000003
# 	setpoint = 10
#
#     transform_base2tip3 = get_relative_posquat(sim, "base_link", "link_3.0 _tip")
#     rot = posquat2trans(transform_base2tip3)[0:3, 0:3]
#     pid_out = pid.calc(cur_force, force_set)
#     ze = np.array([0, 0, pid_out*transfom_factor]).transpose()
#     ze = np.matmul(rot, ze)
#     z = pos-ze
#     transform_base2tip3[:3] = z
#     desire_pos_quat_in_force = np.array();
#     move_ik_kdl_finger_wdls(sim, desire_pos_quat_in_force)


def execute_grasp(sim, viewer=None):
    while not (sum(sim.data.sensordata[-2:] > np.array([0, 0])) == 2):
        sim.data.ctrl[7] = sim.data.qpos[7] - 0.01
        sim.step()
        viewer.render()

def joint_kdl_to_list(q):
    if q == None:
        return None
    return [q[i] for i in range(q.rows())]
