import numpy as np
from scipy.spatial.transform import Rotation
import math
from mujoco_py import functions
import PyKDL as kdl

import robot_control
import tactile_perception as tacperception
import tactile_allegro_mujo_const
import robot_control as robcontrol
import sympy as sy
from math import sin, cos, sqrt
import time

def calculate_cov(mat):
    average = np.average(mat, axis=0)  # axis=0 按列求均值
    substract = np.empty(mat.shape)
    for i in range(mat.shape[0]):  # 遍历行
        substract[i] = mat[i] - average
    cov = np.matmul(substract.T, substract) / (mat.shape[0] - 1)
    return cov


def cross_product_matrix_from_vector3d(vector3d):
    """
    cross product matrix (skew symmetric matrix) from 3d-vector
    """
    vector3d = np.reshape(np.array(vector3d), 3)
    cpm = np.mat(np.zeros((3, 3)))
    cpm[0, 1] = -vector3d[2]
    cpm[0, 2] = vector3d[1]
    cpm[1, 0] = vector3d[2]
    cpm[1, 2] = -vector3d[0]
    cpm[2, 0] = -vector3d[1]
    cpm[2, 1] = vector3d[0]
    return cpm


def delta_pose2next_pose(d_pose, pose_0):
    """
    d_pose or delta_pose means twist*duration
    """
    d_pose = np.reshape(np.array(d_pose), 6)
    pose_t = np.zeros(7)
    pose_0 = np.reshape(np.array(pose_0), 7)
    # T_t = np.mat(np.eye(4))
    r_0 = Rotation.from_quat(pose_0[3:]).as_dcm()

    delta_angles_norm = np.linalg.norm(d_pose[3:])
    if delta_angles_norm == 0:
        pose_t[3:] = pose_0[3:]
    else:
        omega = d_pose[3:] / delta_angles_norm
        theta = delta_angles_norm
        omega_hat = cross_product_matrix_from_vector3d(omega)
        d_r = np.mat(np.eye(3)) + omega_hat * math.sin(theta) + np.matmul(omega_hat, omega_hat) * (
                    1.0 - math.cos(theta))
        quat_t = Rotation.from_dcm(np.matmul(d_r, r_0)).as_quat()
        pose_t[3:] = quat_t
    p_t = pose_0[:3] + d_pose[:3]
    pose_t[:3] = p_t
    return pose_t


def get_G_contact(pos_contact, x_state):
    #   pose_contact is the pose of active taxel which comes from the ground truth
    #   x_state is the current estimated object's pose
    S = get_S(pos_contact[:3] - x_state[:3])  # Get S(c_i - p), palm frame
    # T_contact = posquat2trans(pos_contact)
    # T_contact = get_T(sim, c_point_name)  # contact point in cup frame
    # R_contact = T_contact[:3, :3]  # Get R of contact point
    R_contact = np.mat(np.eye(3))  # Get R of contact point
    G_contact = get_G(R_contact, S)
    return G_contact


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_geom_posquat(sim, name):
    rot = sim.data.get_geom_xmat(name)
    tform = np.eye(4)
    tform[0:3, 0:3] = rot
    tform[0:3, 3] = sim.data.get_geom_xpos(name).transpose()
    posquat = trans2posquat(tform)
    return posquat


# 获得body的位置姿态四元数
def get_body_posquat(sim, name):
    pos = sim.data.get_body_xpos(name)
    quat = sim.data.get_body_xquat(name)
    posquat = np.hstack((pos, quat))
    return posquat


def get_relative_posquat_geom(sim, src, tgt):
    posquat_src = get_geom_posquat(sim, src)
    trans_src = posquat2trans(posquat_src)
    posquat_tgt = get_body_posquat(sim, tgt)
    trans_tgt = posquat2trans(posquat_tgt)
    srcHtgt = np.matmul(np.linalg.inv(trans_src), trans_tgt)
    return trans2posquat(srcHtgt)

def pose_trans_palm_to_world(sim, position, orien):
    quat = sim.data.get_body_xquat("palm_link")
    quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    palm_rot = Rotation.from_quat(quat).as_matrix()
    palm_position = sim.data.get_body_xpos("palm_link")
    p_ret = palm_position + np.ravel(np.matmul(palm_rot, position))
    o_ret = np.matmul(palm_rot, orien)
    return p_ret, o_ret
def pose_trans_world_to_palm(sim, position, orien):
    quat = sim.data.get_body_xquat("palm_link")
    quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    palm_rot = Rotation.from_quat(quat).as_matrix()
    palm_position = sim.data.get_body_xpos("palm_link")

    p_ret = np.matmul(palm_rot.transpose(), position - palm_position)
    o_ret = np.matmul(orien,palm_rot.transpose())
    return p_ret, o_ret

def vec_world_to_palm(sim, vec):
    quat = sim.data.get_body_xquat("palm_link")
    quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    palm_rot = Rotation.from_quat(quat).as_matrix()
    v_ret = np.matmul(palm_rot.transpose(), vec)
    return v_ret

def vec_palm_to_world(sim, vec):
    quat = sim.data.get_body_xquat("palm_link")
    quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    palm_rot = Rotation.from_quat(quat).as_matrix()
    v_ret = np.matmul(palm_rot, vec)
    return v_ret

# 获得相对的位置姿态的四元数
# outputs: position, rotation:w x y z
def get_relative_posquat(sim, src, tgt):
    posquat_src = get_body_posquat(sim, src)
    trans_src = posquat2trans(posquat_src)
    posquat_tgt = get_body_posquat(sim, tgt)
    trans_tgt = posquat2trans(posquat_tgt)
    srcHtgt = np.matmul(np.linalg.inv(trans_src), trans_tgt)
    return trans2posquat(srcHtgt)


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


# 转换成四元数
def trans2posquat(tform):
    pos = (tform[0:3, 3]).transpose()
    quat = from_matrix(tform[0:3, 0:3])
    quat = np.hstack((quat[3], quat[0:3]))
    # position rotation:w x y z
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


# 四元数转换为旋转矩阵为位置的表示
def posquat2trans(posquat):
    # input is w x y z
    # quat = np.array(posquat[-4:])
    quat = np.array(posquat[3:])
    pos = np.array(posquat[:3])
    quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    r = Rotation.from_quat(quat)
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
            if np.linalg.norm(qpos1[i] - qpos2[i]) > delta:
                return False
        return True


def posquat_equal(posquat1, posquat2, delta=0.04):
    posquat1 = np.array(posquat1)
    posquat2 = np.array(posquat2)
    # delta = 0.04

    # if (np.shape(posquat1)[0] is not np.shape(posquat2)[0]) and (np.shape(posquat2)[0] is not 7):
    if np.shape(posquat1)[0] is not np.shape(posquat2)[0]:
        print('ee have different sizes, exit the program')
        exit()
    else:
        for i in range(3):
            if np.linalg.norm(posquat1[i] - posquat2[i]) > delta:
                return False

        quat1_conj = conj_quat(posquat1[-4:])
        quat_res = mul_quat(quat1_conj, posquat2[-4:])
        quat_res = quat_res - np.array([1, 0, 0, 0])
        # print('quat_res', quat_res)

        for i in range(4):
            if np.linalg.norm(quat_res) > delta * 5:
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


# author: ycj
# update:lqg
def quat2euler_xyz(Rq):
    r = Rotation.from_quat(Rq)
    euler0 = r.as_euler('xyz', degrees=True)
    return euler0


# def pos_quat2pos_xyz_rpy(pos_quat):
#     # input must be w x y z
#     #mujoco的四元数需要调换位置最前面的放在最后面
#     #欧拉角转四元数的函数有问题 弧度和角度之间的转换
#     PI = 3.141592654
#     quat = np.hstack((pos_quat[4:], pos_quat[3]))
#     # quat = np.hstack((pos_quat[1:], pos_quat[0]))
#     r = R.from_quat(quat)
#     euler0 = r.as_euler('xyz')
#     pos_xyz_angle = np.zeros(6, dtype = np.double)
#     pos_xyz_angle[0:3] = pos_quat[0:3]
#     pos_xyz_angle[-3:] = euler0
#     # pos_xyz_angle[-3:] = euler0 * 57.2958
#     # pos_euler_xyz = np.hstack([pos_quat[0:3], euler0])
#     return pos_xyz_angle


def pos_quat2pos_xyz_rpy_wxyz(pos_quat):
    # input must be w x y z
    quat = np.hstack((pos_quat[4:], pos_quat[3]))
    r = Rotation.from_quat(quat)
    euler0 = r.as_euler('xyz')
    pos_xyz_angle = np.zeros(6, dtype=np.double)
    pos_xyz_angle[0:3] = pos_quat[0:3]
    pos_xyz_angle[-3:] = euler0
    return pos_xyz_angle


# function from quaterion to Euler angle
def pos_quat2pos_xyz_rpy_xyzw(pos_quat):
    # input must be w x y z
    quat = pos_quat[3:]
    r = Rotation.from_quat(quat)
    euler0 = r.as_euler('xyz')
    pos_xyz_angle = np.zeros(6, dtype=np.double)
    pos_xyz_angle[0:3] = pos_quat[0:3]
    pos_xyz_angle[-3:] = euler0
    return pos_xyz_angle

# function from quaterion to Euler angle
def pos_quat2axis_angle(pos_quat):
    # input must be w x y z
    quat = np.hstack((pos_quat[4:], pos_quat[3]))
    axis_angle = Rotation.from_quat(quat).as_rotvec()
    pos_xyz_axis_angle = np.zeros(6, dtype=np.double)
    pos_xyz_axis_angle[:3] = pos_quat[:3]
    pos_xyz_axis_angle[3:] = axis_angle
    return pos_xyz_axis_angle

def pos_euler_xyz_2_matrix(euler):
    rot = Rotation.from_euler('xyz', euler)
    return rot.as_matrix()

def rotvec_2_Matrix(rotvec):
    rot = Rotation.from_rotvec(rotvec).as_matrix()
    return rot

def move_ik(sim, ee_tget_posquat, gripper_action=0.04):
    # ee_target is in world frame
    ee_curr_posquat = get_geom_posquat(sim, "center_hand_position")
    max_step = 1000
    threshold = 0.001
    for i in range(max_step):
        if posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold):
            break
        try:
            ee_jac = jac_geom(sim, "center_hand_position")
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5,
                             quat2vel(mul_quat(ee_tget_posquat[3:], conj_quat(ee_curr_posquat[3:])), 1)))
            qvel = np.matmul(np.linalg.pinv(ee_jac), vel.transpose())
            sim.data.ctrl[:7] = sim.data.qpos[:7] + qvel
            sim.data.ctrl[7] = gripper_action
            sim.step()
            # viewer.render()

            ee_curr_posquat = get_geom_posquat(sim, "center_hand_position")
        except Exception as e:
            return 0
    return posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)


def move_ik_finger(sim, kdl_kin, ee_tget_posquat, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    max_step = 1000
    threshold = 0.001
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold):
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp = np.array(q_pos_test[13:17])
            print("q_pos_temp:", q_pos_temp)
            ee_jac = kdl_kin.jacobian(q_pos_temp)
            # if i == 0:
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, \
            # quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                             quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

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


def move_ik_kdl_finger_pinv(sim, kdl_kin, ee_tget_posquat, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp = np.array(q_pos_test[13:17])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, \
            # quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                             quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

            print("vel:", vel)
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_kin)

            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input = kdl.JntArray(num_joints)
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
    return posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)


def move_ik_kdl_finger_wdls_middle(sim, kdl_kin, ee_tget_posquat, viewer=None):
    # ee_target is in world frame
    # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
            break
        try:
            q_pos_test = sim.data.qpos
            # q_pos_temp  = np.array(q_pos_test[13:17])
            q_pos_temp = np.array(q_pos_test[17:21])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, \
            # quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                             quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

            # 转化速度到twist形式
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_kin)
            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input = kdl.JntArray(num_joints)
            for i, q_i in enumerate(q_pos_temp):
                q_pos_input[i] = q_i

            matrix_weight = np.eye(4)
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


def move_ik_kdl_finger_wdls_king(sim, kdl_kin, ee_tget_posquat, viewer=None):
    # ee_target is in world frame
    ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
    max_step = 1000
    no_step = 0
    threshold = 0.001
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    # ee_jac = jac_geom(sim, "link_3.0_tip")
    for i in range(max_step):
        if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
            print("**********************************************************************")
            break
        try:
            q_pos_test = sim.data.qpos
            q_pos_temp = np.array(q_pos_test[13:17])
            # q_pos_temp  = np.array(q_pos_test[17:21])
            # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, \
            # quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                             quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
            print("vel:", vel)
            # 转化速度到twist形式
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
            vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

            _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_kin)
            num_joints = len(kdl_kin.get_joint_names())
            q_out = kdl.JntArray(num_joints)

            q_pos_input = kdl.JntArray(num_joints)
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
    return posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)


def get_T(sim, body_name):  # 获取palm参考系下的T矩阵（trans矩阵）
    # Get T of contact point
    T_contact = posquat2trans(get_relative_posquat(sim, "palm_link", body_name))
    return T_contact

def get_relative_T(sim, body_src, body_tgt):  # 获取palm参考系下的T矩阵（trans矩阵）
    # Get T of contact point
    T_contact = posquat2trans(get_relative_posquat(sim, body_src, body_tgt))
    return T_contact


def get_T_cup(sim, body_name):  # 获取cup参考系下的T矩阵（trans矩阵）
    # Get T of contact point
    T_contact = posquat2trans(get_relative_posquat(sim, "cup", body_name))
    return T_contact


def get_S(r):
    S = np.zeros((3, 3))
    S[0, 1] = -r[2]
    S[0, 2] = r[1]
    S[1, 0] = r[2]
    S[1, 2] = -r[0]
    S[2, 0] = -r[1]
    S[2, 1] = r[0]

    # print("cHeck S:", S)
    return S


def mul_three(mat_1, mat_2, mat_3):
    mat = np.matmul(mat_1, np.matmul(mat_2, mat_3))
    return mat


def get_G(R_contact, S):
    G_upper = np.hstack((R_contact, np.zeros((3, 3))))
    G_lower = np.hstack((np.matmul(S, R_contact), R_contact))
    G_contact = np.vstack((G_upper, G_lower))  # Get grasp matrix G of contact point
    # print("  >>Check G:", G_contact)
    return G_contact


def get_P(V_o, V_c):
    print(V_o)
    print(V_c)
    P = np.eye(6)
    oc = V_c - V_o
    P[3:, :3] = get_S(oc)
    return P


def Euler2posquat(Eula):
    a1 = Eula[0, 3]
    a2 = Eula[0, 4]
    a3 = Eula[0, 5]
    # a1 = Eula[0, 3] * math.pi / 180
    # a2 = Eula[0, 4] * math.pi / 180
    # a3 = Eula[0, 5] * math.pi / 180
    w = math.cos(a1 / 2.) * math.cos(a2 / 2.) * math.cos(a3 / 2.) + math.sin(a1 / 2.) * math.sin(a2 / 2.) * math.sin(
        a3 / 2.)
    x = math.cos(a1 / 2.) * math.sin(a2 / 2.) * math.cos(a3 / 2.) + math.sin(a1 / 2.) * math.cos(a2 / 2.) * math.sin(
        a3 / 2.)
    y = math.cos(a1 / 2.) * math.cos(a2 / 2.) * math.sin(a3 / 2.) - math.sin(a1 / 2.) * math.sin(a2 / 2.) * math.cos(
        a3 / 2.)
    z = math.sin(a1 / 2.) * math.cos(a2 / 2.) * math.cos(a3 / 2.) - math.cos(a1 / 2.) * math.sin(a2 / 2.) * math.sin(
        a3 / 2.)
    posquat = np.array([Eula[0, 0], Eula[0, 1], Eula[0, 2], x, y, z, w])
    # print(posquat)
    return posquat


def Euler2posquat1D(Eula):
    a1 = Eula[3]
    a2 = Eula[4]
    a3 = Eula[5]
    # a1 = Eula[0, 3] * math.pi / 180
    # a2 = Eula[0, 4] * math.pi / 180
    # a3 = Eula[0, 5] * math.pi / 180
    w = math.cos(a1 / 2.) * math.cos(a2 / 2.) * math.cos(a3 / 2.) + math.sin(a1 / 2.) * math.sin(a2 / 2.) * math.sin(
        a3 / 2.)
    x = math.cos(a1 / 2.) * math.sin(a2 / 2.) * math.cos(a3 / 2.) + math.sin(a1 / 2.) * math.cos(a2 / 2.) * math.sin(
        a3 / 2.)
    y = math.cos(a1 / 2.) * math.cos(a2 / 2.) * math.sin(a3 / 2.) - math.sin(a1 / 2.) * math.sin(a2 / 2.) * math.cos(
        a3 / 2.)
    z = math.sin(a1 / 2.) * math.cos(a2 / 2.) * math.cos(a3 / 2.) - math.cos(a1 / 2.) * math.sin(a2 / 2.) * math.sin(
        a3 / 2.)
    posquat = np.array([Eula[0], Eula[1], Eula[2], x, y, z, w])
    # print(posquat)
    return posquat

def posquat2posrotvec(posquat):
    posrotvec = np.zeros(6)
    posrotvec[:3] = posquat[:3]
    _quat = np.hstack((posquat[4:], posquat[3]))
    posrotvec[3:] = Rotation.from_quat(_quat).as_rotvec()
    return posrotvec

def posquat2posrotvec_hacking(posquat):
    posrotvec = np.zeros(6)
    posrotvec[:3] = posquat[:3]
    _quat = np.hstack((posquat[4:], posquat[3]))
    rm = Rotation.from_quat(_quat).as_matrix()
    # compute rot_vec
    theta_tmp = math.acos((np.trace(rm) - 1.0) / 2.0)
    omega_tmp = np.array([0., 0., 0.])
    omega_tmp[0] = (rm[2][1] - rm[1][2]) / (2 * math.sin(theta_tmp))
    omega_tmp[1] = (rm[0][2] - rm[2][0]) / (2 * math.sin(theta_tmp))
    omega_tmp[2] = (rm[1][0] - rm[0][1]) / (2 * math.sin(theta_tmp))
    if omega_tmp[1] < 0.0:
        omega = (-1.0) * omega_tmp
        theta = 2.0 * math.pi - theta_tmp
    else:
        omega = omega_tmp
        theta = theta_tmp

    posrotvec[3:] = theta * omega
    return posrotvec

def getrotvecfromposquat(posquat):
    posrotvec = np.zeros(6)
    posrotvec[:3] = posquat[:3]
    _quat = np.hstack((posquat[4:], posquat[3]))
    return Rotation.from_quat(_quat).as_matrix()
def posquat2pos_p_o(posquat):
    # pos_p = np.zeros(3)
    # pos_o = np.zeros(3, 3)
    pos_p = posquat[:3]
    _quat = np.hstack((posquat[4:], posquat[3]))
    pos_o = Rotation.from_quat(_quat).as_matrix()
    return pos_p, pos_o

def rm2rotvec(rm):
    rmrotvec = Rotation.from_matrix(rm).as_rotvec()
    return rmrotvec



def joint_kdl_to_list(q):
    if q == None:
        return None
    return [q[i] for i in range(q.rows())]

def contact_compute(sim, model, fingername, tacperception, x_state, cur_angles, robctrl):
    # body jocobian matrix and velocity
    G_contact = np.zeros([6, 6])
    if not tactile_allegro_mujo_const.betterJ_FLAG:
        if fingername == 'ff':
            cur_jnt = cur_angles[0:4]
            # Get Jacobi J
            Jac = robctrl.kdl_kin_ff.jacobian(cur_jnt)
            if tacperception.is_ff_contact == True:
                pos_contact = tacperception.get_contact_taxel_position(sim, model, fingername, "palm_link")
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])
        if fingername == 'mf':
            cur_jnt = cur_angles[4:8]
            # Get Jacobi J
            Jac = robctrl.kdl_kin_mf.jacobian(cur_jnt)
            if tacperception.is_mf_contact == True:
                pos_contact = tacperception.get_contact_taxel_position(sim, \
                                                                       model, fingername, "palm_link")
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])
        if fingername == 'rf':
            cur_jnt = cur_angles[8:12]
            # Get Jacobi J
            Jac = robctrl.kdl_kin_rf.jacobian(cur_jnt)
            if tacperception.is_rf_contact == True:
                pos_contact = tacperception.get_contact_taxel_position(sim, \
                                                                       model, fingername, "palm_link")
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])
        if fingername == 'th':
            cur_jnt = cur_angles[12:16]
            # Get Jacobi J
            Jac = robctrl.kdl_kin_th.jacobian(cur_jnt)
            if tacperception.is_th_contact == True:
                pos_contact = tacperception.get_contact_taxel_position(sim, \
                                                                       model, fingername, "palm_link")
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])
    else:
        if fingername == 'ff':
            taxel_name = tacperception.get_contact_taxel_name(sim, model, 'ff')
            cur_jnt = cur_angles[0:4]
            Jac = robctrl.robjac_offset(sim, fingername, cur_jnt, taxel_name)
            if tacperception.is_ff_contact == True:
                pos_contact = tacperception.get_contact_taxel_position_from_name(sim, \
                                                model, fingername, "palm_link", taxel_name)
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])

        if fingername == 'mf':
            taxel_name = tacperception.get_contact_taxel_name(sim, model, 'mf')
            cur_jnt = cur_angles[4:8]
            Jac = robctrl.robjac_offset(sim, fingername, cur_jnt, taxel_name)
            if tacperception.is_mf_contact == True:
                pos_contact = tacperception.get_contact_taxel_position_from_name(sim, \
                                                                       model, fingername, "palm_link",taxel_name)
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])


        if fingername == 'rf':
            taxel_name = tacperception.get_contact_taxel_name(sim, model, 'rf')
            cur_jnt = cur_angles[8:12]
            Jac = robctrl.robjac_offset(sim, fingername, cur_jnt, taxel_name)
            if tacperception.is_rf_contact == True:
                pos_contact = tacperception.get_contact_taxel_position_from_name(sim, \
                                                                       model, fingername, "palm_link",taxel_name)
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])
        if fingername == 'th':
            taxel_name = tacperception.get_contact_taxel_name(sim, model, 'th')
            cur_jnt = cur_angles[12:16]
            Jac = robctrl.robjac_offset(sim, fingername, cur_jnt, taxel_name)
            if tacperception.is_th_contact == True:
                pos_contact = tacperception.get_contact_taxel_position_from_name(sim, \
                                                                       model, fingername, "palm_link",taxel_name)
                #    the G_contact is partial grasping matrix because the noised object pose, refer to:
                #    Eq.2.14, Chapter 2 Robot Grasping Foundations/ B. León et al., From Robot to Human Grasping Simulation,
                #    Cognitive Systems Monographs 19, DOI: 10.1007/978-3-319-01833-1_2
                G_contact = get_G_contact(pos_contact, x_state)
                # G_contact = G_contact.transpose()
            else:
                G_contact = np.zeros([6, 6])

    return G_contact, Jac
    # return G_contact, Jac, u_t0, cur_jnt


def H_calculator(W1, W2, W3, pos_CO_x, pos_CO_y, pos_CO_z):
    """
    Calculate a 3*6 H matrix
    """
    # print("  W1, W2, W3, pos_CO_x, pos_CO_y, pos_CO_z", W1, W2, W3, pos_CO_x, pos_CO_y, pos_CO_z)
    H = sy.Matrix([
        [1, 0, 0, pos_CO_x * (
                W1 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                math.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                2 * W1 * W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + pos_CO_y * (
                     -2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W3 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     -2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
         pos_CO_x * (W2 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W2 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_y * (
                     -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_z * (
                     -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 ** 2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), pos_CO_x * (
                     W3 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W2 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_y * (
                     -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W3 ** 2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     -2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2))],
        [0, 1, 0, pos_CO_x * (-2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W3 * cos(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                          1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_z * (
                     -W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), pos_CO_x * (
                     -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_y * (
                     W2 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + pos_CO_z * (
                     -W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
         pos_CO_x * (-2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 ** 2 * cos(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W3 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_z * (
                     -W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 * W3 ** 2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2))],
        [0, 0, 1, pos_CO_x * (-2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W2 * cos(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                          1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W1 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W2 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (W1 ** 2 + W2 ** 2 + W3 ** 2))),
         pos_CO_x * (-2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W2 ** 2 * cos(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 ** 2 * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W2 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2))),
         pos_CO_x * (-2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_y * (
                     W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 * W3 ** 2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W3 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 2 * W3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2))]
    ])
    return H

def H_calculator_pn(W1, W2, W3, normal_CO_x, normal_CO_y, normal_CO_z):
    H = sy.Matrix([[0, 0, 0, normal_CO_x * (
                W1 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                            1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                            2 * W1 * W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + normal_CO_y * (
                             -2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W3 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_z * (
                             -2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W2 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
                 normal_CO_x * (W2 * (
                             -W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                            1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                            2 * W2 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                                W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (
                                                        W1 ** 2 + W2 ** 2 + W3 ** 2))) + normal_CO_y * (
                             -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (
                                         3 / 2)) + normal_CO_z * (
                             -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 ** 2 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), normal_CO_x * (W3 * (
                    -W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W2 ** 2 * W3 / (
                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                                               W1 ** 2 + W2 ** 2 + W3 ** 2))) + normal_CO_y * (
                             -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W3 ** 2 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W3 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_z * (
                             -2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2))],
                [0, 0, 0, normal_CO_x * (-2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W3 * cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                                     1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                                     W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_y * (W1 * (
                            -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 3 / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2))) + normal_CO_z * (
                             -W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), normal_CO_x * (
                             -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (
                                         3 / 2)) + normal_CO_y * (W2 * (
                            -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 2 * W2 / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + normal_CO_z * (
                             -W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
                 normal_CO_x * (-2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                             W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 ** 2 * cos(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 * sin(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_y * (W3 * (
                             -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                     sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 2 * W3 / (
                             W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (
                                                                        W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                                                        W1 ** 2 + W2 ** 2 + W3 ** 2))) + normal_CO_z * (
                             -W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 * W3 ** 2 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2))],
                [0, 0, 0, normal_CO_x * (-2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W2 * cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                                     1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                                     W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_y * (
                             W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_z * (W1 * (
                            -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 3 / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W2 ** 2 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2))), normal_CO_x * (
                             -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W2 ** 2 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_y * (
                             W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_z * (W2 * (
                            -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 2 * W2 / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 3 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2))), normal_CO_x * (
                             -2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                         1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                         W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(
                         sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (
                                         3 / 2)) + normal_CO_y * (W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                                                              W1 ** 2 + W2 ** 2 + W3 ** 2) ** (
                                                                              3 / 2) - 2 * W2 * W3 ** 2 * (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)) + normal_CO_z * (W3 * (
                            -W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (1 - cos(
                    sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (2 * W1 ** 2 * W3 / (
                            W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 2 * W3 / (
                                                                       W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2))]])

    # print(H)
    return H

def F_calculator_4Ginv(ju):
    """
    Use the F_mat which is already complete.
    The GT_inv is spliced from GT1_inv, GT2_inv, GT3_inv and GT4_inv.
    """
    R1_CH = sy.eye(3)
    R2_CH = sy.eye(3)
    R3_CH = sy.eye(3)
    R4_CH = sy.eye(3)

    r100, r101, r102, r110, r111, r112, r120, r121, r122 = R1_CH[0, 0], R1_CH[0, 1], R1_CH[0, 2], R1_CH[1, 0], R1_CH[1,
                                                                                                                     1], \
                                                           R1_CH[1, 2], R1_CH[2, 0], R1_CH[2, 1], R1_CH[2, 2]
    r200, r201, r202, r210, r211, r212, r220, r221, r222 = R2_CH[0, 0], R2_CH[0, 1], R2_CH[0, 2], R2_CH[1, 0], R2_CH[1,
                                                                                                                     1], \
                                                           R2_CH[1, 2], R2_CH[2, 0], R2_CH[2, 1], R2_CH[2, 2]
    r300, r301, r302, r310, r311, r312, r320, r321, r322 = R3_CH[0, 0], R3_CH[0, 1], R3_CH[0, 2], R3_CH[1, 0], R3_CH[1,
                                                                                                                     1], \
                                                           R3_CH[1, 2], R3_CH[2, 0], R3_CH[2, 1], R3_CH[2, 2]
    r400, r401, r402, r410, r411, r412, r420, r421, r422 = R4_CH[0, 0], R4_CH[0, 1], R4_CH[0, 2], R4_CH[1, 0], R4_CH[1,
                                                                                                                     1], \
                                                           R4_CH[1, 2], R4_CH[2, 0], R4_CH[2, 1], R4_CH[2, 2]
    ju = np.ravel(ju.T)
    # print("??", ju)
    ju10, ju11, ju12, ju13, ju14, ju15 = ju[0], ju[1], ju[2], ju[3], ju[4], ju[5]
    ju20, ju21, ju22, ju23, ju24, ju25 = ju[6], ju[7], ju[8], ju[9], ju[10], ju[11]
    ju30, ju31, ju32, ju33, ju34, ju35 = ju[12], ju[13], ju[14], ju[15], ju[16], ju[17]
    ju40, ju41, ju42, ju43, ju44, ju45 = ju[18], ju[19], ju[20], ju[21], ju[22], ju[23]

    F = sy.Matrix([
        [1, ju13 * (-r101 * r112 + r102 * r111) / (
                r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 r100 * r112 - r102 * r110) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 -r100 * r111 + r101 * r110) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 -r201 * r212 + r202 * r211) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 r200 * r212 - r202 * r210) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 -r200 * r211 + r201 * r210) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 -r301 * r312 + r302 * r311) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 r300 * r312 - r302 * r310) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 -r300 * r311 + r301 * r310) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 -r401 * r412 + r402 * r411) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 r400 * r412 - r402 * r410) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 -r400 * r411 + r401 * r410) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         ju13 * (-r101 * r122 + r102 * r121) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 r100 * r122 - r102 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 -r100 * r121 + r101 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 -r201 * r222 + r202 * r221) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 r200 * r222 - r202 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 -r200 * r221 + r201 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 -r301 * r322 + r302 * r321) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 r300 * r322 - r302 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 -r300 * r321 + r301 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 -r401 * r422 + r402 * r421) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 r400 * r422 - r402 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 -r400 * r421 + r401 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         0, 0, 0],
        [ju13 * (r101 * r112 - r102 * r111) / (
                r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 -r100 * r112 + r102 * r110) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 r100 * r111 - r101 * r110) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 r201 * r212 - r202 * r211) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 -r200 * r212 + r202 * r210) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 r200 * r211 - r201 * r210) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 r301 * r312 - r302 * r311) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 -r300 * r312 + r302 * r310) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 r300 * r311 - r301 * r310) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 r401 * r412 - r402 * r411) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 -r400 * r412 + r402 * r410) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 r400 * r411 - r401 * r410) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         1, ju13 * (-r111 * r122 + r112 * r121) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 r110 * r122 - r112 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 -r110 * r121 + r111 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 -r211 * r222 + r212 * r221) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 r210 * r222 - r212 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 -r210 * r221 + r211 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 -r311 * r322 + r312 * r321) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 r310 * r322 - r312 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 -r310 * r321 + r311 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 -r411 * r422 + r412 * r421) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 r410 * r422 - r412 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 -r410 * r421 + r411 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         0, 0, 0],
        [ju13 * (r101 * r122 - r102 * r121) / (
                r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 -r100 * r122 + r102 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 r100 * r121 - r101 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 r201 * r222 - r202 * r221) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 -r200 * r222 + r202 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 r200 * r221 - r201 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 r301 * r322 - r302 * r321) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 -r300 * r322 + r302 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 r300 * r321 - r301 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 r401 * r422 - r402 * r421) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 -r400 * r422 + r402 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 r400 * r421 - r401 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         ju13 * (r111 * r122 - r112 * r121) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju14 * (
                 -r110 * r122 + r112 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju15 * (
                 r110 * r121 - r111 * r120) / (
                 r100 * r111 * r122 - r100 * r112 * r121 - r101 * r110 * r122 + r101 * r112 * r120 + r102 * r110 * r121 - r102 * r111 * r120) + ju23 * (
                 r211 * r222 - r212 * r221) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju24 * (
                 -r210 * r222 + r212 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju25 * (
                 r210 * r221 - r211 * r220) / (
                 r200 * r211 * r222 - r200 * r212 * r221 - r201 * r210 * r222 + r201 * r212 * r220 + r202 * r210 * r221 - r202 * r211 * r220) + ju33 * (
                 r311 * r322 - r312 * r321) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju34 * (
                 -r310 * r322 + r312 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju35 * (
                 r310 * r321 - r311 * r320) / (
                 r300 * r311 * r322 - r300 * r312 * r321 - r301 * r310 * r322 + r301 * r312 * r320 + r302 * r310 * r321 - r302 * r311 * r320) + ju43 * (
                 r411 * r422 - r412 * r421) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju44 * (
                 -r410 * r422 + r412 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420) + ju45 * (
                 r410 * r421 - r411 * r420) / (
                 r400 * r411 * r422 - r400 * r412 * r421 - r401 * r410 * r422 + r401 * r412 * r420 + r402 * r410 * r421 - r402 * r411 * r420),
         1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    print("  F_val(4 small_GT_inv)", F.shape, "\n",
          F[0, :], "\n",
          F[1, :], "\n",
          F[2, :], "\n",
          F[3, :], "\n",
          F[4, :], "\n",
          F[5, :])
    return F

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm
def vec2rot(vec):
    rot = np.zeros([3, 3])
    rot_x = np.zeros(3)
    vec_normalize = normalize(vec)
    x = vec_normalize[0]
    y = vec_normalize[1]
    rot_x[0] = y / math.sqrt(x**2 + y**2)
    rot_x[1] = - x / math.sqrt(x**2 + y**2)
    rot_y = np.cross(vec_normalize, rot_x)
    rot[:3, 0] = rot_x
    rot[:3, 1] = rot_y
    rot[:3, 2] = vec_normalize.T
    # print("    vec==rot:", rot)
    return rot


def rolling_average(angle):

    return avr