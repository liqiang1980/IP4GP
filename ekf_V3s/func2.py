import math
from scipy.spatial.transform import Rotation
import func as f
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF

def Camera_set(viewer, model):
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = model.stat.extent * 1.0
    viewer.cam.lookat[2] += .1
    viewer.cam.lookat[0] += .5
    viewer.cam.lookat[1] += .5
    viewer.cam.elevation = -0
    viewer.cam.azimuth = 0


def robot_init(sim):
    sim.data.ctrl[0] = 0.8
    sim.data.ctrl[1] = -0.78
    sim.data.ctrl[2] = 1.13
    sim.data.ctrl[3] = -1.
    sim.data.ctrl[4] = 0
    sim.data.ctrl[5] = -0.3


def calculate_cov(mat):
    average = np.average(mat, axis=0)  # axis=0 按列求均值
    substract = np.empty(mat.shape)
    for i in range(mat.shape[0]):  # 遍历行
        substract[i] = mat[i] - average
    cov = np.matmul(substract.T, substract) / (mat.shape[0] - 1)
    return cov


def get_normal_from_formula(coe, point):
    nor = np.array([0., 0., -1.])
    nor[0] = 2. * coe[0] * point[0] + coe[1] * point[1] + coe[3]
    nor[1] = coe[1] * point[0] + 2. * coe[2] * point[1] + coe[4]
    return nor


def surface_cup(cur_x, cur_y, cur_z):
    s = [129.4, 1.984, 139., -0.666, 0.32, -0.207]  # 曲面方程的6个系数
    return get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s


def surface_bottle(cur_x, cur_y, cur_z):
    # -330.2*x^2-380.2*xy-259.4*y^2+17.66*x-11.82*y-0.183
    s = [-330.2, -380.2, -259.4, 17.66, -11.82, -0.183]  # 曲面方程的6个系数
    return get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s


def index_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[0:72]) > 0.0).any():  # 食指
        sim.data.ctrl[7] = sim.data.ctrl[7] + input1
        sim.data.ctrl[8] = sim.data.ctrl[8] + input1
        sim.data.ctrl[9] = sim.data.ctrl[9] + input1
    else:
        sim.data.ctrl[7] = sim.data.ctrl[7] + input2
        sim.data.ctrl[8] = sim.data.ctrl[8] + input2
        sim.data.ctrl[9] = sim.data.ctrl[9] + input2

def middle_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[144:216]) > 0.0).any():  # 中指
        sim.data.ctrl[11] = sim.data.ctrl[11] + input1
        sim.data.ctrl[12] = sim.data.ctrl[12] + input1
        sim.data.ctrl[13] = sim.data.ctrl[13] + input1
    else:
        sim.data.ctrl[11] = sim.data.ctrl[11] + input2
        sim.data.ctrl[12] = sim.data.ctrl[12] + input2
        sim.data.ctrl[13] = sim.data.ctrl[13] + input2

def little_thumb(sim, input1, input2):
    if not (np.array(sim.data.sensordata[288:360]) > 0.0).any():  # 小拇指
        sim.data.ctrl[15] = sim.data.ctrl[15] + input1
        sim.data.ctrl[16] = sim.data.ctrl[16] + input1
        sim.data.ctrl[17] = sim.data.ctrl[17] + input1
    else:
        sim.data.ctrl[15] = sim.data.ctrl[15] + input2
        sim.data.ctrl[16] = sim.data.ctrl[16] + input2
        sim.data.ctrl[17] = sim.data.ctrl[17] + input2

def pre_thumb(sim, viewer):
    for _ in range(20):
        sim.data.ctrl[18] = sim.data.ctrl[18] + 0.05
        sim.step()
    viewer.render()

def thumb(sim, input1, input2):
    if not (np.array(sim.data.sensordata[432:504]) > 0.0).any():  # da拇指
        sim.data.ctrl[19] = sim.data.ctrl[19] + input1
        sim.data.ctrl[20] = sim.data.ctrl[20] + input1
        sim.data.ctrl[21] = sim.data.ctrl[21] + input1*5
    else:
        sim.data.ctrl[19] = sim.data.ctrl[19] + input2
        sim.data.ctrl[20] = sim.data.ctrl[20] + input2
        sim.data.ctrl[21] = sim.data.ctrl[21] + input2*5

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
    d_pose = np.reshape(np.array(d_pose),6)
    pose_t = np.zeros(7)
    pose_0 = np.reshape(np.array(pose_0), 7)
    # T_t = np.mat(np.eye(4))
    R_0 = Rotation.from_quat(pose_0[3:]).as_dcm()

    delta_angles_norm = np.linalg.norm(d_pose[3:])
    if delta_angles_norm == 0:
        pose_t[3:] = pose_0[3:]
    else:
        omega = d_pose[3:]/delta_angles_norm
        theta = delta_angles_norm
        omega_hat = cross_product_matrix_from_vector3d(omega)
        d_R = np.mat(np.eye(3)) + omega_hat*math.sin(theta) + np.matmul(omega_hat, omega_hat)*(1.0 - math.cos(theta))
        quat_t = Rotation.from_dcm(np.matmul(d_R, R_0)).as_quat()
        pose_t[3:] = quat_t
    p_t = pose_0[:3] + d_pose[:3]
    pose_t[:3] = p_t
    return pose_t

def get_normal(sim, model, c_points, trans_cup2palm):
    pos_contact_avg_cupX = np.empty([1, 0])
    pos_contact_avg_cupY = np.empty([1, 0])
    pos_contact_avg_cupZ = np.empty([1, 0])
    for i in c_points:
        c_point_name_zz = model._sensor_id2name[i]
        #todo why here the cartesian mean is used, not in the contact position computation (pos_contact0)
        posquat_contact_cup_zz = f.get_relative_posquat(sim, "cup", c_point_name_zz)
        pos_contact_avg_cupX = np.append(pos_contact_avg_cupX, posquat_contact_cup_zz[0])
        pos_contact_avg_cupY = np.append(pos_contact_avg_cupY, posquat_contact_cup_zz[1])
        pos_contact_avg_cupZ = np.append(pos_contact_avg_cupZ, posquat_contact_cup_zz[2])

    # get mean position:
    pos_contact_avg_cup = np.array([pos_contact_avg_cupX.mean(), pos_contact_avg_cupY.mean(), pos_contact_avg_cupZ.mean()])
    #############################  Get normal of contact point on the cup   ########################################
    nor_contact_in_cup, res = surface_cup(pos_contact_avg_cup[0], pos_contact_avg_cup[1],
                                             pos_contact_avg_cup[2])
    nor_in_p = np.matmul(trans_cup2palm, np.hstack((nor_contact_in_cup, 1)).T).T[:3]
    # Normalization:
    den = (nor_in_p[0] ** 2 + nor_in_p[1] ** 2 + nor_in_p[2] ** 2) ** 0.5
    nn_nor_contact_in_p = np.array(
        [nor_in_p[0] / den, nor_in_p[1] / den, nor_in_p[2] / den])
    print("normal after normalization:", nn_nor_contact_in_p)

    return nn_nor_contact_in_p, res


def get_G(sim, c_point_name, pos_contact, y_t_update):
    # print("chuck:", pos_contact, y_t_update)
    S = f.get_S(pos_contact - y_t_update[:3])  # Get S(c_i - p), palm frame
    #todo looks like this is related to the palm frame, right?
    #todo, is there an contact frame related to a taxel?
    #todo could you please comment the formula you are using for f.get_T();
    T_contact = f.get_T(sim, c_point_name)  # contact point in cup frame
    R_contact = T_contact[:3, :3]  # Get R of contact point
    #todo name is the same with this function,
    #todo could you please comment the formula you are using for f.get_G();
    G_contact = f.get_G(R_contact, S)
    return G_contact

def get_c_point_name(model, c_points):  # get the median of all contact_nums, and translate to contact_name
    if len(c_points) % 2 == 0:  # even number of contact points
        c_points = np.hstack((-1, c_points))  # change to odd
        #todo why the median of taxels are the contact point?
        c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
        print(np.median(c_points))
    else:
        c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
        print(np.median(c_points))

    return c_point_name

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range