import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import func as f

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

import time
from mujoco_py import const, cymj, functions

import scipy.interpolate as spi
from scipy.spatial.transform import Rotation as R
import fcl_python
import fcl

from sympy import *
import math
import tactile_allegro_mujo_const

#指定一个表格，来对应相应的关节
#关节对应顺序表
#6   食指根部旋转    #7食指   从下往上第二关节旋转  #8   食指从下往上第三关节旋转 #9   食指最后一个关节   link_3.0_tip    食指指尖
#10 中指根部旋转   #11中指从下往上第二关节旋转   #12 中指从下往上第三关节旋转#13 中指最后一个关节   link_7.0_tip    中指指尖
#14 无名根部旋转   #15无名从下往上第二关节旋转   #16 无名从下往上第三关节旋转#17 无名最后一个关节   link_11.0_tip  无名指尖
#18拇指大旋转        #19 拇指根部旋转       #20 拇指大弯曲关节       #21 拇指小弯曲关节   link_15.0_tip  拇指指尖
#手指上的触觉点说明：  每个手指尖有72个触觉点， 6*12分布， 一共有4组触觉点
# 4*72个点

def Camera_set():
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = model.stat.extent * 1.0
    viewer.cam.lookat[2] += .1
    viewer.cam.lookat[0] += .5
    viewer.cam.lookat[1] += .5
    viewer.cam.elevation = -0
    viewer.cam.azimuth = 0

def robot_init():
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3

def hand_init():
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = -0.00502071
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = 0.2
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = 0.68513787
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = 0.85640426

#雅克比测试
def jac_geom(sim, geom_name):
        jacp = sim.data.get_geom_jacp(geom_name)
        jacr = sim.data.get_geom_jacr(geom_name)
        jacp = jacp.reshape(3, -1)
        jacr = jacr.reshape(3, -1)
        print("jacp:", jacp)
        print("jacr:", jacr)
        return np.vstack((jacp[:, :7], jacr[:, :7]))

#手掌坐标系渲染
def show_coordinate(sim, body_name):
    palm_link_pose = f.get_body_posquat(sim, body_name)
    rot_palm_link = f.as_matrix(np.hstack((palm_link_pose[4:], palm_link_pose[3])))
    #这里的最初的轴是它的y轴
    palm_link_rot_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    palm_link_rot_y = np.array([[1, 0, 0], [0, 0, 1], [0,1,0]]) #绕X轴的旋转 -90
    palm_link_rot_z = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])#绕y轴的旋转 90

    # view中的mat是绕全局坐标系下的旋转，而不是局部坐标系下的旋转
    #红色是x轴， 绿色是y轴，蓝色是z轴
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_x, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_y, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_z, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

def control_input():
    if not (np.array(sensor_data) > 0.0).any():
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + 0.005

#仅测试了link_3.0_tip 需要使用请对相应接口进行修改即可
#pykdl库的安装
def KDL_forward():
    # q_pos = sim.data.qpos
    # link_3_q = np.array(q_pos[13:17])
    q_pos_1 = np.array([sim.data.qpos[126], sim.data.qpos[127], sim.data.qpos[164], sim.data.qpos[201]])
    # link_3_q = q_pos_1
    pose_calc = kdl_kin_1.forward(q_pos_1)
    relative_pose = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    relative_pose_trans = f.posquat2trans(relative_pose)

#触觉点可视化 arrow
def touch_visual(a, save_point_output):
    global max_size
    truth = f.get_relative_posquat(sim, "base_link", "cup")

    save_point_use = np.array([[0, 0, 0, 0, 0, 0, 0]])
    # save_point_use = np.append(save_point_use, np.array([truth]), axis=0)
    print (a)
    for i in a:
        for k,l in enumerate(i):
            #s_name is the taxel's name which can be found in UR5_tactile_allegro_hand.xml
            s_name = model._sensor_id2name[i[k]]
            # print (s_name)
            sensor_pose = f.get_body_posquat(sim, s_name)
            relative_pose = f.get_relative_posquat(sim, "base_link", s_name)
            save_point_use = np.append(save_point_use, np.array([relative_pose]), axis=0)
            rot_sensor = f.as_matrix(np.hstack((sensor_pose[4:], sensor_pose[3])))
            #用于控制方向，触觉传感器的方向问题
            # test_rot = np.array([[1, 0, 0],[0,0,1],[0,1,0]])
            test_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

            # viewer.add_marker(pos=sensor_pose[:3], mat =test_rot, type=const.GEOM_ARROW, label="contact", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    if save_point_use.shape[0] > max_size:
        save_point_output = save_point_use
        np.save("output.npy", save_point_output)
        # where_a = np.where(np.array(sensor_data) > 0.0)
        # print(where_a)

    max_size = max(save_point_use.shape[0], max_size)
    viewer.render()

def rot_mat_2_vec(T_location, originVector):
    # originVector是原始向量,T_location转换后向量
    # T_location = Vector((1.0 , 0 ,.0))

    T_location_norm = T_location

    # T_location_norm.normalize()
    T_location_norm = T_location/np.linalg.norm(T_location)

    # print(T_location_norm)
    temp_ = np.dot()
    sita = math.acos(T_location_norm@originVector)
    n_vector1 = T_location_norm.cross(originVector)

    n_vector = n_vector1/np.linalg.norm(n_vector1)
    # n_vector.normalize()

    n_vector_invert = np.array([
        [0,-n_vector[2],n_vector[1]],
        [n_vector[2],0,-n_vector[0]],
        [-n_vector[1],n_vector[0],0]
    ])

    # print(sita)
    # print(n_vector_invert)

    I = np.array(
        [[1,  0, 0],
        [0,  1, 0],
        [0,  0, 1]]
    )
    R_w2c = I + math.sin(sita)*n_vector_invert + n_vector_invert@(n_vector_invert)*(1-math.cos(sita))
    return R_w2c
# 逆运动学解算
def move_ik_control():
    desire_pos_quat = np.array([ 0.08916218, 0.04625495, 0.05806168, 0.69976209, -0.02073365, 0.7129112, -0.04075237])
    curr_posquat = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    flag_control = f.move_ik_kdl_finger_wdls_king(sim, desire_pos_quat)
    return flag_control

#接触过程中的全部触觉点的记录和渲染
def save_point_visual(pos):
    delta = 0.01
    count = 0
    for i in save_point:
        if np.linalg.norm(pos[:3] - i[:3]) > delta:
            count += 1
    if count == len(save_point):
        save_point.append(pos)
    print(len(save_point))
    test_rot_point = np.array([[1, 0, 0],[0,0,1],[0,1,0]])
    for i in save_point:
        viewer.add_marker(pos=i[:3], mat=test_rot_point, type=const.GEOM_SPHERE, label="", size=np.array([0.005, 0.005, 0.005]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

#关节空间的位置闭合抓取
def control_grasp_joint_control():
    if not (np.array(sensor_data[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any():
        flag = False
        if(flag == False):
            flag = move_ik_control()
        KDL_forward()
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + 0.005
    else:
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2]
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3]
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4]

    if not (np.array(sensor_data[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any():
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] + 0.006
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] + 0.003
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] + 0.003
    else:
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2]
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3]
        sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4]

    if not (np.array(sensor_data[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any():
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] + 0.006
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] + 0.003
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] + 0.003
    else:
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2]
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3]
        sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4]

    if not (np.array(sensor_data[sensor_data[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]]) > 0.0).any():
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] + 0.015   #0.01可以
        # sim.data.ctrl[19] = sim.data.ctrl[19] + 0.002
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + 0.002
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + 0.002
    else:
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1]
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2]
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3]
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4]

def matrix_6_6_get(param1, param2, param3, param4, param5, param6):
    matrix_6_6 = np.array([[param1, 0, 0, 0, 0, 0],
                            [0, param2, 0, 0, 0, 0],
                            [0, 0, param3, 0, 0, 0],
                            [0, 0, 0, param4, 0, 0],
                            [0, 0, 0, 0, param5, 0],
                            [0, 0, 0, 0, 0, param6]])
    return matrix_6_6

def get_grasp_matrix_trans(contact_point_name, x_t_1):
    posquat_base_point = f.get_relative_posquat(sim, "palm_link", contact_point_name)
    R_i = f.posquat2trans(posquat_base_point)[0:3, 0:3]
    contact_point_pos = np.array([posquat_base_point[0:3]])
    object_position = x_t_1[0, 0:3]#物体的目标姿态只能通过估计得到的位置和姿态获得
    delta_p_c = np.array(contact_point_pos - object_position)
    cross_product = cross_product_func(delta_p_c)
    #grasp matrix 获取
    zero = np.zeros((3, 3))
    grasp_matrix_up = np.hstack([R_i, zero])
    cross_product_R_i = np.matmul(R_i, cross_product)
    grasp_matrix_down = np.hstack([cross_product_R_i, R_i])
    grasp_matrix = np.vstack([grasp_matrix_up, grasp_matrix_down])
    grasp_matrix_trans = grasp_matrix.T
    return grasp_matrix_trans

def get_Ht(contact_point_name, grasp_matrix_trans):
    pos_relative_to_cup = f.get_relative_posquat(sim, "cup", contact_point_name)
    de_n_c = derivate(pos_relative_to_cup) #临时变量 [3*3]

    cup_pose_global = f.get_relative_posquat(sim, "palm_link", "cup")
    cup_trans_global = f.posquat2trans(cup_pose_global)
    R_cup_global = cup_trans_global[0:3, 0:3]
    R_cup_global_inv = np.linalg.inv(R_cup_global)

    # 坐标变换
    de_n_c = np.matmul(R_cup_global_inv, de_n_c)
    Ht = np.matmul(de_n_c, grasp_matrix_trans[0:3, :]) #3*6
    return Ht

def Pt_update(Kt, Ht, Pt):
    Kt_mul_Ht = np.matmul(Kt, Ht)
    I_mat = np.eye(6)
    err_I_Kt_Ht = I_mat - Kt_mul_Ht
    pt_update = np.matmul(err_I_Kt_Ht, Pt)
    return pt_update

def get_Kt(Ht, Pt, Rt):
    Kt_behind = mul_three(Ht, Pt, Ht.transpose())  #3*6  * 6*6 * 6*3
    Kt_behind_Rt = Kt_behind + Rt   #3*3
    Kt_behind_Rt = Kt_behind_Rt.astype(np.float)
    Kt_behind_Rt_inv = np.linalg.inv(Kt_behind_Rt) #3*3
    Kt = mul_three(Pt, Ht.transpose(), Kt_behind_Rt_inv) #6*6  * 6*3 * 3*3 = 6*3
    return Kt

def get_Zt(contact_point_name):
    normal_init = np.array([[1, 0, 0]]).transpose()
    #这里的R_i可能有问题，主要是相对的坐标关系，是不是在同一个坐标系下
    posquat_base_point = f.get_relative_posquat(sim, "palm_link", contact_point_name)
    R_i = f.posquat2trans(posquat_base_point)[0:3, 0:3]
    Zt = np.matmul(R_i, normal_init)
    return Zt

def get_err_ht_Zt(contact_point_name, normal_fcl_func):
    Zt = get_Zt(contact_point_name)
    ht = normal_fcl_func

    if np.sign(Zt[2][0]) == np.sign(ht[2][0]):
        ht = -ht

    err_Zt_ht = Zt + ht #主要是决定方向
    return err_Zt_ht

# 手指关节的问题，确定手指关节
def two_finger_predict(contact_point_name_1, contact_point_name_2, normal_fcl_1, normal_fcl_2):
    global x_t, x_t_1
    global q_pos_pre_1, q_pos_pre_2
    global Pt_1, count_time
    global q_pos_1, q_pos_2

    q_pos_1 = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
    delta_Ut_1 = q_pos_1 - q_pos_pre_1
    q_pos_pre_1 = q_pos_1

    q_pos_2 = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1], \
                        sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2], \
                        sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3], \
                        sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])
    delta_Ut_2 = q_pos_2 - q_pos_pre_2
    q_pos_pre_2 = q_pos_2

    u_t = np.hstack((delta_Ut_1, delta_Ut_2)).transpose()

    print("U_t:", u_t)
    print("count_time:", count_time)
    count_time += 1

    grasp_matrix_trans_1 = get_grasp_matrix_trans(contact_point_name_1, x_t_1)
    grasp_matrix_trans_pinv_1 = np.linalg.pinv(grasp_matrix_trans_1)
    grasp_matrix_trans_2 = get_grasp_matrix_trans(contact_point_name_2, x_t_1)
    grasp_matrix_trans_pinv_2 = np.linalg.pinv(grasp_matrix_trans_2)
    # grasp_matrix_trans_pinv = np.hstack((grasp_matrix_trans_pinv_1, grasp_matrix_trans_pinv_2)).transpose()
    grasp_matrix_trans_pinv = np.hstack((grasp_matrix_trans_pinv_1, grasp_matrix_trans_pinv_2))
    # print("grasp_matrix_trans_pinv:", grasp_matrix_trans_pinv)
    # # 雅克比的伪逆 每个关节角的变化也需要记录
    J_finger = np.zeros([12, 8])
    J_1 = kdl_kin_1.jacobian(q_pos_1)
    J_2 = kdl_kin_2.jacobian(q_pos_2)
    J_finger[:6, :4] = J_1
    J_finger[6:, 4:] = J_2
    # print("J_finger:", J_finger)

    # G*J*u_t*delta_t
    prediction_part = mul_three(grasp_matrix_trans_pinv, J_finger, u_t)
    # print("prediction_part:", prediction_part)
    # prediction = prediction_part[0, 0:]
    prediction = prediction_part
    # 这里可能有问题
    x_t = x_t_1 - prediction
    # x_t = x_t_1 + prediction

    #大拇指控制的問題 首先是调整大拇指 kp值的
    #Pt的更新
    de_F = matrix_6_6_get(1, 1, 1, 1, 1, 1)
    # two
    # V_t = matrix_6_6_get(0.00001,-0.000001,0.00001,0.0005,0.005,-0.0005)
    # V_t = matrix_6_6_get(-0.000001,-0.000001,0.000001,0.0005,0.005,-0.0005)
    # V_t = matrix_6_6_get(0.0,-0.0,0.0,0.0,0.0,-0.0)
    V_t = matrix_6_6_get(0.00001, -0.000001, -0.000001, 0.0025, -0.000002, -0.0005)
    temp_Pt = mul_three(de_F, Pt_1, de_F.transpose())
    Pt = temp_Pt + V_t

    # 1.无接触时的切换问题  2.初始化的问题 3. normal_fcl获取的问题
    Ht_1 = get_Ht(contact_point_name_1, grasp_matrix_trans_1[0:3, :]) #3*6
    Ht_2 = get_Ht(contact_point_name_2, grasp_matrix_trans_2[0:3, :]) #3*6
    Ht_use = np.vstack((Ht_1, Ht_2))

    Rt = np.eye(6)
    Kt = get_Kt(Ht_use, Pt, Rt)  #6*3 -> 6*6

    err_Zt_ht_1 = get_err_ht_Zt(contact_point_name_1, normal_fcl_1)
    err_Zt_ht_2 = get_err_ht_Zt(contact_point_name_2, normal_fcl_2)
    err_Zt_ht_use = np.vstack((err_Zt_ht_1, err_Zt_ht_2))

    Kt_err_Zt_ht = np.matmul(Kt, err_Zt_ht_use).transpose()

    # x_t_update是局部变量
    x_t_update = x_t + Kt_err_Zt_ht #1*6
    x_t_update = x_t_update.astype(np.float)
    x_t_1 = x_t_update

    Pt_1 = Pt_update(Kt, Ht_use, Pt)
    return x_t_update

# 接口就这几个变量
# 需要同时修改两个vt的变量
def one_finger_predict(contact_point_name_1, normal_fcl_1):
    global x_t, x_t_1
    global q_pos_pre_1
    global Pt_1, count_time

    q_pos_1 = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3], \
                        sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
    delta_Ut_1 = q_pos_1 - q_pos_pre_1
    q_pos_pre_1 = q_pos_1

    u_t = delta_Ut_1.transpose()

    print("count_time:", count_time)
    count_time += 1

    grasp_matrix_trans_1 = get_grasp_matrix_trans(contact_point_name_1, x_t_1)
    grasp_matrix_trans_pinv_1 = np.linalg.pinv(grasp_matrix_trans_1)

    # # 雅克比的伪逆 每个关节角的变化也需要记录
    J_finger = kdl_kin_1.jacobian(q_pos_1)

    # G*J*u_t*delta_t
    prediction_part = mul_three(grasp_matrix_trans_pinv_1, J_finger, u_t)
    prediction = prediction_part[0, 0:]
    # 这里可能有问题
    x_t = x_t_1 - prediction
    # x_t = x_t_1 + prediction

    #大拇指控制的問題 首先是调整大拇指 kp值的
    #Pt的更新
    de_F = matrix_6_6_get(1, 1, 1, 1, 1, 1)
    # one
    V_t = matrix_6_6_get(0.00001, -0.000001, -0.000001, 0.0005, 0.005, -0.0005)
    # V_t = matrix_6_6_get(0.0,-0.0,0.0,0.0,0.0,-0.0)
    temp_Pt = mul_three(de_F, Pt_1, de_F.transpose())
    Pt = temp_Pt + V_t

    # 1.无接触时的切换问题  2.初始化的问题 3. normal_fcl获取的问题
    Ht_use = get_Ht(contact_point_name_1, grasp_matrix_trans_1[0:3, :]) #3*6

    Rt = np.eye(3)
    Kt = get_Kt(Ht_use, Pt, Rt)  #6*3 -> 6*6

    err_Zt_ht_use = get_err_ht_Zt(contact_point_name_1, normal_fcl_1)

    Kt_err_Zt_ht = np.matmul(Kt, err_Zt_ht_use).transpose()

    # x_t_update是局部变量
    x_t_update = x_t + Kt_err_Zt_ht #1*6
    x_t_update = x_t_update.astype(np.float)
    x_t_1 = x_t_update

    Pt_1 = Pt_update(Kt, Ht_use, Pt)
    return x_t_update

#前向prediction model。 将手指运动通过grasp matrix 转换到物体姿态的变化
#prediction 更新公式 ：x_t = x_t_1 + G*J*u_t*delta_t
def prediction_model(sim):
    #状态变量定义
    global first_contact, second_contact, x_t, x_t_1, contact_point_name_1, contact_point_name_2, count_time
    global save_pose_x_t_xyz, save_pose_GD_xyz, save_pose_x_t_rpy, save_pose_GD_rpy, save_count_time
    global q_pos_pre_1, normal_record_fcl, normal_record_finger
    global q_pos_pre_2
    #update_step:
    global Pt_1, normal_fcl_1, normal_fcl_2 #注意是全局变量

#分为两种情况；
# 1 首先只有一个手指接触的情况
# 2 两个手指接触的情况
    if (np.array(sensor_data[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
                tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any():
        if first_contact == True:
            #x_t_1的更新
            x_t_1 = f.get_relative_posquat(sim, "palm_link", "cup")
            delta_x = np.array([0.002, 0.002, 0.002, 0, 0, 0])
            x_t_1 = np.array([f.pos_quat2pos_XYZ_RPY(x_t_1)]) + delta_x
            x_t = x_t_1
            print("x_t_1:", x_t_1)

            #这里需要修改matrix_6_6_get
            Pt_1 = matrix_6_6_get(0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001)

            q_pos_1 = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1], \
                                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2], \
                                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3], \
                                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
            q_pos_pre_1 = q_pos_1

            # 记录第一个接触点：
            a = np.where(np.array(sensor_data[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
                tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0)
            print("id of actived taxel in ff", np.array(sensor_data[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
                tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]))
            for i in a:
                s_name = model._sensor_id2name[i[0]]
                contact_point_name_1 = s_name
            first_contact = False
        x_t_update = np.zeros((1, 6))

        if (np.array(sensor_data[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
                tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any():
            if second_contact == True:
                q_pos_2 = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1], \
                                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2], \
                                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3], \
                                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])
                q_pos_pre_2 = q_pos_2

                print("second contact********************")
                # 记录第一个接触点：
                a = np.where(np.array(sensor_data[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
                                                  tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0)
                for i in a:
                    s_name = model._sensor_id2name[i[0]]
                    contact_point_name_2 = s_name
                second_contact = False

            x_t_update = two_finger_predict(contact_point_name_1, contact_point_name_2, normal_fcl_1, normal_fcl_2)
            print("x_t_update222222222222222222222222222222********************")
        else:
            x_t_update = one_finger_predict(contact_point_name_1, normal_fcl_1)
            print("x_t_update1111111111111111111111111111111********************")

        print("x_t_update:", x_t_update)
        #cup真值的获取
        ground_truth = f.get_relative_posquat(sim, "palm_link", "cup")
        ground_truth = np.array([f.pos_quat2pos_XYZ_RPY(ground_truth)])

        print("ground_truth:", ground_truth)

        # 用于修正其中RPY角的问题
        if ground_truth[0][4] < 0:
            ground_truth[0][4] = -ground_truth[0][4]
            ground_truth[0][5] = -ground_truth[0][5]

        print("ground_truth:", ground_truth)

        save_count_time = np.append(save_count_time, np.array([count_time]))
        save_pose_x_t_xyz = np.append(save_pose_x_t_xyz, np.array(x_t_update[0, 0:6]), axis=0)
        save_pose_GD_xyz = np.append(save_pose_GD_xyz, np.array([ground_truth[0, 0:6]]), axis=0)
        # normal_record_fcl = np.append(normal_record_fcl, -ht.transp ose(), axis = 0)
        # normal_record_finger = np.append(normal_record_finger, Zt.transpose(), axis = 0)
        np.save("save/save_pose_x_t.npy", save_pose_x_t_xyz)
        np.save("save/save_pose_GD.npy", save_pose_GD_xyz)
        np.save("save/save_normal_record_fcl.npy", normal_record_fcl)
        np.save("save/save_normal_record_finger.npy", normal_record_finger)
        np.save("save/save_count_time.npy", save_count_time)

def cross_product_func(delta_p_c):
    matrix_cross = np.array([ [0, -delta_p_c[0][2],  delta_p_c[0][1]],  [delta_p_c[0][2], 0, -delta_p_c[0][0]], [-delta_p_c[0][1], delta_p_c[0][0] , 0]])
    return matrix_cross

def mul_three(mat1, mat2, mat3):
    temp = np.matmul(mat1, mat2)
    temp1 = np.matmul(temp, mat3)
    return temp1

def derivate(pos_relative_to_cup):
    x, y = symbols('x, y')
    # z = -95.14 + 0.06723*x + 0.02707*y - 0.002612*x**2 - 0.0004485*x*y - 0.002529*y**2 - 0.0002661*x**3 + 0.0001468*x**2*y - 0.0002576*x*y**2 + 0.0001326*y**3 + 4.594e-05*x**4 + 3.959e-07*x**3*y + (9.146e-05)*(x**2)*(y**2) - 2.723e-07*x*y**3 + 4.597e-05*y**4 + 2.809e-08*x**5 + 2.589e-08*x**4*y + 4.971e-08*x**3*y**2 - 2.688e-10*x**2*y**3 + 2.559e-08*x*y**4 + 2.583e-08*y**5
    z = -0.0949 + 0.008409*x - 0.102*y - 2.044*x**2 - 2.009*x*y - 3.527*y**2 - 152.2*x**3 + 315.8*x**2*y - 260.8*x*y**2 + 310.6*y**3 + 45540*x**4 + 691.8*x**3*y + (91490)*(x**2)*(y**2) + 1228*x*y**3 + 4.643e+04*y**4 - 2.053e+04*x**5 - 4.477e+04*x**4*y + 1.758e+04*x**3*y**2 - 9.156e+04*x**2*y**3 + 4.595e+04*x*y**4 - 3.815e+04*y**5

    dx = diff(z, x)   # 对x求偏导
    dy = diff(z, y)   # 对y求偏导

    dxdydz_normalize = (dx**2 + dy**2 + 1)**(1/2)
    dx_normalize = dx / dxdydz_normalize
    dy_normalize = dy / dxdydz_normalize
    dz_normalize = 1 / dxdydz_normalize
    dxdx = diff(dx_normalize, x)
    dxdy = diff(dx_normalize, y)
    dydx = diff(dy_normalize, x)
    dydy = diff(dy_normalize, y)
    dzdx = diff(dz_normalize, x)
    dzdy = diff(dz_normalize, y)
    dxdx_result = dxdx.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    dxdy_result = dxdy.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    dydx_result = dydx.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    dydy_result = dydy.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    dzdx_result = dzdx.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    dzdy_result = dzdy.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    matrix = np.array([[dxdx_result,dydx_result,0],[dxdy_result,dydy_result, 0],[dzdx_result, dzdy_result, 0]])
    return matrix

def  contact_fcl():
    return 0

#ekf融合预测和normal后验更新
def ekf():
    return 0

def collision_part(pos_R_cup_global, pos_R_fingertip_global):
    global mesh_cup
    global mesh_fingertip
    R_cup_global = pos_R_cup_global[0:3, 0:3]
    pos_cup_global = pos_R_cup_global[0:3, 3]*1000

    R_fingertip_global = pos_R_fingertip_global[0:3, 0:3]
    pos_fingertip_global = pos_R_fingertip_global[0:3, 3] *1000

    t_cup_global = fcl.Transform(R_cup_global, pos_cup_global)

    t_fingertip_global = fcl.Transform(R_fingertip_global, pos_fingertip_global)

    o_cup = fcl.CollisionObject(mesh_cup, t_cup_global)
    o_fingertip = fcl.CollisionObject(mesh_fingertip, t_fingertip_global)

    req = fcl.CollisionRequest(enable_contact=True)
    res = fcl.CollisionResult()
    n_contacts = fcl.collide(o_cup, o_fingertip, req, res)

    return res

#碰撞检测显示
# finger_name "link_3.0_fcl"
def collision_test():
    global normal_fcl_1, normal_fcl_2
    # 变量提取
    link_3_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_3.0_fcl")
    link_3_tip_trans_global = f.posquat2trans(link_3_tip_pos_global)

    link_15_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_15.0_fcl")
    link_15_tip_trans_global = f.posquat2trans(link_15_tip_pos_global)

    cup_pose_global = f.get_relative_posquat(sim, "palm_link", "cup")
    cup_trans_global = f.posquat2trans(cup_pose_global)

    res1 = collision_part(cup_trans_global, link_3_tip_trans_global)
    res2 = collision_part(cup_trans_global, link_15_tip_trans_global)

    if res1.is_collision:
        contact = res1.contacts[0]
        normals = contact.normal
        normal_fcl_1 = np.array([[normals[0], normals[1], normals[2]]]).transpose()
        print("normal1:", normals)

    if res2.is_collision:
        contact = res2.contacts[0]
        normals = contact.normal
        normal_fcl_2 = np.array([[normals[0], normals[1], normals[2]]]).transpose()
        print("normal2:", normals)

    # req = fcl.DistanceRequest(enable_nearest_points=True)
    # res = fcl.DistanceResult()
    #
    # dist = fcl.distance(o_cup,o_fingertip,
    #                     req, res)

#fcl 碰撞结果显示
def print_collision_result(o1_name, o2_name, result):
    print( 'Collision between {} and {}:'.format(o1_name, o2_name))
    print( '-'*30)
    print( 'Collision?: {}'.format(result.is_collision))
    print( 'Number of contacts: {}'.format(len(result.contacts)))
    print( '')

#fcl 碰撞最短距离显示
def print_distance_result(o1_name, o2_name, result):
    print( 'Distance between {} and {}:'.format(o1_name, o2_name))
    print( '-'*30)
    print( 'Distance: {}'.format(result.min_distance))
    print( 'Closest Points:')
    print( result.nearest_points[0])
    print( result.nearest_points[1])
    print( '')

#****************************上面是函数定义************************************#
xml_path = "../UR5/UR5_tactile_allegro_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

#这里定义了link_3.0_tip,link_15.0_tip的测试，使用其他关节请重新定义kdl_kin变量
robot = URDF.from_xml_file('../UR5/allegro_hand_tactile_right.urdf')
kdl_tree = kdl_tree_from_urdf_model(robot)
kdl_kin_1 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
kdl_kin_2 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
kdl_chain = kdl_tree.getChain("palm_link", "link_3.0_tip")

robot_init()
Camera_set()

for i in range(50):
    for _ in range(50):
        sim.step()
    viewer.render()

# Move to cup
pose_cup = f.get_body_posquat(sim, "cup")
trans_cup = f.posquat2trans(pose_cup)
pos_pregrasp = [0.5, -0.05, 0.1]

#参数设定
# trans_pregrasp = np.array([[0, 0, 1, 0.12],
#                          [0, 1, 0, -0.25],
#                          [-1, 0, 0, 0.01],
#                          [0, 0, 0, 1]])
# trans_pregrasp = np.array([[0, 0, 1, 0.10],
#                          [0, 1, 0, -0.20],
#                          [-1, 0, 0, 0.01],
#                          [0, 0, 0, 1]])
# trans_pregrasp = np.array([[0, 0, 1, 0.08],
#                          [0, 1, 0, -0.20],
#                          [-1, 0, 0, 0.01],
#                          [0, 0, 0, 1]])

# #2
trans_pregrasp = np.array([[0, 0, 1, 0.08],
                         [0, 1, 0, -0.22],
                         [-1, 0, 0, 0.01],
                         [0, 0, 0, 1]])
# trans_pregrasp = np.array([[0, 0, 1, 0.07],
#                          [0, 1, 0, -0.15],
#                          [-1, 0, 0, 0.01],
#                          [0, 0, 0, 1]])

posequat = f.get_prepose_posequat(trans_cup, trans_pregrasp)
ctrl_wrist_pos = posequat[:3]
ctrl_wrist_quat = posequat[3:]

sim.model.eq_active[0] = True
for i in range(4):
    sim.data.mocap_pos[0] = ctrl_wrist_pos
    sim.data.mocap_quat[0] = ctrl_wrist_quat

    for _ in range(50):
        sim.step()
    viewer.render()

print("action start")
save_point = []
save_point_output = np.array([[]])
max_size = 0

#姿态估计所需要的变量
first_contact = True
second_contact = True
x_t = np.array([0, 0, 0, 0, 0, 0])
x_t_1 = np.array([0, 0, 0, 0, 0, 0])
contact_point_name_1 = ""
contact_point_name_2 = ""
count_time = 0

#全局变量的初始化
save_pose_x_t_xyz = np.array([[0, 0, 0, 0, 0, 0]])
save_pose_GD_xyz = np.array([[0, 0, 0, 0, 0, 0]])
save_pose_x_t_rpy = np.array([[0, 0, 0]])
save_pose_GD_rpy = np.array([[0, 0, 0]])
save_count_time = np.array([0])
q_pos_1 = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1], \
                   sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2], \
                    sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3], \
                    sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
q_pos_2 = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])
q_pos_pre_1 = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1], \
                   sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2], \
                    sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3], \
                    sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
q_pos_pre_2 = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3], \
                    sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])

normal_record_fcl = np.array([[0], [0], [0]]).transpose()
normal_record_finger = np.array([[0], [0], [0]]).transpose()

#update_step:
Pt_1 = 0.00001 * np.eye(6)

#normal:
normal_fcl_1 = np.array([[0, 0, 0]]).transpose()
normal_fcl_1 = np.array([[0, 0, 0]]).transpose()

#fcl库加载cup 的 BVH模型
obj_cup = fcl_python.OBJ("cup_1.obj")
verts_cup = obj_cup.get_vertices()
tris_cup = obj_cup.get_faces()
# Create mesh geometry

mesh_cup = fcl.BVHModel()
mesh_cup.beginModel(len(verts_cup), len(tris_cup))
mesh_cup.addSubModel(verts_cup, tris_cup)
mesh_cup.endModel()
print("len_verts_cup:", len(verts_cup))

#fcl库加载finger_tip 的 BVH模型
obj_fingertip = fcl_python.OBJ("fingertip_part.obj")
verts_fingertip = obj_fingertip.get_vertices()
tris_fingertip = obj_fingertip.get_faces()
print("len_verts_fingertip:", len(verts_fingertip))
print("len_tris_fingertip:", len(tris_fingertip))

mesh_fingertip = fcl.BVHModel()
mesh_fingertip.beginModel(len(verts_fingertip), len(tris_fingertip))
mesh_fingertip.addSubModel(verts_fingertip, tris_fingertip)
mesh_fingertip.endModel()

while True:
    sensor_data = sim.data.sensordata
    show_coordinate(sim, "palm_link")
    if not (np.array(sensor_data[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any():
        # sim.data.ctrl[6] = sim.data.ctrl[6] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + 0.001
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + 0.001
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + 0.001
    else:
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + 0.001
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + 0.001
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + 0.001

    if not (np.array(sensor_data[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
        tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any():
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] + 0.1
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + 0
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + 0.00
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + 0.00
    else:
        # sim.data.ctrl[19] = sim.data.ctrl[19] + 0.0005
        # #1
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + 0.005
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + 0.01

    # contact = sim.data.contact

    #前向prediction_model
    collision_test()
    prediction_model(sim)
    # print("sim.data.qpos:\n", np.where(np.array(sim.data.qpos) > 0.01))
    #碰撞检测，输出normal
    # print("Pt_1:", Pt_1)

    if (np.array(sensor_data) > 0.0).any():
            a = np.where(np.array(sensor_data) > 0.0)
            show_coordinate(sim, "palm_link")
            touch_visual(a, save_point_output)
    sim.step()
    viewer.render()

#     for _ in range(50):
#         if (np.array(sensor_data) > 0.0).any():
#             a = np.where(np.array(sensor_data) > 0.0)
#             show_coordinate(sim, "palm_link")
#             touch_visual(a, save_point_output)
#
#         sim.step()
#     viewer.render()
#
# while True:
#     sim.step()
#     viewer.render()
