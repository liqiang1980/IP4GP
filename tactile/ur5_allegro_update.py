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

#指定一个表格，来对应相应的关节
#关节对应顺序表
#6   食指根部旋转    #7食指   从下往上第二关节旋转  #8   食指从下往上第三关节旋转 #9   食指最后一个关节   link_3.0_tip    食指指尖
#10 中指根部旋转   #11中指从下往上第二关节旋转   #12 中指从下往上第三关节旋转#13 中指最后一个关节   link_7.0_tip    中指指尖
#14 无名根部旋转   #15无名从下往上第二关节旋转   #16 无名从下往上第三关节旋转#17 无名最后一个关节   link_11.0_tip  无名指尖
#18拇指大旋转        #19 拇指根部旋转                             #20 拇指大弯曲关节                           #21 拇指小弯曲关节   link_15.0_tip  拇指指尖
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
    sim.data.ctrl[0] = 0.8
    sim.data.ctrl[1] = -0.78
    sim.data.ctrl[2] = 1.13
    sim.data.ctrl[3] = -1.
    sim.data.ctrl[4] = 0
    sim.data.ctrl[5] = -0.3

def hand_init():
    sim.data.ctrl[6] = -0.00502071
    sim.data.ctrl[7] = 0.2
    sim.data.ctrl[8] = 0.68513787
    sim.data.ctrl[9] = 0.85640426

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
    palm_link_rot_x = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
    palm_link_rot_y = np.array([[1, 0, 0],[0,0,1],[0,1,0]]) #绕X轴的旋转 -90
    palm_link_rot_z = np.array([[0, 0, -1],[0,1,0],[1,0,0]])#绕y轴的旋转 90

    # view中的mat是绕全局坐标系下的旋转，而不是局部坐标系下的旋转
    #红色是x轴， 绿色是y轴，蓝色是z轴
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_x, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_y, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_z, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

def control_input():
    if not (np.array(sensor_data) > 0.0).any():
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.005

#仅测试了link_3.0_tip 需要使用请对相应接口进行修改即可
#pykdl库的安装
def KDL_forward():
    q_pos = sim.data.qpos
    link_3_q = np.array(q_pos[13:17])
    pose_calc = kdl_kin.forward(link_3_q)
    relative_pose = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    relative_pose_trans = f.posquat2trans(relative_pose)

#触觉点可视化 arrow
def touch_visual(a, save_point_output):
    global max_size
    truth = f.get_relative_posquat(sim, "base_link", "cup")

    save_point_use = np.array([[0,0,0,0,0,0,0]])
    save_point_use = np.append(save_point_use, np.array([truth]),axis = 0)
    for i in a:
        for k,l in enumerate(i):
            s_name = model._sensor_id2name[i[k]]
            sensor_pose = f.get_body_posquat(sim, s_name)
            relative_pose = f.get_relative_posquat(sim, "base_link", s_name)
            save_point_use = np.append(save_point_use, np.array([relative_pose]),axis = 0)

            rot_sensor = f.as_matrix(np.hstack((sensor_pose[4:], sensor_pose[3])))
            #用于控制方向，触觉传感器的方向问题
            test_rot = np.array([[1, 0, 0],[0,0,1],[0,1,0]])
            viewer.add_marker(pos=sensor_pose[:3], mat =test_rot, type=const.GEOM_ARROW, label="contact", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

    if save_point_use.shape[0] > max_size:
        save_point_output = save_point_use
        np.save("output.npy", save_point_output)
        where_a = np.where(np.array(sensor_data) > 0.0)

    max_size = max(save_point_use.shape[0],max_size)
    viewer.render()

# 逆运动学解算
def move_ik_control():
    desire_pos_quat = np.array([ 0.08916218, 0.04625495, 0.05806168, 0.69976209, -0.02073365, 0.7129112, -0.04075237])
    curr_posquat = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    flag_control = f.move_ik_kdl_finger_wdls_king(sim, desire_pos_quat)
    return flag_control

#接触过程中的全部触觉点的记录和渲染
def save_point_visual(pos):
    delta = 0.01
    count = 0;
    for i in save_point:
        if np.linalg.norm(pos[:3] - i[:3])>delta:
            count += 1
    if count == len(save_point):
        save_point.append(pos)
    print(len(save_point))
    test_rot_point = np.array([[1, 0, 0],[0,0,1],[0,1,0]])
    for i in save_point:
        viewer.add_marker(pos=i[:3], mat =test_rot_point, type=const.GEOM_SPHERE, label="", size=np.array([0.005, 0.005, 0.005]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

#关节空间的位置闭合抓取
def control_grasp_joint_control():
    if not (np.array(sensor_data[0:72]) > 0.0).any():
        flag = False
        if(flag == False):
            flag = move_ik_control()
        KDL_forward()
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.005
    else:
        sim.data.ctrl[7] = sim.data.ctrl[7]
        sim.data.ctrl[8] = sim.data.ctrl[8]
        sim.data.ctrl[9] = sim.data.ctrl[9]

    if not (np.array(sensor_data[72:144]) > 0.0).any():
        sim.data.ctrl[11] = sim.data.ctrl[11] + 0.006
        sim.data.ctrl[12] = sim.data.ctrl[12] + 0.003
        sim.data.ctrl[13] = sim.data.ctrl[13] + 0.003
    else:
        sim.data.ctrl[11] = sim.data.ctrl[11]
        sim.data.ctrl[12] = sim.data.ctrl[12]
        sim.data.ctrl[13] = sim.data.ctrl[13]

    if not (np.array(sensor_data[144:216]) > 0.0).any():
        sim.data.ctrl[15] = sim.data.ctrl[15] + 0.006
        sim.data.ctrl[16] = sim.data.ctrl[16] + 0.003
        sim.data.ctrl[17] = sim.data.ctrl[17] + 0.003
    else:
        sim.data.ctrl[15] = sim.data.ctrl[15]
        sim.data.ctrl[16] = sim.data.ctrl[16]
        sim.data.ctrl[17] = sim.data.ctrl[17]

    if not (np.array(sensor_data[216:288]) > 0.0).any():
        sim.data.ctrl[18] = sim.data.ctrl[18] + 0.015   #0.01可以
        # sim.data.ctrl[19] = sim.data.ctrl[19] + 0.002
        sim.data.ctrl[20] = sim.data.ctrl[20] + 0.002
        sim.data.ctrl[21] = sim.data.ctrl[21] + 0.002
    else:
        sim.data.ctrl[18] = sim.data.ctrl[18]
        sim.data.ctrl[19] = sim.data.ctrl[19]
        sim.data.ctrl[20] = sim.data.ctrl[20]
        sim.data.ctrl[21] = sim.data.ctrl[21]


#前向prediction model。 将手指运动通过grasp matrix 转换到物体姿态的变化
#prediction 更新公式 ：x_t = x_t_1 + G*J*u_t*delta_t
def prediction_model(sim):
    #状态变量定义
    global first_contact                                 #首次接触判断flag
    global x_t                                                     #实时物体姿态
    global x_t_1                                                #上一时刻物体姿态
    global contact_point_name                #接触点名字
    global debug_temps
    global count_time                                    #时间记录
    global save_pose_x_t_xyz                    #输出记录： pose的xyz
    global save_pose_GD_xyz                    #输出记录： ground truth 的xyz
    global save_pose_x_t_rpy                    #输出记录： pose的xyz
    global save_pose_GD_rpy                    #输出记录： ground truth 的rpy
    global save_count_time
    global q_pos_test_pre                            #上一个时刻的q_pos 用于计算关节空间偏差

    #update_step:
    global Pt_1

    #获取手指的姿态（link_3_tip_trans） 和 cup姿态（cup_trans）
    link_3_tip_pos = f.get_relative_posquat(sim, "palm_link", "link_3.0_fcl")
    link_3_tip_trans = f.posquat2trans(link_3_tip_pos)
    cup_pose = f.get_relative_posquat(sim, "palm_link", "cup")
    cup_trans = f.posquat2trans(cup_pose)

    if count_time == 1:
            np.save("pos_save/link_3_tip_trans.npy", link_3_tip_trans)
            np.save("pos_save/cup_trans.npy", cup_trans)

    print("count_time:", count_time)

    # 1.接触时的 初始姿态记录与更新
    # 2. 接触点记录
    # 3. 初始关节位置记录
    if (np.array(sensor_data[0:71]) > 0.0).any():
        if first_contact == True:
            x_t_1 = f.get_relative_posquat(sim, "palm_link", "cup")
            x_t_1 = np.array([f.pos_quat2pos_XYZ_RPY(x_t_1)])
            x_t = x_t_1

            #这里需要修改
            Pt_1 = np.array([[0.01, 0, 0, 0, 0, 0],
                            [0, 0.01, 0, 0, 0, 0],
                            [0, 0, 0.01, 0, 0, 0],
                            [0, 0, 0, 0.1, 0, 0],
                            [0, 0, 0, 0, 0.1, 0],
                            [0, 0, 0, 0, 0, 0.1]])


            q_pos_test = sim.data.qpos
            q_pos_temp  = np.array([q_pos_test[11], q_pos_test[127], q_pos_test[164], q_pos_test[201]])
            q_pos_test_pre = q_pos_temp

            # 记录第一个接触点：
            a = np.where(np.array(sensor_data) > 0.0)
            for i in a:
                s_name = model._sensor_id2name[i[0]]
                contact_point_name = s_name
            first_contact = False

        # 127,164,201是食指三个关节
        #当前时刻与上一时刻的关节的差值记录
        q_pos_test = sim.data.qpos
        q_pos_temp  = np.array([q_pos_test[11], q_pos_test[127], q_pos_test[164], q_pos_test[201]])
        delta = q_pos_temp - q_pos_test_pre
        q_pos_test_pre = q_pos_temp
        ee_jac_3 = kdl_kin.jacobian(q_pos_temp)

        count_time += 1

        u_t = delta.T

        #获取接触点位置和旋转矩阵 （相对于palm_link）
        posquat_base_point = f.get_relative_posquat(sim, "palm_link", contact_point_name)
        R_i = f.posquat2trans(posquat_base_point)[0:3, 0:3]
        cup_pose = f.get_relative_posquat(sim, "palm_link", "cup")
        R_i_cup = f.posquat2trans(cup_pose)[0:3, 0:3]
        contact_point_posquat = f.get_relative_posquat(sim, "palm_link", contact_point_name)
        contact_point_pos = np.array([posquat_base_point[0:3]])

        #获取目标物体位置
        object_position = x_t_1[0, 0:3]#物体的目标姿态只能通过估计得到的位置和姿态获得
        zero = np.zeros((3,3))
        delta_p_c = np.array(contact_point_pos - object_position)

        #叉乘矩阵获取
        cross_product = cross_product_func(delta_p_c)

        #grasp matrix 获取
        grasp_matrix_up = np.hstack([R_i, zero]);
        cross_product_R_i = np.matmul(R_i, cross_product)
        grasp_matrix_down = np.hstack([cross_product_R_i, R_i])
        grasp_matrix = np.vstack([grasp_matrix_up, grasp_matrix_down])
        grasp_matrix_trans = grasp_matrix.T
        grasp_matrix_trans_pinv = np.linalg.pinv(grasp_matrix_trans)

        # # 雅克比的伪逆
        J_finger_3 = ee_jac_3
        # G*J*u_t*delta_t
        prediction_part1 = np.matmul(grasp_matrix_trans_pinv, J_finger_3)
        prediction_part2 = np.matmul(prediction_part1, u_t)
        prediction = prediction_part2[0, 0:]
        # print("prediction:", prediction)

        #通过prediction model得到的 估计姿态更新
        x_t = x_t_1 - prediction
        x_t_1 = x_t

        #参数初始化
        #Pt的更新
        de_F = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        # update_step
        V_t = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])

        de_F_mul_Pt_1 = np.matmul(de_F, Pt_1)
        temp_Pt = np.matmul(de_F_mul_Pt_1, de_F.transpose())
        Pt = temp_Pt + V_t

        # pos_relative_to_cup 还未定义 先使用真值测试算法可靠性
        pos_relative_to_cup = f.get_relative_posquat(sim, "cup", contact_point_name)
        de_n_c = derivate(pos_relative_to_cup) #临时变量 [3*3]
        Ht = np.matmul(de_n_c, grasp_matrix_trans[0:3, :]) #3*6

        Rt = np.array([[0.5, 0, 0],  #3*3
                        [0, 0.5, 0],
                        [0, 0, 0.5]])

        # # kt = Pt * de_n_y.transpose() * (de_n_y * Pt * de_n_y.transpose() + Rt)
        Kt_behind = mul_three(Ht, Pt, Ht.transpose())  #3*6  * 6*6 * 6*3
        Kt_behind_Rt = Kt_behind + Rt   #3*3
        Kt_behind_Rt = Kt_behind_Rt.astype(np.float)
        Kt_behind_Rt_inv = np.linalg.inv(Kt_behind_Rt) #3*3
        Kt = mul_three(Pt, Ht.transpose(), Kt_behind_Rt_inv) #6*6  * 6*3 * 3*3 = 6*3
        print("Kt:", Kt)
        #

        #normals on fingertip
        normal_init = np.array([[1, 0, 0]]).transpose()
        posquat_base_point = f.get_relative_posquat(sim, "palm_link", contact_point_name)
        R_palm_contact = f.posquat2trans(posquat_base_point)[0:3, 0:3]

        posquat_base_cup = f.get_relative_posquat(sim, "palm_link", "cup")
        R_palm_cup = f.posquat2trans(posquat_base_cup)[0:3, 0:3]
        R_palm_cup_inv = np.linalg.inv(R_palm_cup)
        R_cup_contact = np.matmul(R_palm_contact, R_palm_cup_inv)

        normal_on_finger = np.matmul(R_cup_contact, normal_init)
        print("normal_on_finger:", normal_on_finger)

        # #如果没有接触就返回上一接触时的接触点，如果有接触就返回接触点法向量
        Zt = normal_on_finger
        ht = normal_fcl
        err_Zt_ht = Zt + ht

        Kt_err_Zt_ht = np.matmul(Kt, err_Zt_ht).transpose()
        # print("Kt_err_Zt_ht:", Kt_err_Zt_ht)
        x_t_update = x_t + np.matmul(Kt, err_Zt_ht).transpose() #1*6

        # print("********************************", count_time)
        x_t_1 = x_t_update
        x_t_1 = x_t_1.astype(np.float)

        Kt_mul_Ht = np.matmul(Kt, Ht)
        I_mat = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        err_I_Kt_Ht = I_mat - Kt_mul_Ht
        pt_update = np.matmul(err_I_Kt_Ht, Pt)
        Pt_1 = pt_update

        #cup真值的获取
        ground_truth = f.get_relative_posquat(sim, "palm_link", "cup")
        ground_truth = np.array([f.pos_quat2pos_XYZ_RPY(ground_truth)])

        #cup估计姿态 和 ground truth 记录，用于 图线输出。
        #x_t 修改成了 x_t_update
        save_count_time = np.append(save_count_time, np.array([count_time]))
        save_pose_x_t_xyz = np.append(save_pose_x_t_xyz, np.array(x_t_update[0, 0:3]), axis=0)
        save_pose_GD_xyz = np.append(save_pose_GD_xyz, np.array([ground_truth[0, 0:3]]), axis=0)
        save_pose_x_t_rpy = np.append(save_pose_x_t_rpy, np.array(x_t_update[0, 3:6]), axis=0)
        save_pose_GD_rpy = np.append(save_pose_GD_rpy, np.array([ground_truth[0, 3:6]]), axis=0)
        np.save("save/save_pose_x_t_xyz.npy", save_pose_x_t_xyz)
        np.save("save/save_pose_GD_xyz.npy", save_pose_GD_xyz)
        np.save("save/save_pose_x_t_rpy.npy", save_pose_x_t_rpy)
        np.save("save/save_pose_GD_rpy.npy", save_pose_GD_rpy)
        np.save("save/save_count_time.npy", save_count_time)

def cross_product_func(delta_p_c):
    matrix_cross = np.array([ [0, -delta_p_c[0][2],  delta_p_c[0][1]],  [delta_p_c[0][2], 0, -delta_p_c[0][0]], [-delta_p_c[0][1], delta_p_c[0][0] , 0]])
    return matrix_cross

def mul_three(mat1, mat2, mat3):
    temp = np.matmul(mat1, mat2)
    temp1 = np.matmul(temp, mat3)
    return temp1
# derivate_normal_and_contact_point
# de_n_y = de_n_c * de_c_y 是偏导数的矩阵

# kt = P * de_n_y.transpose() * (de_n_y * P * de_n_y.transpose() + R)

# R 的矩阵的定义
# P 是 方阵 6*6 因为是6维的观测量
# de_n_y is 3*3 mul 3*6 = 3*6
# 3*6 mul * 6*6 mul 6*3 = 3*3
# p = 6*6
# 6*6 mul 6*3 = 6*3
# kt = 6*3
# y_t = y_t_1 + Kt * (z_t - h(y_t))
# 6*1 + 6*3(3*1) = 6*1
#
# Kt 属于

def derivate(pos_relative_to_cup):
    x, y = symbols('x, y')
    z = -95.14 + 0.06723*x + 0.02707*y - 0.002612*x**2 - 0.0004485*x*y - 0.002529*y**2 - 0.0002661*x**3 + 0.0001468*x**2*y - 0.0002576*x*y**2 + 0.0001326*y**3 + 4.594e-05*x**4 + 3.959e-07*x**3*y + (9.146e-05)*(x**2)*(y**2) - 2.723e-07*x*y**3 + 4.597e-05*y**4 + 2.809e-08*x**5 + 2.589e-08*x**4*y + 4.971e-08*x**3*y**2 - 2.688e-10*x**2*y**3 + 2.559e-08*x*y**4 + 2.583e-08*y**5
    # result = z.subs({x: 1, y: 2})   # 用数值分别对x、y进行替换
    dx = diff(z, x)   # 对x求偏导
    dy = diff(z, y)   # 对y求偏导
    dxdx = diff(dx, x)
    dxdy = diff(dx, y)
    dydx = diff(dy, x)
    dydy = diff(dy, y)
    result11 = dxdx.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    result12 = dxdy.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    result21 = dydx.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    result22 = dydy.subs({x: pos_relative_to_cup[0], y: pos_relative_to_cup[1]})
    matrix = np.array([[result11,result12,0],[result21,result22, 0],[0, 0, 0]])
    # G**T[0:2, 0:5] * matrix
    return matrix

def  contact_fcl():
    return 0

#ekf融合预测和normal后验更新
def ekf():
    return 0

#碰撞检测显示
def collision_test():
    global mesh_cup
    global mesh_fingertip
    link_3_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_3.0_fcl")
    link_3_tip_trans_global = f.posquat2trans(link_3_tip_pos_global)
    cup_pose_global = f.get_relative_posquat(sim, "palm_link", "cup")
    cup_trans_global = f.posquat2trans(cup_pose_global)
    pos_R_cup_global = cup_trans_global
    pos_R_fingertip_global = link_3_tip_trans_global

    R_cup_global = pos_R_cup_global[0:3, 0:3]
    pos_cup_global = pos_R_cup_global[0:3, 3]*1000

    R_fingertip_global = pos_R_fingertip_global[0:3, 0:3]
    pos_fingertip_global = pos_R_fingertip_global[0:3, 3] *1000

    t_cup_global = fcl.Transform(R_cup_global, pos_cup_global)

    t_fingertip_global =  fcl.Transform(R_fingertip_global, pos_fingertip_global)

    o_cup = fcl.CollisionObject(mesh_cup, t_cup_global)
    o_fingertip = fcl.CollisionObject(mesh_fingertip, t_fingertip_global)

    req = fcl.CollisionRequest(enable_contact=True)
    res = fcl.CollisionResult()
    n_contacts = fcl.collide(o_cup, o_fingertip, req, res)

    #碰撞时输出normal
    if res.is_collision:
        contact = res.contacts[0]
        normals = contact.normal
        normal_fcl = normals.transpose()
        print("normal:", normals)

    print_collision_result('cup', 'fingertip', res)

    req = fcl.DistanceRequest(enable_nearest_points=True)
    res = fcl.DistanceResult()

    dist = fcl.distance(o_cup,o_fingertip,
                        req, res)
    print_distance_result('o_cup', 'o_fingertip', res)

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
xml_path = "../../UR5/UR5_tactile_allegro_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

#这里仅定义了link_3.0_tip的测试，使用其他关节请重新定义kdl_kin变量
robot = URDF.from_xml_file('../../UR5/allegro_hand_tactile_right.urdf')
kdl_tree = kdl_tree_from_urdf_model(robot)
kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
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

trans_pregrasp = np.array([[0, 0, 1, 0.12],
                         [0, 1, 0, -0.25],
                         [-1, 0, 0, 0.01],
                         [0, 0, 0, 1]])

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
x_t = np.array([0, 0, 0, 0, 0, 0])
x_t_1 = np.array([0, 0, 0, 0, 0, 0])
contact_point_name = ""
debug_temps = np.array([])
count_time  = 0

#全局变量的初始化
save_pose_x_t_xyz = np.array([[0,0,0]])
save_pose_GD_xyz = np.array([[0,0,0]])
save_pose_x_t_rpy = np.array([[0,0,0]])
save_pose_GD_rpy = np.array([[0,0,0]])
save_count_time = np.array([0])
q_pos_test_pre =  np.array([sim.data.qpos[11], sim.data.qpos[127], sim.data.qpos[164], sim.data.qpos[201]])

#update_step:
Pt_1 = np.array([[0.001, 0, 0, 0, 0, 0],
                [0, 0.001, 0, 0, 0, 0],
                [0, 0, 0.001, 0, 0, 0],
                [0, 0, 0, 0.001, 0, 0],
                [0, 0, 0, 0, 0.001, 0],
                [0, 0, 0, 0, 0, 0.001]])

#normal:
normal_fcl = np.array([[0, 0, 0]]).transpose()
#fcl库加载cup 的 BVH模型
obj_cup = fcl_python.OBJ( "cup_1.obj")
verts_cup = obj_cup.get_vertices()
tris_cup = obj_cup.get_faces()
# Create mesh geometry

mesh_cup = fcl.BVHModel()
mesh_cup.beginModel(len(verts_cup), len(tris_cup))
mesh_cup.addSubModel(verts_cup, tris_cup)
mesh_cup.endModel()
print("len_verts_cup:", len(verts_cup))

#fcl库加载finger_tip 的 BVH模型
obj_fingertip = fcl_python.OBJ( "fingertip_part.obj")
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
    if not (np.array(sensor_data[0:72]) > 0.0).any():
        # sim.data.ctrl[6] = sim.data.ctrl[6] + 0.005
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.005
    else:
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.0005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.0005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.0005

    contact = sim.data.contact

    #前向prediction_model
    prediction_model(sim)
    #碰撞检测，输出normal
    collision_test()
    # print("Pt_1:", Pt_1)

    for _ in range(50):
        if (np.array(sensor_data) > 0.0).any():
            a = np.where(np.array(sensor_data) > 0.0)
            show_coordinate(sim, "palm_link")
            touch_visual(a, save_point_output)

        sim.step()
    viewer.render()

while True:
    sim.step()
    viewer.render()
