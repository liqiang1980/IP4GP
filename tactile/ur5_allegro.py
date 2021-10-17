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

def show_coordinate(sim, body_name):
    palm_link_pose = f.get_body_posquat(sim, body_name)
    rot_palm_link = f.as_matrix(np.hstack((palm_link_pose[4:], palm_link_pose[3])))
    # viewer.add_marker(pos=sensor_pose[:3],)
    # print("rot_sensor:", rot_palm_link)
    #这里的最初的轴是它的y轴
    palm_link_rot_x = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
    palm_link_rot_y = np.array([[1, 0, 0],[0,0,1],[0,1,0]]) #绕X轴的旋转 -90
    palm_link_rot_z = np.array([[0, 0, -1],[0,1,0],[1,0,0]])#绕y轴的旋转 90

    # view中的mat是绕全局坐标系下的旋转，而不是局部坐标系下的旋转
    #红色是y轴， 绿色是x轴，蓝色是z轴
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_x, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    #手掌z轴的表示
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_y, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    #手掌x轴的表示
    viewer.add_marker(pos=palm_link_pose[:3], mat =palm_link_rot_z, type=const.GEOM_ARROW, label="palm_link_coordinate", size=np.array([0.001, 0.001, 0.5]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))

def control_input():
    if not (np.array(sensor_data) > 0.0).any():
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.005

#仅测试了link_3.0_tip 需要使用请对相应接口进行修改即可
def KDL_forward():
    q_pos = sim.data.qpos
    link_3_q = np.array(q_pos[13:17])
    pose_calc = kdl_kin.forward(link_3_q)
    relative_pose = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    relative_pose_trans = f.posquat2trans(relative_pose)
    print("relative_pose:\n", relative_pose)

def touch_visual(a):
    for i in a:
        for k,l in enumerate(i):
            s_name = model._sensor_id2name[i[k]]
            print("s_name:", s_name)
            sensor_pose = f.get_body_posquat(sim, s_name)
            rot_sensor = f.as_matrix(np.hstack((sensor_pose[4:], sensor_pose[3])))
            save_point_visual(sensor_pose)
            #用于控制方向，触觉传感器的方向问题
            test_rot = np.array([[1, 0, 0],[0,0,1],[0,1,0]])
            # viewer.add_marker(pos=sensor_pose[:3], mat =rot_sensor, type=2, label="test", size=np.array([0.3, 0.001, 0.001]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
            viewer.add_marker(pos=sensor_pose[:3], mat =test_rot, type=const.GEOM_ARROW, label="contact", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    viewer.render()

# 逆运动学解算
def move_ik_control():
    # desire_pos_quat = np.array( [ 0.08891832, 0.04635626, 0.0589164, 0.70435902, -0.02107556, 0.7083719, -0.04053936])
    desire_pos_quat = np.array([ 0.08916218, 0.04625495, 0.05806168, 0.69976209, -0.02073365, 0.7129112, -0.04075237])

    curr_posquat = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    # f.move_interperate_point(sim, desire_pos_quat, curr_posquat, viewer)
    flag_control = f.move_ik_kdl_finger_wdls_king(sim, desire_pos_quat)
    return flag_control

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

# def move_interperate_point(sim, desire_pos_quat):
#     curr_posquat = f.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
#     delta_k = 5
#     # X = np.arange(0, 1, 1)
#     # Y = [curr_posquat, desire_pos_quat]
#     for i in delta_k:
#         interpolate_point[i] = curr_posquat + (desire_pos_quat-curr_posquat)/delta_k*i
#     count_execute = 0;
#     for k in enumerate(interpolate_point):
#         while(count_execute < 200):
#             f.move_ik_kdl_finger_wdls(sim, k)
#             count_execute += 1
#         count_execute = 0

def control_grasp_joint_control():
    if not (np.array(sensor_data[0:72]) > 0.0).any():
        flag = False
        if(flag == False):
            flag = move_ik_control()
        KDL_forward()
        print("sim.data.ctrl:\n", sim.data.ctrl)
        # sim.data.ctrl[6] = sim.data.ctrl[6] + 0.005
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.005
        sim.data.ctrl[8] = sim.data.ctrl[8] + 0.005
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.005
    else:
        sim.data.ctrl[7] = sim.data.ctrl[7]
        sim.data.ctrl[8] = sim.data.ctrl[8]
        sim.data.ctrl[9] = sim.data.ctrl[9]

    if not (np.array(sensor_data[72:144]) > 0.0).any():
        # sim.data.ctrl[10] = sim.data.ctrl[10] + 0.005
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

#****************************上面是函数定义************************************#
xml_path = "/home/ycj/tactile/UR5/UR5_allegro_test.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

#这里仅定义了link_3.0_tip的测试，使用其他关节请重新定义kdl_kin变量
robot = URDF.from_xml_file('/home/ycj/tactile/UR5/allegro_hand_tactile_right.urdf')
kdl_tree = kdl_tree_from_urdf_model(robot)
kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
kdl_chain = kdl_tree.getChain("palm_link", "link_3.0_tip")

robot_init()
# 相机视角
Camera_set()

for i in range(50):
    for _ in range(50):
        sim.step()
    viewer.render()

# Move to bottle
pose_bottle = f.get_body_posquat(sim, "bottle")
trans_bottle = f.posquat2trans(pose_bottle)
pos_pregrasp = [0.5, -0.05, 0.1]

trans_pregrasp = np.array([[0, 0, 1, 0.15],
                         [0, 1, 0, -0.25],
                         [-1, 0, 0, 0.1],
                         [0, 0, 0, 1]])

posequat = f.get_prepose_posequat(trans_bottle, trans_pregrasp)
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

while True:
    sensor_data = sim.data.sensordata
    show_coordinate(sim, "palm_link")
    print("sensor_data:\n", len(sensor_data))
    
    if not (np.array(sensor_data[0:72]) > 0.0).any():
        flag = False
        if(flag == False):
            flag = move_ik_control()
        KDL_forward()

    contact = sim.data.contact

    if (np.array(sensor_data) > 0.0).any():
        a = np.where(np.array(sensor_data) > 0.0)

    for _ in range(50):
        if (np.array(sensor_data) > 0.0).any():
            a = np.where(np.array(sensor_data) > 0.0)
            print("a:", a)
            # print("where_a:", a);
            show_coordinate(sim, "palm_link")
            touch_visual(a)

        sim.step()
    viewer.render()

while True:
    sim.step()
    viewer.render()
