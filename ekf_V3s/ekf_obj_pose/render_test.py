import math
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import fcl
import fcl_python
import func as f
import func2 as f2
import storeQR as sQR
import surfaceFitting as sf
import tactile_allegro_mujo_const
import test_Plot_plus as plt_plus

#used for configure parameters
import sys
from xml.dom.minidom import parseString


def read_xml(xml_file):
        with open(xml_file, 'r') as f:
                data = f.read()
        return parseString(data)

if (len(sys.argv) < 2):
        print ("Error: Missing parameter.")
else:
        dom = read_xml(sys.argv[1])
        hand_name = dom.getElementsByTagName('name')[0].firstChild.data
        hand_param = []
        hand_param.append(hand_name)

        #parse fingers' parameters
        #the parameters will be organized in list type in the following way
        #['allegro', ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}]]
        fingers = dom.getElementsByTagName('finger')
        for finger in fingers:
                finger_name = finger.getAttribute("name")
                is_used = finger.getElementsByTagName("used")[0]
                print(is_used.firstChild.data)
                js = finger.getElementsByTagName('init_posture')
                for jnt in js:
                        j_init_dic = {
                            "j1":jnt.getElementsByTagName("j1")[0].firstChild.data,
                            "j2":jnt.getElementsByTagName("j2")[0].firstChild.data,
                            "j3":jnt.getElementsByTagName("j3")[0].firstChild.data,
                            "j4":jnt.getElementsByTagName("j4")[0].firstChild.data
                        }
                finger_param = [finger_name, is_used.firstChild.data, j_init_dic]
                hand_param.append(finger_param)
        print(hand_param)
        print(hand_param[1][1])
        print(hand_param[2][1])
        print(hand_param[3][1])
        print(hand_param[4][1])
        #access to data in configure file
        #hand_param[0]: name of the hand
        #hand_param[1]: parameter of "ff" finger
        #hand_param[1][0]: name of "ff" finger
        #hand_param[1][1]: is "ff" finger used for grasping
        #hand_param[1][2]["j1"]: init of j1 of "ff" finger

        #parse object info
        object_param = []
        object_name = dom.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        object_position_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_position')[0].firstChild.data
        object_orientation_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_orientation')[0].firstChild.data
        object_param.append(object_name)
        object_param.append(object_position_noise)
        object_param.append(object_orientation_noise)
        print(object_param)

######################################
# v3b: Gaussian noise add to h() to become z_t
#
#
#######################################
#########################################   GLOBAL VARIABLES   #########################################################
xml_path = "../../robots/UR5_tactile_allegro_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

pose_cup = f.get_body_posquat(sim, object_param[0])
trans_cup = f.posquat2trans(pose_cup)
trans_pregrasp = np.array([[0, 0, 1, 0.1],  # cup参考系
                           [0, 1, 0, -0.23],
                           [-1, 0, 0, 0.07],
                           [0, 0, 0, 1]])
posequat = f.get_prepose_posequat(trans_cup, trans_pregrasp)  # 转为世界参考系
print("INIT:", posequat)
ctrl_wrist_pos = posequat[:3]
ctrl_wrist_quat = posequat[3:]
max_size = 0
flag = False  # 首次接触判断flag，False表示尚未锁定第一个接触点
c_point_name = ""
q_pos_pre = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
y_t = np.array([0, 0, 0, 0, 0, 0])
n_o = np.array([0, 0, 0, 0, 0, 0])
pos_contact1 = np.array([0, 0, 0])
pos_contact_last = np.array([0, 0, 0])
S1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
y_t_pre = np.array([0, 0, 0, 0, 0, 0])
posquat_contact_pre = np.array([0, 0, 0, 0, 0, 0, 0])
surface_x = np.empty([1, 0])
surface_y = np.empty([1, 0])
surface_z = np.empty([1, 0])
count_time = 0  # 时间记录
# nor_in_p = np.zeros(3)
# G_big = np.zeros([6, 6])
save_model = 0
P = np.eye(6)
u_t = np.empty([1, 4])
# fin_num = 0
trans_palm2cup = f.posquat2trans(f.get_relative_posquat(sim, "cup", "palm_link"))
trans_cup2palm = f.posquat2trans(f.get_relative_posquat(sim, "palm_link", "cup"))
z0 = np.zeros(3)
z1 = np.zeros(3)
z2 = np.zeros(3)
z3 = np.zeros(3)
Pmodel = 1
conver_rate = 40

# 以下为EKF后验过程的变量
P_ori = 1000 * np.ones([22, 22])

y_t_update = np.array([np.zeros(10)])

# kinematic chain for all fingers
robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
#first finger
kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
#middle finger
kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
#ring finger
kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
#thumb
kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
kdl_tree = kdl_tree_from_urdf_model(robot)

# 记录用变量
save_count_time = np.array([0])
save_pose_y_t_xyz = np.array([0, 0, 0])
save_pose_y_t_rpy = np.array([0, 0, 0])
save_pose_GD_xyz = np.array([0, 0, 0])
save_pose_GD_rpy = np.array([0, 0, 0])
save_error_xyz = np.array([0, 0, 0])
save_error_rpy = np.array([0, 0, 0])

# fcl库加载cup 的 BVH模型
obj_cup = fcl_python.OBJ("cup_1.obj")
verts_cup = obj_cup.get_vertices()
tris_cup = obj_cup.get_faces()

# Create mesh geometry
mesh_cup = fcl.BVHModel()
mesh_cup.beginModel(len(verts_cup), len(tris_cup))
mesh_cup.addSubModel(verts_cup, tris_cup)
mesh_cup.endModel()
print("len_verts_cup:", len(verts_cup))

# fcl库加载finger_tip 的 BVH模型
obj_fingertip = fcl_python.OBJ("fingertip_part.obj")
verts_fingertip = obj_fingertip.get_vertices()
tris_fingertip = obj_fingertip.get_faces()
print("len_verts_fingertip:", len(verts_fingertip))
print("len_tris_fingertip:", len(tris_fingertip))

mesh_fingertip = fcl.BVHModel()
mesh_fingertip.beginModel(len(verts_fingertip), len(tris_fingertip))
mesh_fingertip.addSubModel(verts_fingertip, tris_fingertip)
mesh_fingertip.endModel()

err_all = np.zeros(6)
err = np.zeros(6)

def interacting(hand_param):
    global err_all

    f2.pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
    for ii in range(37):
        if hand_param[1][1] == '1':
            f2.index_finger(sim, 0.015, 0.00001)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.015, 0.00001)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.015, 0.001)
    # Slow Downt whether any array element along a given axis evaluates to True.
    for ij in range(30):
        if hand_param[1][1] == '1':
            f2.index_finger(sim, 0.0055, 0.004)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.0036, 0.003)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.0032, 0.0029)
        if hand_param[4][1] == '1':
            f2.thumb(sim, 0.003, 0.003)
        #todo EKF() already did the rendering, why here the sim step and rendering still needed?
        for i in range(4):
            for _ in range(5):
                sim.step()
            viewer.render()
    # Rotate
    for ij in range(30):
        # f2.index_finger(sim, 0.0055, 0.0038)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.0003, 0.003)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.0005, 0.005)
        if hand_param[4][1] == '1':
            f2.thumb(sim, 0.003, 0.003)
        for i in range(4):
            for _ in range(5):
                sim.step()
            viewer.render()



############################>>>>>>>>>>>>>>>    MAIN LOOP    <<<<<<<<<<<<<###############################################

f2.robot_init(sim)
f2.Camera_set(viewer, model)
sim.model.eq_active[0] = True
for i in range(5):
    for _ in range(50):
        sim.step()
    viewer.render()

    sim.data.mocap_pos[0] = ctrl_wrist_pos  # mocap控制需要用世界参考系
    sim.data.mocap_quat[0] = ctrl_wrist_quat  # mocap控制需要用世界参考系
for i in range(4):
    for _ in range(50):
        sim.step()
    viewer.render()

interacting(hand_param)

