import math
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import util_geometry as ug
import mujoco_environment as mu_env
import robot_control as robcontrol
import tactile_allegro_mujo_const
import viz

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

#########################################   GLOBAL VARIABLES   #########################################################
xml_path = "../../robots/UR5_tactile_allegro_hand_obj_frozen.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

pose_cup = ug.get_body_posquat(sim, object_param[0])
trans_cup = ug.posquat2trans(pose_cup)
trans_pregrasp = np.array([[0, 0, 1, 0.1],  # cup参考系
                           [0, 1, 0, -0.23],
                           [-1, 0, 0, 0.07],
                           [0, 0, 0, 1]])
posequat = ug.get_prepose_posequat(trans_cup, trans_pregrasp)  # 转为世界参考系
print("INIT:", posequat)
ctrl_wrist_pos = posequat[:3]
ctrl_wrist_quat = posequat[3:]

q_pos_pre = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])

# kinematic chain for all fingers
robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
#first finger
kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
#middle finger
kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
#ring finger
kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
#thumb
kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
kdl_tree = kdl_tree_from_urdf_model(robot)


def interacting(hand_param):
    global err_all
    rob_control.pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
    for ii in range(37):
        if hand_param[1][1] == '1':
            rob_control.index_finger(sim, 0.015, 0.00001)
        if hand_param[2][1] == '1':
            rob_control.middle_finger(sim, 0.015, 0.00001)
        if hand_param[3][1] == '1':
            rob_control.ring_finger(sim, 0.015, 0.001)
    # Slow Down whether any array element along a given axis evaluates to True.
    for ij in range(300):
        if hand_param[1][1] == '1':
            rob_control.index_finger(sim, 0.0055, 0.004)
        if hand_param[2][1] == '1':
            rob_control.middle_finger(sim, 0.0036, 0.003)
        if hand_param[3][1] == '1':
            rob_control.ring_finger(sim, 0.0032, 0.0029)
        if hand_param[4][1] == '1':
            rob_control.thumb(sim, 0.003, 0.003)
        sim.step()
        viewer.render()

    # Rotate
    for ij in range(10000):
        if hand_param[1][1] == '1':
            rob_control.index_finger(sim, 0.0055, 0.0038)
        if hand_param[2][1] == '1':
            rob_control.middle_finger(sim, 0.0003, 0.003)
        if hand_param[3][1] == '1':
            rob_control.ring_finger(sim, 0.0005, 0.005)
        if hand_param[4][1] == '1':
            rob_control.thumb(sim, 0.003, 0.003)
        sim.step()
        viewer.render()


############################>>>>>>>>>>>>>>>    MAIN LOOP    <<<<<<<<<<<<<###############################################

rob_control = robcontrol
rob_control.robot_init(sim)
mu_env.Camera_set(viewer, model)
sim.model.eq_active[0] = True
for i in range(500):
    sim.step()
    viewer.render()

sim.data.mocap_pos[0] = ctrl_wrist_pos  # mocap控制需要用世界参考系
sim.data.mocap_quat[0] = ctrl_wrist_quat  # mocap控制需要用世界参考系
for i in range(200):
    sim.step()
    viewer.render()


interacting(hand_param)

