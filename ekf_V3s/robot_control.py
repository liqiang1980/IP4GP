import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import viz


def robot_init(sim):
    sim.data.ctrl[0] = 0.8
    sim.data.ctrl[1] = -0.78
    sim.data.ctrl[2] = 1.13
    sim.data.ctrl[3] = -1.
    sim.data.ctrl[4] = 0
    sim.data.ctrl[5] = -0.3


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


def middle_finger_vel(sim, input1, input2):
    print(sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1])
    if not (np.array(sim.data.sensordata[144:216]) > 0.0).any():  # 中指
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
    else:
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2


def ring_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[288:360]) > 0.0).any():  # 小拇指
        sim.data.ctrl[15] = sim.data.ctrl[15] + input1
        sim.data.ctrl[16] = sim.data.ctrl[16] + input1
        sim.data.ctrl[17] = sim.data.ctrl[17] + input1
    else:
        sim.data.ctrl[15] = sim.data.ctrl[15] + input2
        sim.data.ctrl[16] = sim.data.ctrl[16] + input2
        sim.data.ctrl[17] = sim.data.ctrl[17] + input2


def pre_thumb(sim, viewer):
    for _ in range(1000):
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

def config_robot():
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
    return kdl_kin0, kdl_kin1, kdl_kin2, kdl_kin3, kdl_tree

def interaction(sim, model, viewer, hand_param, ekf_grasping):
    err_all = np.loadtxt("./err_inHand_v3bi.txt")
    pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
    for ii in range(1000):
        if hand_param[1][1] == '1':
            index_finger(sim, 0.0055, 0.00001)
        if hand_param[2][1] == '1':
            middle_finger(sim, 0.0016, 0.00001)
        if hand_param[3][1] == '1':
            ring_finger(sim, 0.002, 0.00001)
        if hand_param[4][1] == '1':
            thumb(sim, 0.0003, 0.00001)
        # EKF()
        if not np.all(sim.data.sensordata == 0):
            viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
        viewer.render()