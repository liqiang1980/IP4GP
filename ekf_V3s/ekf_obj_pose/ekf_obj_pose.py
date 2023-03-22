import threading

import config_param
import robot_control
import mujoco_environment as mu_env
import ekf
import tactile_perception
import tactile_plotter
import numpy as np
from threading import Thread
import sys, termios, tty, os, time
import matplotlib.pyplot as plt
import forward_kinematics
from mujoco_py import load_model_from_path
from scipy.spatial.transform import Rotation
# sys.path.append('../')
import tactile_allegro_mujo_const as tac_const
import forward_kinematics
# from mujoco_py import load_model_from_path
import qgFunc as qg
# from scipy.spatial.transform import Rotation


class basicDataClass:
    def __init__(self, xml_path):
        # self.data_path = tac_const.txt_dir[6]
        # self.data_path = tac_const.txt_dir[3]
        self.data_path = tac_const.txt_dir[2]
        # self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[90:400]  # txt:2
        self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[60:]  # txt:6
        # self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[60:370]  # txt:3
        self.hand_pose = np.loadtxt(self.data_path + 'hand_pose.txt')  # hand position & quaternion(xyzw) in world
        # self.no_working_tac = {21: 'touch_0_4_9', 27: 'touch_0_4_8', 33: 'touch_0_4_7', 40: 'touch_0_5_6', 108: 'touch_2_1_1',
        #                        252: 'touch_9_1_1', 396: 'touch_13_1_1', 504: 'touch_16_1_1'}
        self.no_working_tac = {22: 'touch_0_5_9', 62: 'touch_0_3_2', 201: 'touch_7_4_3', 516: 'touch_16_3_1',
                               518: 'touch_16_3_3', 504: 'touch_16_1_1', 108: 'touch_2_1_1', 252: 'touch_9_1_1',
                               396: 'touch_13_1_1',}
        self.model = load_model_from_path(xml_path)

        """ Trans: hand to world """
        self.hand_world_posquat = self.hand_pose
        self.hand_world_posquat[:3] += np.array([0.01, 0.01, 0])
        self.hand_world_R = Rotation.from_quat(self.hand_world_posquat[3:]).as_matrix()
        self.hand_world_T = np.mat(np.eye(4))
        self.hand_world_T[:3, :3] = self.hand_world_R
        self.hand_world_T[:3, 3] = np.mat(self.hand_world_posquat[:3]).T
        self.world_hand_T = np.linalg.pinv(self.hand_world_T)

        """ Parameters for saving one data in temporarily """
        self.time_stamp = 0
        self.obj_palm_posrotvec = [0.0] * 6
        self.joint_pos = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        self.joint_vel = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        self.taxel_data = [0.0] * tac_const.TAC_TOTAL_NUM
        self.tac_tip_pos = {}


""" Parameters load """
hand_param, object_param, alg_param = config_param.pass_arg()
""" XML load """
xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand.xml"
if int(object_param[3]) == 1:
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_obj_frozen.xml"
elif int(object_param[3]) == 2:
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_obj_upsidedown.xml"
elif int(object_param[3]) == 3:
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_cylinder.xml"
elif int(object_param[3]) == 4:
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_cylinder_frozen.xml"

""" Instantiate basic data class """
basicData = basicDataClass(xml_path=xml_path)
""" Instantiate FK class """
fk = forward_kinematics.ForwardKinematics(hand_param=hand_param)
""" Instantiate ekf class """
grasping_ekf = ekf.EKF()
grasping_ekf.set_contact_flag(False)
grasping_ekf.set_store_flag(alg_param[0])
""" Instantiate tac-perception class """
tacperception = tactile_perception.cls_tactile_perception(xml_path=xml_path, fk=fk)
""" Instantiate robot class """
robctrl = robot_control.ROBCTRL(obj_param=object_param, hand_param=hand_param, model=basicData.model, xml_path=xml_path, fk=fk)

""" Tac_in_tip Initialization """
f_param = hand_param[1:]
print("Initializing...")
for f_part in f_param:
    f_name = f_part[0]
    tac_id = f_part[3]  # tac_id = [min, max]
    basicData.tac_tip_pos[f_name] = []
    for tid in range(tac_id[0], tac_id[1], 1):
        tac_name = basicData.model._sensor_id2name[tid]
        pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name, xml_path=xml_path)
        basicData.tac_tip_pos[f_name].append(pos_tac_tip)
print("                       All tac parts ready.")

first_contact_flag = False
for i, _data in enumerate(basicData.data_all):
    print("ROUND:", i)
    basicData.time_stamp = _data[0]
    obj_world_posquat = _data[1:8]  # posquat of object in world (xyzw)
    obj_world_R = Rotation.from_quat(obj_world_posquat[3:]).as_matrix()
    obj_palm_R = np.matmul(basicData.world_hand_T[:3, :3], obj_world_R)
    obj_palm_rotvec = Rotation.from_matrix(obj_palm_R).as_rotvec()
    obj_palm_pos = np.ravel(basicData.world_hand_T[:3, 3].T) + np.ravel(np.matmul(basicData.world_hand_T[:3, :3], obj_world_posquat[:3]))
    basicData.obj_palm_posrotvec[:3] = obj_palm_pos
    basicData.obj_palm_posrotvec[3:] = obj_palm_rotvec

    basicData.joint_pos = _data[8:24]  # joint position
    basicData.joint_vel = _data[24:40]  # joint vel
    # _taxel_data = _data[40:]  # tac data
    _taxel_data = _data[40: 40+540]  # tac data (no palm)
    basicData.taxel_data = (_taxel_data.astype('int32')).tolist()  # tac data
    for key in basicData.no_working_tac:
        basicData.taxel_data[key] = 0

    """EKF process"""
    if not first_contact_flag and (np.array(basicData.taxel_data) > 0.0).any():
        first_contact_flag = True
    if first_contact_flag:  # EKF Start
        print(robctrl.cnt_test, "EKF round:")
        robctrl.interaction(object_param=object_param,
                            ekf_grasping=grasping_ekf,
                            tacp=tacperception,
                            basicData=basicData)
