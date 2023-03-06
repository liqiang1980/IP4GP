import numpy as np
import tactile_allegro_mujo_const as tacCONST
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_kinematics import joint_kdl_to_list
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import util_geometry as ug
import time
import fcl
import math
import viz
import PyKDL as kdl
from mujoco_py import const
from tactile_perception import taxel_pose
from enum import Enum
from sig_filter import lfilter
from scipy.spatial.transform import Rotation
import object_geometry as og
from copy import deepcopy


class IK_type(Enum):
    IK_V_POSITION_ONLY = 1
    IK_V_FULL = 2
    IK_V_WDLS = 3


class ROBCTRL:
    def __init__(self, obj_param, hand_param, model, xml_path, fk):
        self.cnt_test = 0
        self.obj_param = obj_param
        self.f_param = hand_param[1:]
        self.f_size = len(self.f_param)
        print("f_size:", self.f_size)
        self.model = model
        self.xml_path = xml_path
        self.fk = fk
        self.FIRST_INTERACTION_FLAG = False
        self.robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')

        self.x_state = [0] * (6 + 3 * self.f_size)
        self.x_bar = [0] * (6 + 3 * self.f_size)
        self.P_state_cov = 0.1 * np.identity(6 + 3 * self.f_size)

        self.x_state_cur = [0, 0, 0, 0, 0, 0]
        self.gd_cur = [0, 0, 0, 0, 0, 0]

        """ Growing variable, Pre store data for plot """
        self.x_state_all = [0., 0., 0., 0., 0., 0., 0.]
        self.x_bar_all = [0., 0., 0., 0., 0., 0., 0.]
        self.gd_all = [0, 0, 0, 0, 0, 0, 0]

        if tacCONST.PN_FLAG == 'pn':
            self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """ Mathematical components """
        self.pos_contact_cup = {"ff": np.zeros(3), "ffd": np.zeros(3), "ffq": np.zeros(3),
                                "mf": np.zeros(3), "mfd": np.zeros(3), "mfq": np.zeros(3),
                                "rf": np.zeros(3), "rfd": np.zeros(3), "rfq": np.zeros(3),
                                "th": np.zeros(3), "thd": np.zeros(3), "palm": np.zeros(3)}
        self.nv_contact_cup = {"ff": np.zeros(3), "ffd": np.zeros(3), "ffq": np.zeros(3),
                               "mf": np.zeros(3), "mfd": np.zeros(3), "mfq": np.zeros(3),
                               "rf": np.zeros(3), "rfd": np.zeros(3), "rfq": np.zeros(3),
                               "th": np.zeros(3), "thd": np.zeros(3), "palm": np.zeros(3)}
        self.pos_cup_palm = [0] * 3
        self.rotvec_cup_palm = [0] * 3
        self.quat_cup_palm = [0] * 3
        self.R_cup_palm = np.mat(np.eye(3))
        self.T_cup_palm = np.mat(np.eye(4))

    def Xcomponents_update(self, x):
        self.pos_cup_palm = x[:3]
        self.rotvec_cup_palm = x[3:]
        self.quat_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_quat()  # xyzw
        self.R_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_matrix()
        self.T_cup_palm[:3, :3] = self.R_cup_palm
        self.T_cup_palm[:3, 3] = np.mat(self.pos_cup_palm).T

    def update_augmented_state(self, idx, f_name, tacp, xstate):
        contact_name = tacp.cur_tac[f_name][0]
        # self.pos_contact_cup[f_name] = np.ravel(ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(0, 0.0,size=(1, 3)))
        pos_tac_palm, rotvec_tac_palm, T_contact_palm = self.fk.get_relative_posrot(tac_name=contact_name,
                                                                                    f_name=f_name,
                                                                                    xml_path=self.xml_path)
        T_contact_cup = np.matmul(np.linalg.pinv(self.T_cup_palm), T_contact_palm)
        self.pos_contact_cup[f_name] = np.ravel(np.ravel(T_contact_cup[:3, 3].T))
        self.nv_contact_cup[f_name] = og.get_nv_contact_cup(obj_param=self.obj_param,
                                                            pos_contact_cup=self.pos_contact_cup[f_name])
        xstate[6 + 3 * idx] = self.pos_contact_cup[f_name][0]
        xstate[7 + 3 * idx] = self.pos_contact_cup[f_name][1]
        xstate[8 + 3 * idx] = self.pos_contact_cup[f_name][2]
        return xstate

    def augmented_state(self, basicData, tacp, xstate):
        for i, f_part in enumerate(self.f_param):
            f_name = f_part[0]
            if tacp.is_contact[f_name]:
                contact_name = tacp.cur_tac[f_name][0]
                pos_tac_palm, rotvec_tac_palm, T_contact_palm = self.fk.get_relative_posrot(tac_name=contact_name,
                                                                                            f_name=f_name,
                                                                                            xml_path=self.xml_path)
                T_contact_cup = np.matmul(np.linalg.pinv(self.T_cup_palm), T_contact_palm)
                # self.pos_contact_cup[f_name] = np.ravel(
                #     ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(
                #         0, 0.0, size=(1, 3)))
                self.pos_contact_cup[f_name] = np.ravel(np.ravel(T_contact_cup[:3, 3].T))
                self.nv_contact_cup[f_name] = og.get_nv_contact_cup(obj_param=self.obj_param,
                                                                    pos_contact_cup=self.pos_contact_cup[f_name])
                xstate[6 + i * 3: 6 + i * 3 + 3] = self.pos_contact_cup[f_name]
        return xstate

    def interaction(self, object_param, ekf_grasping, tacp, basicData):
        """
        Do one EKF prediction round
        """
        print("Round ", self.cnt_test, "......................")
        self.cnt_test += 1
        """ Update Joint state and FK """
        self.fk.fk_update_all(basicData=basicData)
        """ 
        Update contact state:
        1. contact_flags
        2. number of contact finger parts
        3. cur_contact_tac state 
        """
        # print(">>>>time 0")
        tacp.is_fingers_contact(basicData=basicData, f_param=self.f_param, fk=self.fk)
        # print(">>>>time 1")

        """ Update gd_state """
        gd_state = basicData.obj_palm_posrotvec
        # print("gd:  ", gd_state)

        """ First interaction, Do Initialization """
        # print("At beginning, xstate: ", self.x_state)
        if not self.FIRST_INTERACTION_FLAG:
            self.FIRST_INTERACTION_FLAG = True
            """ x_state Initialization """
            self.x_state[:6] = gd_state
            np.set_printoptions(suppress=True)
            # print('x_state from beginning before add noise', self.x_state)
            if tacCONST.initE_FLAG:
                init_e = np.hstack(
                    (np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), (1, 3)),
                     np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))
                self.x_state[:6] = np.ravel(self.x_state[:6] + init_e)
            self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components initialization

            """ 
            Augmented state Initialization.
            augmented state with the contact position on the object surface described in the object frame 
            """
            self.x_state = self.augmented_state(basicData=basicData, tacp=tacp, xstate=self.x_state)
            # print("  x_state check: ", self.x_state, "\n", np.ravel(self.x_state)[3:6])

            """ Init the data for plot """
            x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
            x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
            gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
            x_state_plot[0:3] = np.ravel(self.x_state)[0:3]
            x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(np.ravel(self.x_state)[3:6])
            x_bar_plot[0:3] = np.ravel(self.x_state)[0:3]
            x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(np.ravel(self.x_state)[3:6])
            gd_state_plot[0:3] = gd_state[0:3]
            gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])
            self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
            self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
            self.gd_all = np.vstack((self.gd_all, gd_state_plot))

            """Set first contact flags for finger parts"""
            for f_part in self.f_param:
                f_name = f_part[0]
                if tacp.is_contact[f_name]:
                    tacp.is_first_contact[f_name] = True
            """ Initialization Done """
            tacp.Last_tac_renew(f_param=self.f_param)
            self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components update for next EKF round
            print('\n...................Initialization done...................\n')
            return

        """ Detect new contact tacs that have never been touched before """
        if self.cnt_test < 10:
            for idx, f_part in enumerate(self.f_param):
                f_name = f_part[0]
                if tacp.is_contact[f_name] and not tacp.is_first_contact[f_name]:
                    self.x_state = self.update_augmented_state(idx=idx, f_name=f_name,
                                                               tacp=tacp, xstate=self.x_state)
                    tacp.is_first_contact[f_name] = True
        """ If contact, always contact """
        tacp.is_contact = deepcopy(tacp.is_first_contact)  # This code overrides the previous renew of tacp.is_contact
        # print("contact:", tacp.is_contact)
        # print("is_first_contact:", tacp.is_first_contact)

        """ EKF Forward prediction """
        self.x_bar, P_state_cov = ekf_grasping.state_predictor(xstate=self.x_state,
                                                               P_state_cov=self.P_state_cov,
                                                               tacp=tacp,
                                                               robctrl=self)
        # self.x_bar = deepcopy(_x_bar)
        # Mathematical components update by result of Forward prediction for Posteriori estimation
        self.Xcomponents_update(x=self.x_bar[:6])

        """ h_t & z_t updates """
        h_t_position, h_t_nv = ekf_grasping.observe_computation(tacp=tacp, robctrl=self)
        z_t_position, z_t_nv = ekf_grasping.measure_fb(tacp=tacp, robctrl=self)
        if tacCONST.PN_FLAG == 'p':
            self.z_t = np.ravel(z_t_position)
            self.h_t = np.ravel(h_t_position)
        else:
            self.z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
            self.h_t = np.concatenate((h_t_position, h_t_nv), axis=None)

        """ EKF Posteriori estimation """
        if tacCONST.posteriori_FLAG:
            self.x_state, self.P_state_cov = ekf_grasping.ekf_posteriori(x_bar=self.x_bar,
                                                                         z_t=self.z_t,
                                                                         h_t=self.h_t,
                                                                         P_state_cov=P_state_cov,
                                                                         tacp=tacp,
                                                                         robctrl=self)
        else:
            self.x_state = self.x_bar

        """ Update the plot_data """
        x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
        x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
        gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
        x_state_plot[0:3] = self.x_state[0:3]
        x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(self.x_state[3:6])
        x_bar_plot[0:3] = self.x_bar[0:3]
        x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(self.x_bar[3:6])
        gd_state_plot[0:3] = gd_state[0:3]
        gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])
        self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
        self.gd_all = np.vstack((self.gd_all, gd_state_plot))
        self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
        #
        np.savetxt('offline_x_bar_all.txt', self.x_bar_all)
        np.savetxt('offline_x_state_all.txt', self.x_state_all)
        np.savetxt('offline_x_gt_palm.txt', self.gd_all)
        """ Last Update:
        Update last_state by cur_state.
        Update mathematical components.
        """
        tacp.Last_tac_renew(f_param=self.f_param)
        self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components update for next EKF round
        print(".........................................")
