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
    def __init__(self, obj_param, hand_param, model):
        self.cnt_test = 0
        self.obj_param = obj_param
        self.f_param = hand_param[1:]
        self.f_size = len(self.f_param)
        print("f_size:", self.f_size)
        self.model = model
        self.FIRST_INTERACTION_FLAG = False
        self.robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
        # self.robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
        # self.robot = URDF.from_xml_file('../../robots/allegro_hand_tactile_v1.4.urdf')

        # first finger
        # self.kdl_kin_ff = KDLKinematics(self.robot, "palm_link", "link_3.0_tip")
        # # middle finger
        # self.kdl_kin_mf = KDLKinematics(self.robot, "palm_link", "link_7.0_tip")
        # # ring finger
        # self.kdl_kin_rf = KDLKinematics(self.robot, "palm_link", "link_11.0_tip")
        # # thumb
        # self.kdl_kin_th = KDLKinematics(self.robot, "palm_link", "link_15.0_tip")
        #
        # self.kdl_kin_taxel = KDLKinematics(self.robot, "palm_link", "touch_7_4_8")
        # self._ik_wdls_v_kdl = kdl.ChainIkSolverVel_pinv(self.chain)

        self.x_state = [0] * (6 + 3 * self.f_size)
        self.x_bar = [0] * (6 + 3 * self.f_size)
        self.P_state_cov = 0.1 * np.identity(6 + 3 * self.f_size)

        self.x_state_cur = [0, 0, 0, 0, 0, 0]
        self.gd_cur = [0, 0, 0, 0, 0, 0]
        # self.ct_g_z_position = [0, 0, 0]
        # self.ct_p_z_position = [0, 0, 0]
        # self.x_bar_all = [0, 0, 0, 0, 0, 0, 0]
        # self.x_state_all = [0, 0, 0, 0, 0, 0, 0]
        # self.x_gt_world = [0, 0, 0]
        # self.ju_all = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        """ Growing variable, Pre store data for plot """
        self.x_state_all = [0., 0., 0., 0., 0., 0., 0.]
        self.x_bar_all = [0., 0., 0., 0., 0., 0., 0.]
        self.gd_all = [0, 0, 0, 0, 0, 0, 0]

        if tacCONST.PN_FLAG == 'pn':
            # self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            #                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        # print(x, self.rotvec_cup_palm)
        self.quat_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_quat()  # xyzw
        self.R_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_matrix()
        self.T_cup_palm[:3, :3] = self.R_cup_palm
        self.T_cup_palm[:3, 3] = np.mat(self.pos_cup_palm).T

    def fk_offset(self, sim, finger_name, active_taxel_name, ref_frame='world'):
        if finger_name == 'ff':
            q = self.get_cur_jnt(sim)[0:4]
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_ff.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_3.0_tip", active_taxel_name)
        if finger_name == 'mf':
            q = self.get_cur_jnt(sim)[4:8]
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_mf.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_7.0_tip", active_taxel_name)
        if finger_name == 'rf':
            q = self.get_cur_jnt(sim)[8:12]
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_rf.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_11.0_tip", active_taxel_name)
        if finger_name == 'th':
            q = self.get_cur_jnt(sim)[12:16]
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_th.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_15.0_tip", active_taxel_name)
        pos_p_intip, pos_o_intip = ug.posquat2pos_p_o(pose_taxels_intip)

        position_taxel_inpalm = position_tip_inpalm + (np.matmul(orien_tip_inpalm, pos_p_intip)).transpose()
        orien_taxel_inpalm = np.matmul(orien_tip_inpalm, pos_o_intip)
        position_taxel_inworld, orien_taxel_inworld = ug.pose_trans_palm_to_world(sim, position_taxel_inpalm,
                                                                                  orien_taxel_inpalm)
        if ref_frame == 'palm':
            return position_taxel_inpalm, orien_taxel_inpalm
        else:
            return position_taxel_inworld, orien_taxel_inworld

    def get_cur_jnt(self, sim):
        # print("|||shape||||qpos: ", len(sim.data.qpos))

        cur_jnt = np.zeros(tacCONST.FULL_FINGER_JNTS_NUM)
        cur_jnt[0:4] = np.array([sim.data.qpos[tacCONST.FF_MEA_1],
                                 sim.data.qpos[tacCONST.FF_MEA_2],
                                 sim.data.qpos[tacCONST.FF_MEA_3],
                                 sim.data.qpos[tacCONST.FF_MEA_4]])

        cur_jnt[4:8] = np.array([sim.data.qpos[tacCONST.MF_MEA_1],
                                 sim.data.qpos[tacCONST.MF_MEA_2],
                                 sim.data.qpos[tacCONST.MF_MEA_3],
                                 sim.data.qpos[tacCONST.MF_MEA_4]])

        cur_jnt[8:12] = np.array([sim.data.qpos[tacCONST.RF_MEA_1],
                                  sim.data.qpos[tacCONST.RF_MEA_2],
                                  sim.data.qpos[tacCONST.RF_MEA_3],
                                  sim.data.qpos[tacCONST.RF_MEA_4]])

        cur_jnt[12:16] = np.array([sim.data.qpos[tacCONST.TH_MEA_1],
                                   sim.data.qpos[tacCONST.TH_MEA_2],
                                   sim.data.qpos[tacCONST.TH_MEA_3],
                                   sim.data.qpos[tacCONST.TH_MEA_4]])
        return cur_jnt

    def robjac_offset(self, sim, finger_name, q, taxel_name):
        if finger_name == 'ff':
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_ff.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_3.0_tip", taxel_name)
            pos_p_intip, pos_o_intip = ug.posquat2pos_p_o(pose_taxels_intip)
            position_taxel_inpalm = position_tip_inpalm + \
                                    (np.matmul(orien_tip_inpalm, pos_p_intip)).transpose()
            jac = self.kdl_kin_ff.jacobian(q, position_taxel_inpalm)
        if finger_name == 'mf':
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_mf.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_7.0_tip", taxel_name)
            pos_p_intip, pos_o_intip = ug.posquat2pos_p_o(pose_taxels_intip)
            position_taxel_inpalm = position_tip_inpalm + (np.matmul(orien_tip_inpalm, pos_p_intip)).transpose()
            jac = self.kdl_kin_mf.jacobian(q, position_taxel_inpalm)
        if finger_name == 'rf':
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_rf.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_11.0_tip", taxel_name)
            pos_p_intip, pos_o_intip = ug.posquat2pos_p_o(pose_taxels_intip)
            position_taxel_inpalm = position_tip_inpalm + (np.matmul(orien_tip_inpalm, pos_p_intip)).transpose()
            jac = self.kdl_kin_rf.jacobian(q, position_taxel_inpalm)
        if finger_name == 'th':
            position_tip_inpalm, orien_tip_inpalm = self.kdl_kin_th.FK(q)
            pose_taxels_intip = ug.get_relative_posquat(sim, "link_15.0_tip", taxel_name)
            pos_p_intip, pos_o_intip = ug.posquat2pos_p_o(pose_taxels_intip)
            position_taxel_inpalm = position_tip_inpalm + (np.matmul(orien_tip_inpalm, pos_p_intip)).transpose()
            jac = self.kdl_kin_th.jacobian(q, position_taxel_inpalm)
        # orien_taxel_inpalm = np.matmul(orien_tip_inpalm, pos_o_intip)
        return jac

    def active_fingers_taxels_render(self, sim, viewer, tacperception):
        self.active_finger_taxels_render(sim, viewer, 'ff', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'mf', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'rf', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'th', tacperception)
        print('........................................\n')

    def active_finger_taxels_render(self, sim, viewer, finger_name, tacperception):
        if tacperception.is_finger_contact(sim, finger_name):
            taxels_id = tacperception.get_contact_taxel_id_withoffset(sim, finger_name)
            taxels_pose_gt = []
            taxels_pose_fk = []
            print(finger_name + "viz taxels: ", end='')
            for i in taxels_id:
                active_taxel_name = sim.model._sensor_id2name[i]
                print(active_taxel_name + ' ', end='')
                # compute ground truth taxels
                taxel_pose_gt = taxel_pose()
                pose_taxels_w = ug.get_relative_posquat(sim, "world", active_taxel_name)
                pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_taxels_w)
                taxel_pose_gt.position = pos_p_world
                taxel_pose_gt.orien = pos_o_world
                taxels_pose_gt.append(taxel_pose_gt)
            print('')
            viz.active_taxels_visual(viewer, taxels_pose_gt, 'gt')

    def update_augmented_state(self, sim, idx, f_name, tacp, xstate):
        contact_name = tacp.cur_tac[f_name][0]
        self.pos_contact_cup[f_name] = np.ravel(ug.get_relative_posquat(sim, "cup",
                                                                        contact_name)[:3] + np.random.normal(0, 0.0,
                                                                                                             size=(
                                                                                                                 1, 3)))
        self.nv_contact_cup[f_name] = og.get_nv_contact_cup(obj_param=self.obj_param,
                                                            pos_contact_cup=self.pos_contact_cup[f_name])
        xstate[6 + 3 * idx] = self.pos_contact_cup[f_name][0]
        xstate[7 + 3 * idx] = self.pos_contact_cup[f_name][1]
        xstate[8 + 3 * idx] = self.pos_contact_cup[f_name][2]
        return xstate

    def augmented_state(self, sim, tacp, xstate):
        for i, f_part in enumerate(self.f_param):
            f_name = f_part[0]
            if tacp.is_finger_contact(sim=sim, model=self.model, f_part=f_part):
                # contact_name = tacp.get_contact_taxel_name(sim=sim, model=model, f_part=f_part, z_h_flag="h")
                contact_name = tacp.cur_tac[f_name][0]
                self.pos_contact_cup[f_name] = np.ravel(
                    ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(
                        0, 0.0, size=(1, 3)))
                self.nv_contact_cup[f_name] = og.get_nv_contact_cup(obj_param=self.obj_param,
                                                                    pos_contact_cup=self.pos_contact_cup[f_name])
                # xstate = np.append(xstate, [self.pos_contact_cup[f_name]])
                xstate[6+i*3: 6+i*3+3] = self.pos_contact_cup[f_name]
            # else:
            #     xstate = np.append(xstate, [0, 0, 0])
        # print("              First augmented_state:", xstate)
        return xstate

    def interaction(self, sim, model, viewer, object_param, alg_param, ekf_grasping, tacp, fk, char):
        # global first_contact_flag, x_all, gd_all, P_state_cov, x_state, last_angles, x_bar, z_t, h_t
        # f_num = 0  # The number of contact finger parts
        """ Update Joint state and FK """
        fk.fk_update_all(sim=sim)
        """ 
        Update contact state:
        1. contact_flags
        2. number of contact finger parts
        3. cur_contact_tac state 
        """
        # for f_part in self.f_param:
        #     tacp.is_finger_contact(sim=sim, model=model, f_part=f_part)
        tacp.is_fingers_contact(sim=sim, model=model, f_param=self.f_param)

        """ Update gd_state """
        gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")  # wxyz
        gd_state = ug.posquat2posrotvec_hacking(gd_posquat)  # wxyz to xyzw
        # print("gd:  ", gd_state)

        """ First interaction, Do Initialization """
        # print("At beginning, xstate: ", self.x_state)
        if not self.FIRST_INTERACTION_FLAG:
            self.FIRST_INTERACTION_FLAG = True
            """ x_state Initialization """
            # _x_state = ug.get_relative_posquat(sim, "palm_link", "cup")  # get wxyz
            # self.x_state[:6] = ug.pos_quat2axis_angle(_x_state)  # wxyz to xyzw
            self.x_state[:6] = gd_state
            np.set_printoptions(suppress=True)
            print('x_state from beginning before add noise', self.x_state)
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
            self.x_state = self.augmented_state(sim=sim, tacp=tacp, xstate=self.x_state)

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
        for idx, f_part in enumerate(self.f_param):
            f_name = f_part[0]
            if tacp.is_contact[f_name] and not tacp.is_first_contact[f_name]:
                self.x_state = self.update_augmented_state(sim=sim, idx=idx, f_name=f_name,
                                                           tacp=tacp, xstate=self.x_state)
                tacp.is_first_contact[f_name] = True

        """ EKF Forward prediction """
        self.x_bar, P_state_cov = ekf_grasping.state_predictor(xstate=self.x_state,
                                                               P_state_cov=self.P_state_cov,
                                                               tacp=tacp,
                                                               robctrl=self)
        # self.x_bar = deepcopy(_x_bar)
        # Mathematical components update by result of Forward prediction for Posteriori estimation
        self.Xcomponents_update(x=self.x_bar[:6])

        """ h_t & z_t updates """
        h_t_position, h_t_nv = ekf_grasping.observe_computation(tacp=tacp, robctrl=self, sim=sim)
        z_t_position, z_t_nv = ekf_grasping.measure_fb(tacp=tacp, robctrl=self)
        if tacCONST.PN_FLAG == 'p':
            self.z_t = np.ravel(z_t_position)
            self.h_t = np.ravel(h_t_position)
        else:
            self.z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
            self.h_t = np.concatenate((h_t_position, h_t_nv), axis=None)

        """ EKF Posteriori estimation """
        if tacCONST.posteriori_FLAG:
            # self.x_state, self.P_state_cov = ekf_grasping.ekf_posteriori(x_bar=self.x_bar,
            #                                                              z_t=self.z_t,
            #                                                              h_t=self.h_t,
            #                                                              P_state_cov=P_state_cov,
            #                                                              tacp=tacp,
            #                                                              robctrl=self)
            x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, self.x_bar, self.z_t, self.h_t,
                                                               P_state_cov, tacp)
        else:
            self.x_state = self.x_bar

        # delta_t = z_t - h_t
        # self.delta_ct = np.vstack((self.delta_ct, delta_t))
        # self.z_t = np.vstack((self.z_t, z_t))
        # self.h_t = np.vstack((self.h_t, h_t))
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
        # np.savetxt('x_gt_world.txt', self.x_gt_world)
        np.savetxt('x_bar_all.txt', self.x_bar_all)
        np.savetxt('x_state_all.txt', self.x_state_all)
        np.savetxt('x_gt_palm.txt', self.gd_all)
        # np.savetxt('ju_all.txt', self.ju_all)
        # np.savetxt('delta_ct.txt', self.delta_ct)
        # np.savetxt('z_t.txt', self.z_t)
        # np.savetxt('h_t.txt', self.h_t)
        """ Last Update:
        Update last_state by cur_state.
        Update mathematical components.
        """
        tacp.Last_tac_renew(f_param=self.f_param)
        self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components update for next EKF round
        print(".........................................")
        # self.cnt_test += 1

        if self.FIRST_INTERACTION_FLAG:
            viz.vis_state_contact(sim=sim, viewer=viewer, tacp=tacp,
                                  z_t=self.z_t, h_t=self.h_t,
                                  robctrl=self,
                                  char=char,
                                  fk=fk)
            # self.active_fingers_taxels_render(sim, viewer, tacp)
            tacp.fin_num = 0
            # tacp.fin_tri = np.zeros(4)
            # tacp.fin_tri = np.zeros(len(hand_param) - 1)

# first_contact_flag = False
# ff_first_contact_flag = False
# mf_first_contact_flag = False
# rf_first_contact_flag = False
# th_first_contact_flag = False
