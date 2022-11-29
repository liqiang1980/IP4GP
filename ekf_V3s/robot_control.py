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


class IK_type(Enum):
    IK_V_POSITION_ONLY = 1
    IK_V_FULL = 2
    IK_V_WDLS = 3


class ROBCTRL:
    def __init__(self):
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

        self.x_state = [0, 0, 0, 0, 0, 0]
        self.x_state_aug = [0] * (6 + 3 * 12)
        self.x_bar = [0] * (6 + 3 * 12)
        self.P_state_cov = 0.1 * np.identity(6 + 4 * 3)

        # self.gd_cur = [0, 0, 0, 0, 0, 0]
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

        # if tacCONST.PN_FLAG == 'pn':
        #     # self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #     #                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # else:
        #     # self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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

    def update_augmented_state(self, sim, model, f_param, tacp, xstate):
        for idx, f_part in enumerate(f_param):
            if tacp.is_finger_contact(sim, f_part):
                contact_name = tacp.get_contact_taxel_name(sim=sim, model=model, f_part=f_part,
                                                                    z_h_flag="h")
                pos_contact = ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(0, 0.0,
                                                                                                       size=(1, 3))
                xstate[6 + 3 * idx] = pos_contact[0][0]
                xstate[7 + 3 * idx] = pos_contact[0][1]
                xstate[8 + 3 * idx] = pos_contact[0][2]
        return xstate

    def augmented_state(self, sim, model, f_param, tacp, xstate):
        for f_part in f_param:
            if tacp.is_finger_contact(sim=sim, f_part=f_part):
                contact_name = tacp.get_contact_taxel_name(sim=sim, model=model, f_part=f_part,
                                                                    z_h_flag="h")
                pos_contact = ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(0, 0.0,
                                                                                                       size=(1, 3))
                xstate = np.append(xstate, [pos_contact])
            else:
                xstate = np.append(xstate, [0, 0, 0])
        return xstate

    def interaction(self, sim, model, viewer, hand_param, object_param, alg_param, ekf_grasping, tacp, fk, char):
        # global first_contact_flag, x_all, gd_all, P_state_cov, x_state, last_angles, x_bar, z_t, h_t

        f_param = hand_param[1:]
        f_num = 0  # The number of contact finger parts
        """ Update Joint state and FK """
        fk.fk_update_all(sim=sim)
        """ 
        Update contact state:
        1. contact_flags
        2. number of contact finger parts
        3. cur_contact_tac state 
        """
        for f_part in f_param:
            if tacp.is_finger_contact(sim=sim, model=model, f_part=f_part, fk=fk):
                f_num += 1  # The number of contact finger parts
        """ Update gd_state """
        gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
        gd_state = ug.posquat2posrotvec_hacking(gd_posquat)

        """ If any contact_tac exists, Do interaction """
        if any(list(tacp.is_contact.values())):
            """ First interaction, Do Initialization """
            if not self.FIRST_INTERACTION_FLAG:
                self.FIRST_INTERACTION_FLAG = True
                """ x_state Initialization """
                x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                self.x_state = ug.pos_quat2axis_angle(x_state)
                np.set_printoptions(suppress=True)
                print('x_state from beginning before add noise', self.x_state)
                if tacCONST.initE_FLAG:
                    init_e = np.hstack(
                        (np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), (1, 3)),
                         np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))
                    self.x_state = self.x_state + init_e

                """ 
                Augmented state Initialization.
                augmented state with the contact position on the object surface described in the object frame 
                """
                self.x_state_aug = self.augmented_state(sim=sim, model=model, f_param=f_param,
                                                        tacp=tacp, xstate=self.x_state)

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
                # first_contact_flag = True
                for f in tacp.is_contact:
                    if tacp.is_contact[f]:
                        tacp.is_first_contact[f] = True
                """ Initialization Don """
                print('\nInitialization done.\n')
                return

            """ Detect new contact tacs that have never been touched before """
            for f in tacp.is_contact:
                if tacp.is_contact[f] and not tacp.is_first_contact[f]:
                    self.x_state_aug = self.update_augmented_state(sim=sim, model=model, f_param=f_param,
                                                                   tacp=tacp, xstate=self.x_state_aug)
                    tacp.is_first_contact[f] = True

            """ EKF Forward prediction """
            self.x_bar, P_state_cov = ekf_grasping.state_predictor(sim=sim, model=model,
                                                                   hand_param=hand_param,
                                                                   object_param=object_param,
                                                                   xstate_aug=self.x_state_aug,
                                                                   P_state_cov=self.P_state_cov,
                                                                   tacp=tacp,
                                                                   robctrl=self)

            """ h_t & z_t updates """
            h_t_position, h_t_nv = ekf_grasping.observe_computation(self.x_bar, tacp, sim, object_param)
            z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, self.x_bar,
                                                           tacp)
            if tacCONST.PN_FLAG == 'p':
                z_t = np.ravel(z_t_position)
                h_t = np.ravel(h_t_position)
            else:
                z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
                h_t = np.concatenate((h_t_position, h_t_nv), axis=None)

            """ EKF posteriori estimation """
            if tacCONST.posteriori_FLAG:
                self.x_state, self.P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer,
                                                                             self.x_bar, z_t, h_t, P_state_cov,
                                                                             tacp, object_param)
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
            np.savetxt('x_gt_world.txt', self.x_gt_world)
            np.savetxt('x_bar_all.txt', self.x_bar_all)
            np.savetxt('x_state_all.txt', self.x_state_all)
            np.savetxt('x_gt_palm.txt', self.gd_all)
            # np.savetxt('ju_all.txt', self.ju_all)
            # np.savetxt('delta_ct.txt', self.delta_ct)
            # np.savetxt('z_t.txt', self.z_t)
            # np.savetxt('h_t.txt', self.h_t)

        if self.FIRST_INTERACTION_FLAG:
            viz.vis_state_contact(sim, viewer, tacp, z_t, h_t, self.x_bar, self.x_state, char, object_param)
            self.active_fingers_taxels_render(sim, viewer, tacp)
            tacp.fin_num = 0
            # tacp.fin_tri = np.zeros(4)
            # tacp.fin_tri = np.zeros(len(hand_param) - 1)

        """ Last Step: use cur_state to update last_state """
        tacp.tac_update_cur2last(f_param=f_param)

# first_contact_flag = False
# ff_first_contact_flag = False
# mf_first_contact_flag = False
# rf_first_contact_flag = False
# th_first_contact_flag = False
