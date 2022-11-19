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
        self.robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
        # self.robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
        # self.robot = URDF.from_xml_file('../../robots/allegro_hand_tactile_v1.4.urdf')
        # first finger
        self.kdl_kin_ff = KDLKinematics(self.robot, "palm_link", "link_3.0_tip")
        # middle finger
        self.kdl_kin_mf = KDLKinematics(self.robot, "palm_link", "link_7.0_tip")
        # ring finger
        self.kdl_kin_rf = KDLKinematics(self.robot, "palm_link", "link_11.0_tip")
        # thumb
        self.kdl_kin_th = KDLKinematics(self.robot, "palm_link", "link_15.0_tip")

        self.kdl_kin_taxel = KDLKinematics(self.robot, "palm_link", "touch_7_4_8")
        # self._ik_wdls_v_kdl = kdl.ChainIkSolverVel_pinv(self.chain)

        self.x_state_cur = [0, 0, 0, 0, 0, 0]
        self.gd_cur = [0, 0, 0, 0, 0, 0]
        self.ct_g_z_position = [0, 0, 0]
        self.ct_p_z_position = [0, 0, 0]
        self.x_bar_all = [0, 0, 0, 0, 0, 0, 0]
        self.x_state_all = [0, 0, 0, 0, 0, 0, 0]
        self.x_gt_palm = [0, 0, 0, 0, 0, 0, 0]
        self.x_gt_world = [0, 0, 0]
        self.ju_all = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tacCONST.PN_FLAG == 'pn':
            self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tacCONST.PN_FLAG == 'pn':
            z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tacCONST.PN_FLAG == 'pn':
            h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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

    def update_augmented_state(self, sim, model, f_param, tacperception, xstate):
        for idx, f_part in enumerate(f_param):
            f_name = f_part[0]
            if tacperception.is_finger_contact(sim, f_part):
                contact_name = tacperception.get_contact_taxel_name(sim=sim, model=model, f_part=f_part,
                                                                    z_h_flag="h")
                pos_contact = ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(0, 0.0,
                                                                                                       size=(1, 3))
                xstate[6 + 3 * idx] = pos_contact[0][0]
                xstate[7 + 3 * idx] = pos_contact[0][1]
                xstate[8 + 3 * idx] = pos_contact[0][2]
        return xstate

        # if tacperception.is_finger_contact(sim, hand_param[1][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state[6] = pos_contact0[0][0]
        #     x_state[7] = pos_contact0[0][1]
        #     x_state[8] = pos_contact0[0][2]
        #
        # if tacperception.is_finger_contact(sim, hand_param[2][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state[9] = pos_contact0[0][0]
        #     x_state[10] = pos_contact0[0][1]
        #     x_state[11] = pos_contact0[0][2]
        #
        # if tacperception.is_finger_contact(sim, hand_param[3][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state[12] = pos_contact0[0][0]
        #     x_state[13] = pos_contact0[0][1]
        #     x_state[14] = pos_contact0[0][2]
        #
        # if tacperception.is_finger_contact(sim, hand_param[4][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     # print('x_state ', x_state)
        #     x_state[15] = pos_contact0[0][0]
        #     x_state[16] = pos_contact0[0][1]
        #     x_state[17] = pos_contact0[0][2]
        # return x_state

    def augmented_state(self, sim, model, f_param, tacperception, xstate):
        for f_part in f_param:
            f_name = f_part[0]
            if tacperception.is_finger_contact(sim=sim, hand_param_part=f_part):
                contact_name = tacperception.get_contact_taxel_name(sim=sim, model=model, f_part=f_part,
                                                                    z_h_flag="h")
                pos_contact = ug.get_relative_posquat(sim, "cup", contact_name)[:3] + np.random.normal(0, 0.0,
                                                                                                       size=(1, 3))
                xstate = np.append(xstate, [pos_contact])
            else:
                xstate = np.append(xstate, [0, 0, 0])
        return xstate

        # if tacperception.is_finger_contact(sim, hand_param[1][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state = np.append(x_state, [pos_contact0])
        # else:
        #     x_state = np.append(x_state, [0, 0, 0])
        #
        # if tacperception.is_finger_contact(sim, hand_param[2][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state = np.append(x_state, [pos_contact0])
        # else:
        #     x_state = np.append(x_state, [0, 0, 0])
        #
        # if tacperception.is_finger_contact(sim, hand_param[3][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state = np.append(x_state, [pos_contact0])
        # else:
        #     x_state = np.append(x_state, [0, 0, 0])
        #
        # if tacperception.is_finger_contact(sim, hand_param[4][0]):
        #     c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0], z_h_flag="h")
        #     pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0,
        #                                                                                              size=(1, 3))
        #     x_state = np.append(x_state, [pos_contact0])
        # else:
        #     x_state = np.append(x_state, [0, 0, 0])
        # return xstate

    def interaction(self, sim, model, viewer, hand_param, object_param, alg_param, ekf_grasping, tacperception, char):
        global first_contact_flag, x_all, gd_all, ff_first_contact_flag, \
            mf_first_contact_flag, rf_first_contact_flag, th_first_contact_flag, \
            P_state_cov, x_state, last_angles, x_bar, z_t, h_t

        # flag_ff = tacperception.is_finger_contact(sim, hand_param[1][0])
        # flag_mf = tacperception.is_finger_contact(sim, hand_param[2][0])
        # flag_rf = tacperception.is_finger_contact(sim, hand_param[3][0])
        # flag_th = tacperception.is_finger_contact(sim, hand_param[4][0])
        # contact_flags = [False] * (len(hand_param) - 1)  # Record which f_part is contact in current interaction round
        # for i in range(len(hand_param) - 1):
        #     contact_flags[i] = tacperception.is_finger_contact(sim, hand_param[i + 1])

        f_param = hand_param[1:]
        """Update contact"""
        for f_part in f_param:
            tacperception.is_finger_contact(sim=sim, hand_param_part=f_part)

        # if flag_ff or flag_mf or flag_rf or flag_th:
        # if any(contact_flags):
        if any(list(tacperception.is_contact.values())):
            # f_param = hand_param[1:]
            for idx, f_part in enumerate(f_param):
                f_name = f_part[0]
                if tacperception.is_contact[f_name]:
                    tacperception.fin_num += 1
                    tacperception.fin_tri[idx] = 1
            # if tacperception.is_ff_contact:
            #     tacperception.fin_num += 1
            #     tacperception.fin_tri[0] = 1
            # if tacperception.is_mf_contact == True:
            #     tacperception.fin_num += 1
            #     tacperception.fin_tri[1] = 1
            # if tacperception.is_rf_contact == True:
            #     tacperception.fin_num += 1
            #     tacperception.fin_tri[2] = 1
            # if tacperception.is_th_contact == True:
            #     tacperception.fin_num += 1
            #     tacperception.fin_tri[3] = 1

            # print('contacts num ', tacperception.fin_num)
            # print('contacts id ', tacperception.fin_tri)
            # detect the first contact and initialize y_t_update with noise
            # print('get into contact procedure')
            if not first_contact_flag:
                # initialize the co-variance matrix of state estimation
                # P_state_cov = np.random.normal(0, 0.01) * np.identity(6 + 4 * 3)
                # P_state_cov = 0.1 * np.identity(6 + 4 * 3)
                # noise +-5 mm, +-0.002 (axis angle vector)
                # prepare object pose and relevant noise
                init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]),
                                                      (1, 3)),
                                    np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))

                x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                # attention, here orientation we use the axis angle representation.
                # x_state = np.array([ug.pos_quat2axis_angle(x_state)])
                x_state = ug.pos_quat2axis_angle(x_state)
                np.set_printoptions(suppress=True)
                print('x_state from beginning before add noise', x_state)
                if tacCONST.initE_FLAG:
                    x_state = x_state + init_e
                x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
                x_state_plot[0:3] = np.ravel(x_state)[0:3]
                x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(np.ravel(x_state)[3:6])
                self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
                x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
                x_bar_plot[0:3] = np.ravel(x_state)[0:3]
                x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(np.ravel(x_state)[3:6])

                self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))

                # augmented state with the contact position on the object surface described in the object frame
                x_state = self.augmented_state(sim=sim, model=model, f_param=f_param, tacperception=tacperception,
                                               xstate=x_state)
                x_all = x_state

                gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
                gd_state = ug.posquat2posrotvec_hacking(gd_posquat)

                gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
                gd_state_plot[0:3] = gd_state[0:3]
                gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])

                self.x_gt_palm = np.vstack((self.x_gt_palm, gd_state_plot))

                # print('x_state ground truth ', gd_state)
                # gd_state = qg.posquat2posrotvec(gd_posquat)
                first_contact_flag = True
                for f in tacperception.is_contact:
                    if tacperception.is_contact[f]:
                        tacperception.is_first_contact[f] = True
                #
                # if tacperception.is_ff_contact:
                #     ff_first_contact_flag = True
                # if tacperception.is_mf_contact:
                #     mf_first_contact_flag = True
                # if tacperception.is_rf_contact:
                #     rf_first_contact_flag = True
                # if tacperception.is_th_contact:
                #     th_first_contact_flag = True

                """wave filter of jnt angles"""
                last_angles = self.get_cur_jnt(sim)
                # self.mea_filter_js = lfilter(9, 0.01, last_angles, 16)
                self.mea_filter_js = last_angles
                print('return early')
                return

            else:
                for f in tacperception.is_contact:
                    if tacperception.is_contact[f] and not tacperception.is_first_contact[f]:
                        x_state = self.update_augmented_state(sim=sim, model=model, f_param=f_param,
                                                              tacperception=tacperception, xstate=x_state)
                        tacperception.is_first_contact[f] = True

            # elif ((flag_ff == True) and (ff_first_contact_flag) == False):
            #     x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
            #     ff_first_contact_flag = True
            # elif ((flag_mf == True) and (mf_first_contact_flag) == False):
            #     x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
            #     mf_first_contact_flag = True
            # elif ((flag_rf == True) and (rf_first_contact_flag) == False):
            #     x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
            #     rf_first_contact_flag = True
            # elif ((flag_th == True) and (th_first_contact_flag) == False):
            #     x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
            #     th_first_contact_flag = True

            # else:
            #     print('no else')

            # print('P_state_cov ', P_state_cov)
            # x_state = np.ravel(x_state)
            gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
            gd_state = ug.posquat2posrotvec_hacking(gd_posquat)

            """ Prediction step in EKF """
            # todo can not use ground truth update the state at every step
            # x_state[:6] = gd_state
            cur_angles_tmp = self.get_cur_jnt(sim)
            """ do a rolling average """
            cur_angles = cur_angles_tmp
            # cur_angles, self.mea_filter_js.z = \
            #     self.mea_filter_js.lp_filter(cur_angles_tmp, 16)  # wave filter of jnt angles
            x_bar, P_state_cov, ju_all = \
                ekf_grasping.state_predictor(sim, model, hand_param, object_param,
                                             x_state, tacperception, P_state_cov, cur_angles,
                                             last_angles, self)

            last_angles = cur_angles
            # last_angles = cur_angles_tmp
            """
            Compute the axis and angle for plot_data
            """
            x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
            x_bar_plot[0:3] = x_bar[0:3]
            x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(x_bar[3:6])

            gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
            gd_state_plot[0:3] = gd_state[0:3]
            gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])
            self.gd_cur = gd_state_plot

            self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
            self.x_gt_palm = np.vstack((self.x_gt_palm, gd_state_plot))
            self.ju_all = np.vstack((self.ju_all, ju_all[6:12]))

            #
            h_t_position, h_t_nv = ekf_grasping.observe_computation(x_bar, tacperception, sim, object_param)
            #
            z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, x_bar, tacperception)
            #
            if tacCONST.PN_FLAG == 'p':
                z_t = np.ravel(z_t_position)
                h_t = np.ravel(h_t_position)
            else:
                z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
                h_t = np.concatenate((h_t_position, h_t_nv), axis=None)

            # print("++++z_t:", ii, z_t)
            # print("++++h_t:", ii, h_t)
            # end1 = time.time()
            # print('time cost in forward compute ', end1 - start)

            # # posterior estimation
            if tacCONST.posteriori_FLAG:
                x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, P_state_cov, tacperception, object_param)
            else:
                x_state = x_bar

            """Give gd if necessary"""
            # x_state[3:6] = gd_state[3:6]

            # print('x_bar ', x_bar)
            # print('x_state', x_state)
            delta_t = z_t - h_t
            # print('zt - ht ', delta_t)
            self.delta_ct = np.vstack((self.delta_ct, delta_t))
            self.z_t = np.vstack((self.z_t, z_t))
            self.h_t = np.vstack((self.h_t, h_t))
            x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
            x_state_plot[0:3] = x_state[0:3]
            x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(x_state[3:6])
            self.x_state_cur = x_state_plot

            self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
            #
            np.savetxt('x_gt_world.txt', self.x_gt_world)
            np.savetxt('x_bar_all.txt', self.x_bar_all)
            np.savetxt('x_state_all.txt', self.x_state_all)
            np.savetxt('x_gt_palm.txt', self.x_gt_palm)
            np.savetxt('ju_all.txt', self.ju_all)
            np.savetxt('delta_ct.txt', self.delta_ct)
            np.savetxt('z_t.txt', self.z_t)
            np.savetxt('h_t.txt', self.h_t)
        # else:
        #     print('no contact')
        if first_contact_flag:
            viz.vis_state_contact(sim, viewer, tacperception, z_t, h_t, x_bar, x_state, char, object_param)
            self.active_fingers_taxels_render(sim, viewer, tacperception)
            tacperception.fin_num = 0
            # tacperception.fin_tri = np.zeros(4)
            tacperception.fin_tri = np.zeros(len(hand_param) - 1)


first_contact_flag = False
# ff_first_contact_flag = False
# mf_first_contact_flag = False
# rf_first_contact_flag = False
# th_first_contact_flag = False
P_state_cov = 0.1 * np.identity(6 + 4 * 3)
