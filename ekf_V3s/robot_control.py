import numpy as np
import tactile_allegro_mujo_const
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

        self.ct_g_z_position = [0, 0, 0]
        self.ct_p_z_position = [0, 0, 0]
        self.x_bar_all = [0, 0, 0, 0, 0, 0, 0]
        self.x_state_all = [0, 0, 0, 0, 0, 0, 0]
        self.x_gt_palm = [0, 0, 0, 0, 0, 0, 0]
        self.x_gt_world = [0, 0, 0]
        self.ju_all = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tactile_allegro_mujo_const.PN_FLAG == 'pn':
            self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.delta_ct = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tactile_allegro_mujo_const.PN_FLAG == 'pn':
            z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            z_t = self.z_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if tactile_allegro_mujo_const.PN_FLAG == 'pn':
            h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            h_t = self.h_t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    def robot_init(self, sim):
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3

    def p2p_p_ik(self, sim, viewer, p_start, q_end):
        q_est = self.kdl_kin_taxel.inverse(position=p_start.position, rot=p_start.orientation, \
                                           maxiter=1000, eps=0.000001)
        if q_est is not None:
            print('q_est')
            print(q_est)
        else:
            print('no solution was found, use zero pose')
            q_est = [0., 0., 0., 0.]
        return q_est

    def p2p_v_ik(self, sim, viewer, kp, p_start, p_end, ik_type):
        delta_p = p_start.position - p_end.position
        dev = np.linalg.norm(delta_p)
        counter = 0
        delta_p_save = [[0, 0, 0]]
        jnt = [[0, 0, 0, 0]]
        jnt_dot = [[0, 0, 0, 0]]
        twist = [[0, 0, 0, 0, 0, 0]]
        while (dev > 0.0005) and (counter < 10000):
            counter = counter + 1
            q = self.get_cur_jnt(sim)[4:8]
            taxel_p, taxel_o = self.kdl_kin_taxel.FK(q)
            taxel_p_g, taxel_o_g = ug.pose_trans_palm_to_world(sim, taxel_p, taxel_o)
            des_p, des_o = ug.pose_trans_palm_to_world(sim, p_start.position, p_start.orientation)
            viz.cor_frame_visual(viewer, taxel_p_g, np.array(taxel_o_g), 0.03, 'taxel')
            # viz.geo_visual(viewer, des_p, np.array(des_o), 0.001, const.GEOM_BOX, 0, 'h')
            viz.cor_frame_visual(viewer, des_p, np.array(des_o), 0.03, 'des')
            delta_p = p_start.position - taxel_p
            kp_delta_p = kp * np.ravel(delta_p)
            l_delta_p = np.ravel(delta_p)
            delta_p_save.append([l_delta_p[0], l_delta_p[1], l_delta_p[2]])
            delta_o = np.array(np.matmul(p_start.orientation.transpose(), taxel_o))
            theta = math.acos((np.trace(delta_o) - 1.0)/2.0)
            omega = np.array([0., 0., 0.])
            omega[0] = (delta_o[2][1]-delta_o[1][2]) / (2 * math.sin(theta))
            omega[1] = (delta_o[0][2]-delta_o[2][0]) / (2 * math.sin(theta))
            omega[2] = (delta_o[1][0]-delta_o[0][1]) / (2 * math.sin(theta))
            # print('theta ', theta)
            # print('omega is ', omega[0], omega[1], omega[2] )
            # print('local rx ')
            # print(ug.vec2rot(theta * np.array(omega)))
            taxel_p_g, taxel_rot_ax_g = ug.pose_trans_palm_to_world(sim, taxel_p, ug.vec2rot(theta * np.array(omega)))
            viewer.add_marker(pos=taxel_p_g, mat=np.array(taxel_rot_ax_g), type=const.GEOM_ARROW, label='rot',
                      size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

            kp_delta_o = kp * theta * np.array(omega)

            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(kp_delta_p[0], kp_delta_p[1], kp_delta_p[2])
            vel_twist.rot = kdl.Vector(kp_delta_o[0], kp_delta_o[1], kp_delta_o[2])
            twist.append([kp_delta_p[0], kp_delta_p[1], kp_delta_p[2], kp_delta_o[0], kp_delta_o[1], kp_delta_o[2]])
            # print('twist ', kp_delta_p[0], kp_delta_p[1], kp_delta_p[2], kp_delta_o[0], kp_delta_o[1], kp_delta_o[2])

            # sp_w, so_w = ug.pose_trans_palm_to_world(sim, kdl_p, kdl_o)
            # des_vec = [vel, vel, 0]
            # des_rot_z = ug.vec2rot(des_vec)
            # print('position: ', sp_w)
            # # print('orientation: ', so_w)
            # viz.geo_visual(viewer, sp_w, des_rot_z, 0.1, const.GEOM_ARROW, 0, 'des')

            q_dot = kdl.JntArray(len(self.kdl_kin_taxel.get_joint_names()))
            q_pos_input = kdl.JntArray(len(self.kdl_kin_taxel.get_joint_names()))
            for i, q_i in enumerate(self.get_cur_jnt(sim)[4:8]):
                q_pos_input[i] = q_i
            jnt.append(self.get_cur_jnt(sim)[4:8])

            print('ik_type ', ik_type)
            if ik_type == IK_type.IK_V_FULL:
                #pinv solution
                # print('cur jnt is ', self.get_cur_jnt(sim)[4:8])
                succ = self.kdl_kin_taxel._ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_dot)
                print("succ:", succ)
                q_dot = np.array(joint_kdl_to_list(q_dot))
                jnt_dot.append(q_dot)
            if ik_type == IK_type.IK_V_WDLS:
                #wdls solution
                _ik_wdls_v_kdl = kdl.ChainIkSolverVel_wdls(self.kdl_kin_taxel.chain)
                for i, q_i in enumerate(self.get_cur_jnt(sim)[4:8]):
                    q_pos_input[i] = q_i
                matrix_weight = np.eye(4)
                matrix_weight[0][0] = 0
                matrix_weight[1][1] = 0.8
                matrix_weight[2][2] = 0.2
                matrix_weight[3][3] = 0.8
                _ik_wdls_v_kdl.setWeightJS(matrix_weight)
                u, s, v = np.linalg.svd(self.kdl_kin_taxel.jacobian(self.get_cur_jnt(sim)[4:8]))
                _ik_wdls_v_kdl.setLambda(0.9)
                _ik_wdls_v_kdl.CartToJnt(q_pos_input, vel_twist, q_dot)
                q_dot = np.array(joint_kdl_to_list(q_dot))
            if ik_type == IK_type.IK_V_POSITION_ONLY:
                jac = self.kdl_kin_taxel.jacobian(self.get_cur_jnt(sim)[4:8])
                jac_position = jac[:3, :4]
                # does not consider the contribution from first joint
                jac_position[0, 0] = 0.
                jac_position[1, 0] = 0.
                jac_position[2, 0] = 0.
                q_dot = np.matmul(np.linalg.pinv(jac_position), kp_delta_p)
                q_dot = np.ravel(q_dot)
                print('q is ', self.get_cur_jnt(sim)[4:8])
                print('q_dot[0] ', q_dot[0])


            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_1] + q_dot[0]
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = -0.00146
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_2] + q_dot[1]
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_3] + q_dot[2]
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_4] + q_dot[3]

            #compute deviation
            dev = np.linalg.norm(delta_p)
            print('dev ', dev)
            sim.step()
            viewer.render()
        np.savetxt("delta_p_save.txt", delta_p_save)
        np.savetxt("twist.txt", twist)
        np.savetxt("jnt.txt", jnt)
        np.savetxt("jnt_dot.txt", jnt_dot)

    def tip_servo_control(self, sim, viewer, model, finger_name, cur_tac_p, tac_name, goal_tac_p, delta_press):
        #compute the tactile feature errors
        vel = delta_press
        #transfer to the palm frame
        vel_tip = np.array([vel, 0.0, 0.0])
        p, o = self.fk_offset(sim, finger_name, tac_name, 'palm')

        # p_w, o_w = ug.pose_trans_palm_to_world(sim, p, o)
        # viz.cor_frame_visual(viewer, p_w, o_w, 0.1, 'ccc')

        vel_palm = np.matmul(o, vel_tip)
        print(finger_name, 'v_palm ', vel_palm)
        #call hand v inv control
        self.instant_v_ik_control(sim, viewer, finger_name, vel_palm, tac_name)

    def instant_v_ik_control(self, sim, viewer, finger_name, vel, tac_name):
        if finger_name == 'ff':
            jac = self.robjac_offset(sim, finger_name, self.get_cur_jnt(sim)[0:4], tac_name)
            jac_position = jac[:3, :4]
            # does not consider the contribution from first joint
            jac_position[0, 0] = 0.
            jac_position[1, 0] = 0.
            jac_position[2, 0] = 0.
            q_dot = np.matmul(np.linalg.pinv(jac_position), vel.transpose())
            q_dot = 5*np.ravel(q_dot)

            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = 0.0

            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2] + q_dot[1]
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3] + q_dot[2]
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4] + q_dot[3]

        if finger_name == 'mf':
            jac = self.robjac_offset(sim, finger_name, self.get_cur_jnt(sim)[4:8], tac_name)
            jac_position = jac[:3, :4]
            # does not consider the contribution from first joint
            jac_position[0, 0] = 0.
            jac_position[1, 0] = 0.
            jac_position[2, 0] = 0.
            q_dot = np.matmul(np.linalg.pinv(jac_position), vel.transpose())
            q_dot = 5*np.ravel(q_dot)
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_2] + q_dot[1]
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_3] + q_dot[2]
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_4] + q_dot[3]

        if finger_name == 'rf':
            jac = self.robjac_offset(sim, finger_name, self.get_cur_jnt(sim)[8:12], tac_name)
            jac_position = jac[:3, :4]
            # does not consider the contribution from first joint
            jac_position[0, 0] = 0.
            jac_position[1, 0] = 0.
            jac_position[2, 0] = 0.
            q_dot = np.matmul(np.linalg.pinv(jac_position), vel.transpose())
            q_dot = 5*np.ravel(q_dot)
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_2] + q_dot[1]
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_3] + q_dot[2]
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_4] + q_dot[3]

        if finger_name == 'th':
            jac = self.robjac_offset(sim, finger_name, self.get_cur_jnt(sim)[12:16], tac_name)
            jac_position = jac[:3, :4]
            # does not consider the contribution from first joint
            jac_position[0, 0] = 0.
            jac_position[1, 0] = 0.
            jac_position[2, 0] = 0.
            q_dot = np.matmul(np.linalg.pinv(jac_position), vel.transpose())
            q_dot = 5*np.ravel(q_dot)
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
                sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1] + q_dot[0]
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2] + q_dot[1]
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3] + q_dot[2]
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4] + q_dot[3]

    def ik_control(self, sim, viewer, kin_finger, vel, kdl):
        angles = [0, 0.2, 0.2, 0.8]
        ff_q_start = angles[0:4]
        kdl_p, kdl_o = kin_finger.FK(ff_q_start)
        kdl_p_s, kdl_o_s = ug.pose_trans_palm_to_world(sim, kdl_p, kdl_o)
        angles = [0, 0.4, 0.4, 1.2]
        ff_q_end = angles[0:4]
        kdl_p, kdl_o = kin_finger.FK(ff_q_end)
        kdl_p_e, kdl_o_e = ug.pose_trans_palm_to_world(sim, kdl_p, kdl_o)

        for _ in range(200):
            # visualize start_p and end_p
            viz.geo_visual(viewer, kdl_p_s, kdl_o_s, 0.003, const.GEOM_BOX, 0, 'h')
            viz.geo_visual(viewer, kdl_p_e, kdl_o_e, 0.003, const.GEOM_BOX, 0, 'h')

            angles = self.get_cur_jnt(sim)
            kdl_p, kdl_o = kin_finger.FK(angles[0: 4])
            print("kdl p ", kdl_p)
            print("kdl o ", kdl_o)
            vel_twist = kdl.Twist()
            vel_twist.vel = kdl.Vector(vel, vel, 0)
            vel_twist.rot = kdl.Vector(0, 0, 0)

            sp_w, so_w = ug.pose_trans_palm_to_world(sim, kdl_p, kdl_o)
            des_vec = [vel, vel, 0]
            des_rot_z = ug.vec2rot(des_vec)
            print('position: ', sp_w)
            # print('orientation: ', so_w)
            viz.geo_visual(viewer, sp_w, des_rot_z, 0.1, const.GEOM_ARROW, 0, 'des')

            q_out = kdl.JntArray(len(kin_finger.get_joint_names()))
            q_pos_input = kdl.JntArray(len(kin_finger.get_joint_names()))
            for i, q_i in enumerate(self.get_cur_jnt(sim)[0:4]):
                q_pos_input[i] = q_i

            succ = kin_finger._ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
            print("succ:", succ)
            q_out = np.array(joint_kdl_to_list(q_out))

            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1] + q_out[0]
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2] + q_out[1]
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3] + q_out[2]
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
                sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4] + q_out[3]

            # sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = 0
            # sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = 0
            # sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = 0
            # sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = 0
            sim.step()
            viewer.render()
            del viewer._markers[:]

    def inc_finger_jnt(self, sim, finger_name, inc):
        if(finger_name == 'ff'):
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + inc
        # print('cmd ff ', sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2],\
        #       sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3],\
        #      sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] )
        if(finger_name == 'mf'):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] + inc
        if(finger_name == 'rf'):
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] + inc
        if(finger_name == 'th'):
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + inc
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + inc


    def finger_contact(self, sim, viewer, finger_name, tacperception):
        while (tacperception.is_finger_contact(sim, finger_name) != True):
            self.inc_finger_jnt(sim, finger_name, 0.001)
            sim.step()
            viewer.render()
        print(finger_name+'contact')

    def fingers_contact(self, sim, viewer, tacperception):
        self.finger_contact(sim, viewer, 'ff', tacperception)
        self.finger_contact(sim, viewer, 'mf', tacperception)
        self.finger_contact(sim, viewer, 'rf', tacperception)
        self.finger_contact(sim, viewer, 'th', tacperception)

    def moveto_jnt(self, sim, viewer, finger_name, q_est, usedtime):
        if finger_name == 'ff':
            for _ in range(usedtime):
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = q_est[0]
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = q_est[1]
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = q_est[2]
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = q_est[3]
                sim.step()
                viewer.render()
        if finger_name == 'mf':
            for _ in range(usedtime):
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = q_est[0]
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = q_est[1]
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = q_est[2]
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = q_est[3]
                sim.step()
                viewer.render()
        if finger_name == 'rf':
            for _ in range(usedtime):
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_1] = q_est[0]
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = q_est[1]
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = q_est[2]
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = q_est[3]
                sim.step()
                viewer.render()
        if finger_name == 'mf':
            for _ in range(usedtime):
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = q_est[0]
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = q_est[1]
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = q_est[2]
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = q_est[3]
                sim.step()
                viewer.render()
    def move_ik_finger(self, sim, kdl_kin, ee_tget_posquat, gripper_action=0.04, viewer=None):
        # ee_target is in world frame
        ee_curr_posquat = ug.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        max_step = 1000
        no_step = 0
        threshold = 0.001
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
        # ee_jac = jac_geom(sim, "link_3.0_tip")
        for i in range(max_step):
            if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
                break
            try:
                q_pos_test = sim.data.qpos
                q_pos_temp = np.array(q_pos_test[13:17])
                print("q_pos_temp:", q_pos_temp)
                ee_jac = kdl_kin.jacobian(q_pos_temp)
                # if i == 0:
                # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
                vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                                 quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

                qvel = np.matmul(np.linalg.pinv(ee_jac), vel.transpose())
                print("qvel:\n", qvel)
                sim.data.ctrl[6:10] = sim.data.qpos[13:17] + qvel
                # print("sim.data.ctrl[6:10]:\n", sim.data.ctrl[6:10] )
                # sim.data.ctrl[7] = gripper_action
                sim.step()
                viewer.render()

                ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
            except Exception as e:
                return 0
        return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

    def move_ik_kdl_finger_pinv(sim, kdl_kin, ee_tget_posquat, gripper_action=0.04, viewer=None):
        # ee_target is in world frame
        ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        max_step = 1000
        no_step = 0
        threshold = 0.001
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
        # ee_jac = jac_geom(sim, "link_3.0_tip")
        for i in range(max_step):
            if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
                break
            try:
                q_pos_test = sim.data.qpos
                q_pos_temp = np.array(q_pos_test[13:17])
                # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
                vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                                 quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

                print("vel:", vel)
                vel_twist = kdl.Twist()
                vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
                vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

                _ik_v_kdl = kdl.ChainIkSolverVel_pinv(kdl_chain)

                num_joints = len(kdl_kin.get_joint_names())
                q_out = kdl.JntArray(num_joints)

                q_pos_input = kdl.JntArray(num_joints)
                for i, q_i in enumerate(q_pos_temp):
                    q_pos_input[i] = q_i

                succ = _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
                print("succ:", succ)
                q_out = np.array(joint_kdl_to_list(q_out))

                sim.data.ctrl[6:10] = sim.data.qpos[13:17] + q_out
                sim.step()
                viewer.render()

                ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
            except Exception as e:
                return 0
        return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

    def move_ik_kdl_finger_wdls_middle(sim, kdl_kin, ee_tget_posquat, gripper_action=0.04, viewer=None):
        # ee_target is in world frame
        # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
        max_step = 1000
        no_step = 0
        threshold = 0.001
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
        # ee_jac = jac_geom(sim, "link_3.0_tip")
        for i in range(max_step):
            if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
                break
            try:
                q_pos_test = sim.data.qpos
                # q_pos_temp  = np.array(q_pos_test[13:17])
                q_pos_temp = np.array(q_pos_test[17:21])
                # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
                vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                                 quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))

                # 转化速度到twist形式
                vel_twist = kdl.Twist()
                vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
                vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

                _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_chain)
                num_joints = len(kdl_kin.get_joint_names())
                q_out = kdl.JntArray(num_joints)

                q_pos_input = kdl.JntArray(num_joints)
                for i, q_i in enumerate(q_pos_temp):
                    q_pos_input[i] = q_i

                matrix_weight = np.eye(4)
                matrix_weight[0][0] = 0.1
                matrix_weight[1][1] = 0.5
                matrix_weight[2][2] = 0.3
                matrix_weight[3][3] = 0.2
                _ik_v_kdl.setWeightJS(matrix_weight)

                _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
                q_out = np.array(joint_kdl_to_list(q_out))

                # 这里添加了限幅值
                # if(q_out[1]>0.05):
                #     q_out[1] = 0.05
                # if(q_out[1]<-0.05):
                #     q_out[1] = -0.05
                # print("q_out:", q_out)
                # q_out[q_out>1] = 1
                # sim.data.ctrl[6:10]   = sim.data.qpos[13:17] + q_out
                # sim.data.qpos[6:10] = sim.data.qpos[13:17] + q_out
                sim.data.ctrl[10:14] = sim.data.qpos[17:21] + q_out
                sim.step()
                viewer.render()

                # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
                ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
            except Exception as e:
                return 0
        return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

    def move_ik_kdl_finger_wdls_king(sim, kdl_kin, ee_tget_posquat, gripper_action=0.04, viewer=None):
        # ee_target is in world frame
        ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
        # ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_7.0_tip")
        max_step = 1000
        no_step = 0
        threshold = 0.001
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
        # kdl_kin = KDLKinematics(robot, "palm_link", "link_7.0_tip")
        # ee_jac = jac_geom(sim, "link_3.0_tip")
        for i in range(max_step):
            if (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold)):
                print("**********************************************************************")
                break
            try:
                q_pos_test = sim.data.qpos
                q_pos_temp = np.array(q_pos_test[13:17])
                # q_pos_temp  = np.array(q_pos_test[17:21])
                # vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]) / 5, quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
                vel = np.hstack(((ee_tget_posquat[:3] - ee_curr_posquat[:3]),
                                 quat2vel(mul_quat(ee_tget_posquat[-4:], conj_quat(ee_curr_posquat[-4:])), 1)))
                print("vel:", vel)
                # 转化速度到twist形式
                vel_twist = kdl.Twist()
                vel_twist.vel = kdl.Vector(vel[0], vel[1], vel[2])
                vel_twist.rot = kdl.Vector(vel[3], vel[4], vel[5])

                _ik_v_kdl = kdl.ChainIkSolverVel_wdls(kdl_chain)
                num_joints = len(kdl_kin.get_joint_names())
                q_out = kdl.JntArray(num_joints)

                q_pos_input = kdl.JntArray(num_joints)
                for i, q_i in enumerate(q_pos_temp):
                    q_pos_input[i] = q_i

                _ik_v_kdl.CartToJnt(q_pos_input, vel_twist, q_out)
                q_out = np.array(joint_kdl_to_list(q_out))
                sim.data.ctrl[6:10] = sim.data.qpos[13:17] + q_out
                # sim.data.ctrl[10:14] = sim.data.qpos[17:21] + q_out
                sim.step()
                viewer.render()

                ee_curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
            except Exception as e:
                return 0
        return (posquat_equal(ee_curr_posquat[:7], ee_tget_posquat[:7], threshold))

    # def move_interperate_point(sim, desire_pos_quat, curr_posquat, viewer=None):
    #     # curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
    #     delta_k = 5
    #     # X = np.arange(0, 1, 1)
    #     # Y = [curr_posquat, desire_pos_quat]
    #     interpolate_point = []
    #     for i in range(1,delta_k+1):
    #         interpolate_point.append(curr_posquat + (desire_pos_quat-curr_posquat)/delta_k*i)
    #
    #     count_execute = 0;
    #     for k,inter in enumerate(interpolate_point):
    #         done_execute = False
    #         # move_ik_kdl_finger_wdls(sim, inter)
    #         print("inter:", inter)
    #         while(count_execute < 200):
    #             done_execute = move_ik_kdl_finger_wdls(sim, inter)
    #             count_execute += 1
    #             sim.step()
    #             viewer.render()
    #         count_execute = 0

    # 对这里进行修正，进行修改即可
    # def force_control(sim, force_set, cur_force):
    # 	kp, ki, kd = 0.0, 0.3, 0.0
    # 	pid = pid.PID(kp, ki, kd)
    # 	transfom_factor = 0.000003
    # 	setpoint = 10
    #
    #     transform_base2tip3 = get_relative_posquat(sim, "base_link", "link_3.0 _tip")
    #     rot = posquat2trans(transform_base2tip3)[0:3, 0:3]
    #     pid_out = pid.calc(cur_force, force_set)
    #     ze = np.array([0, 0, pid_out*transfom_factor]).transpose()
    #     ze = np.matmul(rot, ze)
    #     z = pos-ze
    #     transform_base2tip3[:3] = z
    #     desire_pos_quat_in_force = np.array();
    #     move_ik_kdl_finger_wdls(sim, desire_pos_quat_in_force)

    def index_finger(self, sim, input1, input2):
        if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
                tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any():
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + input1
        else:
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + input2


    def middle_finger(self, sim, input1, input2):
        if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
                tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any():  # 中指
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] + input1
        else:

            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] + input2


    def middle_finger_vel(self, sim, input1, input2):
        print(sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1])
        if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
                tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any():  # 中指
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        else:
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
            sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2


    def ring_finger(self, sim, input1, input2):
        if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
                tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any():  # 小拇指
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] + input1
        else:
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] + input2

    def thumb_zero(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'th', [0., 0., 0., 0.], 200)
    def ff_zero(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'ff', [0., 0., 0., 0.], 200)

    def mf_zero(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'mf', [0., 0., 0., 0.], 200)

    def rf_zero(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'rf', [0., 0., 0., 0.], 200)

    def hand_zero(self, sim, viewer):
        self.thumb_zero(sim, viewer)
        self.ff_zero(sim, viewer)
        self.mf_zero(sim, viewer)
        self.rf_zero(sim, viewer)

    def thumb_pregrasp(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'th', [0., 0., 0., 0.], 50)
    def ff_pregrasp(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'ff', [0., 0., 0., 0.], 20)

    def mf_pregrasp(self, sim, viewer):
        self.moveto_jnt(sim, viewer, 'mf', [0.7, 0., 0., 0.], 1000)

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
        position_taxel_inworld, orien_taxel_inworld = ug.pose_trans_palm_to_world(sim, position_taxel_inpalm, orien_taxel_inpalm)
        if ref_frame == 'palm':
            return position_taxel_inpalm, orien_taxel_inpalm
        else:
            return position_taxel_inworld, orien_taxel_inworld

    def ct_fingers_taxel_render(self, sim, viewer, model, tacperception):
        #weighted contact position (taxel)
        self.ct_finger_taxel_render(sim, viewer, model, 'ff', tacperception)
        self.ct_finger_taxel_render(sim, viewer, model, 'mf', tacperception)
        self.ct_finger_taxel_render(sim, viewer, model, 'rf', tacperception)
        self.ct_finger_taxel_render(sim, viewer, model, 'th', tacperception)

    def ct_finger_taxel_render(self, sim, viewer, model, finger_name, tacperception):
        if tacperception.is_finger_contact(sim, finger_name) == True:
            pos_zt_world = ug.posquat2trans(tacperception.get_contact_taxel_position(sim, model, finger_name, "world"))
            viz.geo_visual(viewer, pos_zt_world[0:3, 3], pos_zt_world[0:3, 0:3], 0.001, tactile_allegro_mujo_const.GEOM_BOX, 0, "z")
            # tacperception.get_contact_taxel_nv(sim, model, finger_name, "palm_link")
    def active_fingers_taxels_render(self, sim, viewer, tacperception):
        self.active_finger_taxels_render(sim, viewer, 'ff', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'mf', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'rf', tacperception)
        self.active_finger_taxels_render(sim, viewer, 'th', tacperception)
        print('........................................\n')
    def active_finger_taxels_render(self, sim, viewer, finger_name, tacperception):
        if tacperception.is_finger_contact(sim, finger_name) == True:
            taxels_id = tacperception.get_contact_taxel_id_withoffset(sim, finger_name)
            taxels_pose_gt = []
            taxels_pose_fk = []
            print(finger_name + "viz taxels: ", end='')
            for i in taxels_id:
                active_taxel_name = sim.model._sensor_id2name[i]
                print(active_taxel_name+' ', end='')
                # compute ground truth taxels
                taxel_pose_gt = taxel_pose()
                pose_taxels_w = ug.get_relative_posquat(sim, "world", active_taxel_name)
                pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_taxels_w)
                taxel_pose_gt.position = pos_p_world
                taxel_pose_gt.orien = pos_o_world
                taxels_pose_gt.append(taxel_pose_gt)
            print('')
            viz.active_taxels_visual(viewer, taxels_pose_gt, 'gt')

    def rf_move_taxels_render(self, sim, model, viewer, hand_param, tacperception):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = 0.9
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = 1
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = 1
            flag_rf = tacperception.is_finger_contact(sim, hand_param[3][0])
            if flag_rf == True:
                taxels_id = tacperception.get_contact_taxel_id_withoffset(sim, 'rf')
                taxels_pose_gt = []
                taxels_pose_fk = []
                for i in taxels_id:
                    active_taxel_name = sim.model._sensor_id2name[i]
                    #compute ground truth taxels
                    taxel_pose_gt = taxel_pose()
                    pose_taxels_w = ug.get_relative_posquat(sim, "world", active_taxel_name)
                    pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_taxels_w)
                    taxel_pose_gt.position = pos_p_world
                    taxel_pose_gt.orien = pos_o_world
                    taxels_pose_gt.append(taxel_pose_gt)

                    #compute taxels from forward kinematics
                    position_taxel_inworld, orien_taxel_inworld = self.fk_offset(sim, 'rf', active_taxel_name)
                    taxel_pose_fk = taxel_pose()
                    taxel_pose_fk.position = position_taxel_inworld
                    taxel_pose_fk.orien = orien_taxel_inworld
                    taxels_pose_fk.append(taxel_pose_fk)

                viz.active_taxels_visual(viewer, taxels_pose_gt, 'gt')
                viz.active_taxels_visual(viewer, taxels_pose_fk, 'fk')
            sim.step()
            viewer.render()
    def mf_move_taxels_render(self, sim, model, viewer, hand_param, tacperception):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = 0.9
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = 1
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = 1
            flag_mf = tacperception.is_finger_contact(sim, hand_param[2][0])
            if flag_mf == True:
                taxels_id = tacperception.get_contact_taxel_id_withoffset(sim, 'mf')
                taxels_pose_gt = []
                taxels_pose_fk = []
                for i in taxels_id:
                    active_taxel_name = sim.model._sensor_id2name[i]
                    print('taxels name ', active_taxel_name)
                    # compute ground truth taxels
                    taxel_pose_gt = taxel_pose()
                    pose_taxels_w = ug.get_relative_posquat(sim, "world", active_taxel_name)
                    pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_taxels_w)
                    taxel_pose_gt.position = pos_p_world
                    taxel_pose_gt.orien = pos_o_world
                    taxels_pose_gt.append(taxel_pose_gt)

                    # compute taxels from forward kinematics
                    position_taxel_inworld, orien_taxel_inworld = self.fk_offset(sim, 'mf', active_taxel_name)
                    taxel_pose_fk = taxel_pose()
                    taxel_pose_fk.position = position_taxel_inworld
                    taxel_pose_fk.orien = orien_taxel_inworld
                    taxels_pose_fk.append(taxel_pose_fk)

                viz.active_taxels_visual(viewer, taxels_pose_gt, 'gt')
                viz.active_taxels_visual(viewer, taxels_pose_fk, 'fk')
            sim.step()
            viewer.render()
    def rf_pregrasp(self, sim, viewer):
        self.moveto_jnt(self, sim, viewer, 'rf', [0., 0., 0., 0.], 20)
    def hand_pregrasp(self, sim, viewer):
        # self.thumb_pregrasp(sim, viewer)
        # self.ff_pregrasp(sim, viewer)
        self.mf_pregrasp(sim, viewer)
        # self.rf_pregrasp(sim, viewer)

    def mf_taxel_poseture_I(self, sim, viewer):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = 0.6
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = 0.3
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = 0.3
            q = self.get_cur_jnt(sim)[4:8]
            p, o = self.kdl_kin_taxel.FK(q)
            p_e, o_e = ug.pose_trans_palm_to_world(sim, p, o)
            viewer.add_marker(pos=p_e, mat=o_e, type=tactile_allegro_mujo_const.GEOM_BOX, label="",
                              size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
            sim.step()
            viewer.render()
        q = self.get_cur_jnt(sim)[4:8]
        p, o = self.kdl_kin_taxel.FK(q)
        return p, o, q

    def mf_taxel_poseture_II(self, sim, viewer):
        for _ in range(1000):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = 0.2
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = 0.1
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = 0.2
            q = self.get_cur_jnt(sim)[4:8]
            p, o = self.kdl_kin_taxel.FK(q)
            p_e, o_e = ug.pose_trans_palm_to_world(sim, p, o)
            viewer.add_marker(pos=p_e, mat=o_e, type=tactile_allegro_mujo_const.GEOM_BOX, label="",
                              size=np.array([0.001, 0.001, 0.001]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
            sim.step()
            viewer.render()
        q = self.get_cur_jnt(sim)[4:8]
        print('q at II stop ', q)
        p, o = self.kdl_kin_taxel.FK(q)
        return p, o, q


    def pre_thumb(self, sim, viewer):
        for _ in range(1000):
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] + 0.05
            sim.step()
            viewer.render()


    def thumb(self, sim, input1, input2):
        if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
                tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any():  # da拇指
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + input1
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + input1
        else:
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + input2
    def get_cur_jnt(self, sim):
        cur_jnt = np.zeros(tactile_allegro_mujo_const.FULL_FINGER_JNTS_NUM)
        cur_jnt[0:4] = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],
                            sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],
                            sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],
                            sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])

        cur_jnt[4:8] = np.array([sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_1],
                            sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_2],
                            sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_3],
                            sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_4]])

        cur_jnt[8:12] = np.array([sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_1],
                            sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_2],
                            sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_3],
                            sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_4]])

        cur_jnt[12:16] = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1],
                            sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2],
                            sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3],
                            sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])
        return cur_jnt

    # def config_robot_tip_kin():
    #     # kinematic chain for all fingers
    #     robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
    #     # first finger
    #     kdl_kin_ff = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    #     # middle finger
    #     kdl_kin_mf = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    #     # ring finger
    #     kdl_kin_rf = KDLKinematics(robot, "palm_link", "link_11.0_tip")
    #     # thumb
    #     kdl_kin_th = KDLKinematics(robot, "palm_link", "link_15.0_tip")
    #     # kdl_tree = kdl_tree_from_urdf_model(robot)
    #     return kdl_kin_ff, kdl_kin_mf, kdl_kin_rf, kdl_kin_th

    def config_robot(self, taxel_name):
        # kinematic chain for all fingers
        # self.robot = URDF.from_xml_file("../../robots/allegro_hand_right_with_tactile.urdf")
        kdl_kin = KDLKinematics(self.robot, "palm_link", taxel_name)
        return kdl_kin

    def robjac_offset(self,sim, finger_name, q, taxel_name):
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
    def update_augmented_state(self, sim, model, hand_param, tacperception, x_state):
        if tacperception.is_finger_contact(sim, hand_param[1][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state[6] = pos_contact0[0][0]
            x_state[7] = pos_contact0[0][1]
            x_state[8] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[2][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state[9] = pos_contact0[0][0]
            x_state[10] = pos_contact0[0][1]
            x_state[11] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[3][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state[12] = pos_contact0[0][0]
            x_state[13] = pos_contact0[0][1]
            x_state[14] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[4][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            # print('x_state ', x_state)
            x_state[15] = pos_contact0[0][0]
            x_state[16] = pos_contact0[0][1]
            x_state[17] = pos_contact0[0][2]

        return x_state

    def augmented_state(self, sim, model, hand_param, tacperception, x_state):
        if tacperception.is_finger_contact(sim, hand_param[1][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[2][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[3][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[4][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.0, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])
        return x_state

    def interaction(self, sim, model, viewer, hand_param, object_param, alg_param, \
                    ekf_grasping, tacperception, char):
        global first_contact_flag, x_all, gd_all, ff_first_contact_flag, \
            mf_first_contact_flag, rf_first_contact_flag, th_first_contact_flag, \
        P_state_cov, x_state, last_angles, x_bar, z_t, h_t

        flag_ff = tacperception.is_finger_contact(sim, hand_param[1][0])
        flag_mf = tacperception.is_finger_contact(sim, hand_param[2][0])
        flag_rf = tacperception.is_finger_contact(sim, hand_param[3][0])
        flag_th = tacperception.is_finger_contact(sim, hand_param[4][0])

        if ((flag_ff == True) or (flag_mf == True) or (flag_rf == True) or (flag_th == True)):
            if tacperception.is_ff_contact == True:
                tacperception.fin_num += 1
                tacperception.fin_tri[0] = 1
            if tacperception.is_mf_contact == True:
                tacperception.fin_num += 1
                tacperception.fin_tri[1] = 1
            if tacperception.is_rf_contact == True:
                tacperception.fin_num += 1
                tacperception.fin_tri[2] = 1
            if tacperception.is_th_contact == True:
                tacperception.fin_num += 1
                tacperception.fin_tri[3] = 1
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
                init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), \
                                                      (1, 3)), \
                                    np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))

                x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                # attention, here orientation we use the axis angle representation.
                # x_state = np.array([ug.pos_quat2axis_angle(x_state)])
                x_state = ug.pos_quat2axis_angle(x_state)
                np.set_printoptions(suppress=True)
                print('x_state from beginning before add noise', x_state)
                if tactile_allegro_mujo_const.initE_FLAG:
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
                x_state = self.augmented_state(sim, model, hand_param, tacperception, x_state)
                # print('x_state from beginning ', x_state)
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
                if tacperception.is_ff_contact == True:
                    ff_first_contact_flag = True
                if tacperception.is_mf_contact == True:
                    mf_first_contact_flag = True
                if tacperception.is_rf_contact == True:
                    rf_first_contact_flag = True
                if tacperception.is_th_contact == True:
                    th_first_contact_flag = True

                last_angles = self.get_cur_jnt(sim)
                self.mea_filter_js = lfilter(9, 0.01, last_angles, 16)
                # last_angles = np.zeros(tactile_allegro_mujo_const.FULL_FINGER_JNTS_NUM)
                print('return early')
                return
            elif ((flag_ff == True) and (ff_first_contact_flag) == False):
                # x_state = np.ravel(x_state)
                x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                ff_first_contact_flag = True
            elif ((flag_mf == True) and (mf_first_contact_flag) == False):
                # x_state = np.ravel(x_state)
                x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                mf_first_contact_flag = True
            elif ((flag_rf == True) and (rf_first_contact_flag) == False):
                # x_state = np.ravel(x_state)
                x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                rf_first_contact_flag = True
            elif ((flag_th == True) and (th_first_contact_flag) == False):
                # x_state = np.ravel(x_state)
                x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                th_first_contact_flag = True
            # else:
            #     print('no else')

            # print('P_state_cov ', P_state_cov)
            # x_state = np.ravel(x_state)
            gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
            g1 = ug.get_relative_posquat(sim, "world", "cup")
            g2 = ug.get_relative_posquat(sim, "world", "palm_link")

            gd_state = ug.posquat2posrotvec_hacking(gd_posquat)

            """ Prediction step in EKF """
            # todo can not use ground truth update the state at every step
            # x_state[:6] = gd_state
            cur_angles_tmp = self.get_cur_jnt(sim)
            # do a rolling average
            cur_angles, self.mea_filter_js.z = \
                self.mea_filter_js.lp_filter(cur_angles_tmp, 16)
            x_bar, P_state_cov, ju_all = \
                ekf_grasping.state_predictor(sim, model, hand_param, object_param, \
                                      x_state, tacperception, P_state_cov, cur_angles,\
                                                   last_angles, self)

            last_angles = cur_angles
            # last_angles = cur_angles_tmp
            # compute the axis and angle for plot_data
            x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
            x_bar_plot[0:3] = x_bar[0:3]
            x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(x_bar[3:6])

            gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
            gd_state_plot[0:3] = gd_state[0:3]
            gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])

            self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
            self.x_gt_palm = np.vstack((self.x_gt_palm, gd_state_plot))
            self.ju_all = np.vstack((self.ju_all, ju_all[6:12]))

            #
            h_t_position, h_t_nv = ekf_grasping.observe_computation(x_bar, tacperception, sim)
            #
            z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, \
                                                           x_bar, tacperception)
            #
            if tactile_allegro_mujo_const.PN_FLAG == 'p':
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
            if tactile_allegro_mujo_const.posteriori_FLAG:
                x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, \
                                                                   P_state_cov, tacperception)
            else:
                x_state = x_bar

            x_state[3:6] = gd_state[3:6]
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
            viz.vis_state_contact(sim, viewer, tacperception, z_t, h_t, x_bar, x_state, char)
            self.active_fingers_taxels_render(sim, viewer, tacperception)
            tacperception.fin_num = 0
            tacperception.fin_tri = np.zeros(4)

first_contact_flag = False
ff_first_contact_flag = False
mf_first_contact_flag = False
rf_first_contact_flag = False
th_first_contact_flag = False
P_state_cov = 0.1 * np.identity(6 + 4 * 3)


