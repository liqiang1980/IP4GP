import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_kinematics import joint_kdl_to_list
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import util_geometry as ug
import time
import fcl
from scipy.spatial.transform import Rotation

import viz
import PyKDL as kdl
from mujoco_py import const

class ROBCTRL:

    def __init__(self):
        # self.robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
        self.robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
        # first finger
        self.kdl_kin0 = KDLKinematics(self.robot, "palm_link", "link_3.0_tip")
        # middle finger
        self.kdl_kin1 = KDLKinematics(self.robot, "palm_link", "link_7.0_tip")
        # ring finger
        self.kdl_kin2 = KDLKinematics(self.robot, "palm_link", "link_11.0_tip")
        # thumb
        self.kdl_kin3 = KDLKinematics(self.robot, "palm_link", "link_15.0_tip")
        self.ct_g_z_position = [0, 0, 0]
        self.ct_p_z_position = [0, 0, 0]
        self.x_bar_all = [0, 0, 0, 0, 0, 0]
        self.x_gt = [0, 0, 0, 0, 0, 0]

    def robot_init(self, sim):
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
        sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3


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
        for _ in range(200):
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = 0
            sim.step()
            viewer.render()
    def ff_zero(self, sim, viewer):
        for _ in range(200):
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = 0
            sim.step()
            viewer.render()

    def mf_zero(self, sim, viewer):
        for _ in range(200):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = 0
            sim.step()
            viewer.render()

    def rf_zero(self, sim, viewer):
        for _ in range(200):
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = 0
            sim.step()
            viewer.render()

    def hand_zero(self, sim, viewer):
        self.thumb_zero(sim, viewer)
        self.ff_zero(sim, viewer)
        self.mf_zero(sim, viewer)
        self.rf_zero(sim, viewer)

    def thumb_pregrasp(self, sim, viewer):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = 0.5
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = 0
            sim.step()
            viewer.render()
    def ff_pregrasp(self, sim, viewer):
        for _ in range(2000):
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = 0.2
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = 0.2
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = 0.8
            sim.step()
            viewer.render()

    def mf_pregrasp(self, sim, viewer):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_2] = 0.2
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_3] = 0.6
            sim.data.ctrl[tactile_allegro_mujo_const.MF_CTRL_4] = 0
            sim.step()
            viewer.render()

    def rf_pregrasp(self, sim, viewer):
        for _ in range(500):
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_1] = 0
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_2] = 0.5
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_3] = 0.8
            sim.data.ctrl[tactile_allegro_mujo_const.RF_CTRL_4] = 0
            sim.step()
            viewer.render()
    def hand_pregrasp(self, sim, viewer):
        self.thumb_pregrasp(sim, viewer)
        self.ff_pregrasp(sim, viewer)
        self.mf_pregrasp(sim, viewer)
        self.rf_pregrasp(sim, viewer)


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
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + input1 * 5
        else:
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + input2
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
                sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + input2 * 5
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
    #     kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    #     # middle finger
    #     kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    #     # ring finger
    #     kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
    #     # thumb
    #     kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
    #     # kdl_tree = kdl_tree_from_urdf_model(robot)
    #     return kdl_kin0, kdl_kin1, kdl_kin2, kdl_kin3

    def config_robot(self, taxel_name):
        # kinematic chain for all fingers
        # self.robot = URDF.from_xml_file("../../robots/allegro_hand_right_with_tactile.urdf")
        kdl_kin = KDLKinematics(self.robot, "palm_link", taxel_name)
        return kdl_kin

    def update_augmented_state(self, sim, model, hand_param, tacperception, x_state):
        if tacperception.is_finger_contact(sim, hand_param[1][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state[6] = pos_contact0[0][0]
            x_state[7] = pos_contact0[0][1]
            x_state[8] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[2][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state[9] = pos_contact0[0][0]
            x_state[10] = pos_contact0[0][1]
            x_state[11] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[3][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state[12] = pos_contact0[0][0]
            x_state[13] = pos_contact0[0][1]
            x_state[14] = pos_contact0[0][2]


        if tacperception.is_finger_contact(sim, hand_param[4][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            print('x_state ', x_state)
            x_state[15] = pos_contact0[0][0]
            x_state[16] = pos_contact0[0][1]
            x_state[17] = pos_contact0[0][2]

        return x_state

    def augmented_state(self, sim, model, hand_param, tacperception, x_state):
        if tacperception.is_finger_contact(sim, hand_param[1][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[1][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[2][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[2][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[3][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[3][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])

        if tacperception.is_finger_contact(sim, hand_param[4][0]) == True:
            c_point_name0 = tacperception.get_contact_taxel_name(sim, model, hand_param[4][0])
            pos_contact0 = ug.get_relative_posquat(sim, "cup", c_point_name0)[:3] + np.random.normal(0, 0.00, size=(1, 3))
            x_state = np.append(x_state, [pos_contact0])
        else:
            x_state = np.append(x_state, [0, 0, 0])
        return x_state


    def interaction(self, sim, model, viewer, hand_param, object_param, alg_param, \
                    ekf_grasping, tacperception):
        global first_contact_flag, x_all, gd_all, ff_first_contact_flag, \
        mf_first_contact_flag, rf_first_contact_flag, th_first_contact_flag
        last_c_palm = np.zeros(6)
        u_all = np.zeros(16)
        ju_all = np.zeros(24)
        # number of triggered fingers
        tacperception.fin_num = 0
        # The fingers which are triggered are Marked them with "1"
        tacperception.fin_tri = np.zeros(4)

        # Thumb root movement
        self.pre_thumb(sim, viewer)
        # other fingers start moving with the different velocity (contact/without contact)
        for ii in range(1000):
            if hand_param[1][1] == '1':
                self.index_finger(sim, 0.0055, 0.00001)
            if hand_param[2][1] == '1':
                self.middle_finger(sim, 0.0016, 0.00001)
            if hand_param[3][1] == '1':
                self.ring_finger(sim, 0.02, 0.00001)
            if hand_param[4][1] == '1':
                self.thumb(sim, 0.0003, 0.00001)

            start = time.time()
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
                # detect the first contact and initialize y_t_update with noise
                if not first_contact_flag:
                    # initialize the co-variance matrix of state estimation
                    P_state_cov = 0.01 * np.identity(6 + 4 * 3)
                    # P_state_cov = 100 * np.ones([18, 18])
                    # noise +-5 mm, +-0.002 (axis angle vector)
                    # prepare object pose and relevant noise
                    init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), \
                                                          (1, 3)), \
                                        np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))

                    x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                    # attention, here orientation we use the axis angle representation.
                    x_state = np.array([ug.pos_quat2axis_angle(x_state)])
                    if tactile_allegro_mujo_const.initE_FLAG:
                        x_state += init_e
                    # augmented state with the contact position on the object surface described in the object frame
                    x_state = self.augmented_state(sim, model, hand_param, tacperception, x_state)
                    print('x_state from beginning ', x_state)

                    x_all = x_state
                    gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
                    gd_state = ug.posquat2posrotvec(gd_posquat)
                    # gd_state = qg.posquat2posrotvec(gd_posquat)
                    gd_all = gd_state
                    first_contact_flag = True
                    if tacperception.is_ff_contact == True:
                        ff_first_contact_flag =True
                    if tacperception.is_mf_contact == True:
                        mf_first_contact_flag =True
                    if tacperception.is_rf_contact == True:
                        rf_first_contact_flag =True
                    if tacperception.is_th_contact == True:
                        th_first_contact_flag =True

                    #todo can not init last_angle with zeros because when the object
                    # is contacted, the fingers are not in all zero state.
                    last_angles = self.get_cur_jnt(sim)
                    # last_angles = np.zeros(tactile_allegro_mujo_const.FULL_FINGER_JNTS_NUM)

                    continue  # pass the first round
                elif ((flag_ff == True) and (ff_first_contact_flag)== False) :
                    x_state = np.ravel(x_state)
                    x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                    ff_first_contact_flag =True
                elif ((flag_mf == True) and (mf_first_contact_flag)== False) :
                    x_state = np.ravel(x_state)
                    x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                    mf_first_contact_flag =True
                elif ((flag_rf == True) and (rf_first_contact_flag)== False) :
                    x_state = np.ravel(x_state)
                    x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                    rf_first_contact_flag =True
                elif ((flag_th == True) and (th_first_contact_flag)== False) :
                    x_state = np.ravel(x_state)
                    x_state = self.update_augmented_state(sim, model, hand_param, tacperception, x_state)
                    th_first_contact_flag =True
                else:
                    print('no else')


                x_state = np.ravel(x_state)
                gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
                gd_state = ug.posquat2posrotvec(gd_posquat)
                # gd_state = qg.posquat2posrotvec(gd_posquat)
                """ Prediction step in EKF """
                # todo can not use ground truth update the state at every step
                # x_state[:6] = gd_state
                cur_angles = self.get_cur_jnt(sim)

                x_bar, P_state_cov, ju_all = ekf_grasping.state_predictor(sim, model, hand_param, object_param, \
                                                                  x_state, tacperception, P_state_cov, cur_angles, last_angles,self)
                last_angles = cur_angles
                # print("+++xbar, xstate:", x_bar, "\n>>", x_state, "\n>>", x_bar[:6]-x_state[:6])
                self.x_bar_all = np.vstack((self.x_bar_all, x_bar[0:6]))
                self.x_gt = np.vstack((self.x_gt, gd_state))
                """ Save to txt """
                np.savetxt('x_bar_all.txt', self.x_bar_all)
                np.savetxt('x_gt.txt', self.x_gt)
                #
                h_t_position, h_t_nv = ekf_grasping.observe_computation(x_bar, tacperception, sim)
                #
                z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, \
                                                           x_bar, tacperception)
                #
                z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
                h_t = np.concatenate((h_t_position, h_t_nv), axis=None)
                print("++++z_t:", z_t)
                print("++++h_t:", h_t)
                """ z_t and h_t visualization """
                posquat_palm_world = ug.get_relative_posquat(sim, "world", "palm_link")
                T_palm_world = ug.posquat2trans(posquat_palm_world)

                end1 = time.time()
                print('time cost in forward compute ', end1 - start)

                for i in range(4):
                    if tacperception.fin_tri[i] == 1:
                        pos_zt_palm = z_t[3*i:3*i+3]
                        pos_zt_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_zt_palm.T)
                        pos_zt_world = np.ravel(pos_zt_world.T)
                        rot_zt_palm = ug.vec2rot(z_t[3*i+12:3*i+15])
                        rot_zt_world = np.matmul(T_palm_world[:3, :3], rot_zt_palm)
                        #visualize coordinate frame of the global, palm
                        # viz.cor_frame_visual(viewer, T_palm_world[:3, 3], T_palm_world[:3, :3], 0.3, "Palm")
                        # viz.cor_frame_visual(viewer, [0, 0, 0], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1, "Global")
                        viz.geo_visual(viewer, pos_zt_world, rot_zt_world, 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "z")
                        #draw linear vel of contact point (part of twist from ju)
                        #from vel generate frame
                        vel_frame = ug.vec2rot(np.matmul(T_palm_world[:3, :3], ju_all[6*i: 6*i+3]))
                        print('vel_frame determinant ', np.linalg.det(vel_frame))
                        viz.geo_visual(viewer, pos_zt_world, vel_frame, 0.1, tactile_allegro_mujo_const.GEOM_ARROW, i, "z_vel")

                        self.ct_p_z_position = np.vstack((self.ct_p_z_position, pos_zt_palm))
                        self.ct_g_z_position = np.vstack((self.ct_g_z_position, pos_zt_world))
                        np.savetxt('ct_g_z_position.txt', self.ct_g_z_position)
                        np.savetxt('ct_p_z_position.txt', self.ct_p_z_position)

                for i in range(4):
                    if tacperception.fin_tri[i] == 1:
                        pos_ht_palm = h_t[3 * i:3 * i + 3]
                        pos_ht_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_ht_palm.T)
                        pos_ht_world = np.ravel(pos_ht_world.T)
                        rot_ht_palm = ug.vec2rot(h_t[3 * i + 12:3 * i + 15])
                        rot_ht_world = np.matmul(T_palm_world[:3, :3], rot_ht_palm)
                        # rot_h = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                        # viz.geo_visual(viewer, pos_ht_world, rot_ht_world, 0.1, tactile_allegro_mujo_const.GEOM_ARROW)
                        viz.geo_visual(viewer, pos_ht_world, rot_ht_world, 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "h")
                        # viewer.add_marker(pos=pos_ht_world, mat=rot_ht_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                        #           label="h", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
                """ GD Visualization """
                posquat_obj_world = ug.get_relative_posquat(sim, "world", "cup")
                T_obj_world = ug.posquat2trans(posquat_obj_world)
                pos_obj_world = T_obj_world[:3, 3].T
                rot_obj_world = T_obj_world[:3, :3]
                # viz.cor_frame_visual(viewer, pos_obj_world, rot_obj_world, 0.3, "Obj")
                # viewer.add_marker(pos=pos_obj_world, mat=rot_obj_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                #                   label="obj", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
                """ x_state Visualization """
                pos_x_world = (T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], x_state[:3].T)).T
                rot_x_palm = Rotation.from_rotvec(x_state[3:6]).as_matrix()
                rot_x_world = np.matmul(T_palm_world[:3, :3], rot_x_palm)
                # viewer.add_marker(pos=pos_x_world, mat=rot_x_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                #                   label="x_state", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))

                # # posterior estimation
                if tactile_allegro_mujo_const.posteriori_FLAG:
                    x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, \
                                                                   P_state_cov, tacperception)
                else:
                    x_state = x_bar
                print('x_state posterior ', x_state)
                tacperception.fin_num = 0
                tacperception.fin_tri = np.zeros(4)

                """ Save data in one loop """
                # print("   ???GD:\n", gd_state[:6])
                # print("   ???x_state:\n", x_state[:6])
                x_all = np.vstack((x_all, x_state))
                gd_all = np.vstack((gd_all, gd_state))
                """ Save to txt """
                np.savetxt('../x_data.txt', x_all)
                np.savetxt('../gd_data.txt', gd_all)
                """ Visualization h_t and z_t """
            end = time.time()
            print('time cost in one loop ', end - start)
            # if not np.all(sim.data.sensordata == 0):
            #     viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
            sim.step()
            viewer.render()
            del viewer._markers[:]


first_contact_flag = False
ff_first_contact_flag = False
mf_first_contact_flag = False
rf_first_contact_flag = False
th_first_contact_flag = False

gd_all = np.empty(0)

