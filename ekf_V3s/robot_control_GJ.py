import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import util_geometry as ug
import time
import qgFunc as qg
import fcl
from scipy.spatial.transform import Rotation


def robot_init(sim):
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3


def index_finger(sim, input1, input2):
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


def middle_finger(sim, input1, input2):
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


def middle_finger_vel(sim, input1, input2):
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


def ring_finger(sim, input1, input2):
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


def pre_thumb(sim, viewer):
    for _ in range(1000):
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] + 0.05
        sim.step()
        viewer.render()


def thumb(sim, input1, input2):
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


def config_robot_tip_kin():
    # kinematic chain for all fingers
    robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
    # first finger
    kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
    # middle finger
    kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
    # ring finger
    kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
    # thumb
    kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
    kdl_tree = kdl_tree_from_urdf_model(robot)
    return kdl_kin0, kdl_kin1, kdl_kin2, kdl_kin3, kdl_tree


def config_robot(taxel_name):
    # kinematic chain for all fingers
    robot = URDF.from_xml_file("../../robots/allegro_hand_tactile_v1.4.urdf")
    kdl_kin = KDLKinematics(robot, "palm_link", taxel_name)
    return kdl_kin


def augmented_state(sim, model, hand_param, tacperception, x_state):
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


def interaction(sim, model, viewer, hand_param, object_param, alg_param, \
                ekf_grasping, tacperception):
    global contact_flag, x_all, gd_all
    last_c_palm = np.zeros(6)
    u_all = np.zeros(16)
    ju_all = np.zeros(24)
    # number of triggered fingers
    tacperception.fin_num = 0
    # The fingers which are triggered are Marked them with "1"
    tacperception.fin_tri = np.zeros(4)

    # Thumb root movement
    pre_thumb(sim, viewer)
    # other fingers start moving with the different velocity (contact/without contact)
    for ii in range(1000):
        if hand_param[1][1] == '1':
            index_finger(sim, 0.0055, 0.00001)
        if hand_param[2][1] == '1':
            middle_finger(sim, 0.0016, 0.00001)
        if hand_param[3][1] == '1':
            ring_finger(sim, 0.002, 0.00001)
        if hand_param[4][1] == '1':
            thumb(sim, 0.0003, 0.00001)

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
            if not contact_flag:
                # initialize the co-variance matrix of state estimation
                # P_state_cov = 0.01 * np.identity(6 + 4 * 3)
                P_state_cov = 100 * np.ones([18, 18])
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
                last_state = x_state
                x_state = augmented_state(sim, model, hand_param, tacperception, x_state)
                x_all = x_state
                gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
                gd_state = qg.posquat2posrotvec(gd_posquat)
                gd_all = gd_state
                contact_flag = True
                last_angle = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],
                                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],
                                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],
                                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4],
                                       sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_1],
                                       sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_2],
                                       sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_3],
                                       sim.data.qpos[tactile_allegro_mujo_const.MF_MEA_4],
                                       sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_1],
                                       sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_2],
                                       sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_3],
                                       sim.data.qpos[tactile_allegro_mujo_const.RF_MEA_4],
                                       sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1],
                                       sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2],
                                       sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3],
                                       sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4],
                                       ])

                c_name_ff = tacperception.get_contact_taxel_name(sim, model, 'ff')
                last_c_palm_posquat_ff = ug.get_relative_posquat(sim, "palm_link", c_name_ff)  # wxyz
                _quat_ff = np.hstack((last_c_palm_posquat_ff[4:], last_c_palm_posquat_ff[3]))
                last_c_palm_ff = np.hstack((last_c_palm_posquat_ff[:3], Rotation.from_quat(_quat_ff).as_rotvec()))

                c_name_mf = tacperception.get_contact_taxel_name(sim, model, 'mf')
                last_c_palm_posquat_mf = ug.get_relative_posquat(sim, "palm_link", c_name_mf)  # wxyz
                _quat_mf = np.hstack((last_c_palm_posquat_mf[4:], last_c_palm_posquat_mf[3]))
                last_c_palm_mf = np.hstack((last_c_palm_posquat_mf[:3], Rotation.from_quat(_quat_mf).as_rotvec()))

                c_name_rf = tacperception.get_contact_taxel_name(sim, model, 'rf')
                last_c_palm_posquat_rf = ug.get_relative_posquat(sim, "palm_link", c_name_rf)  # wxyz
                _quat_rf = np.hstack((last_c_palm_posquat_rf[4:], last_c_palm_posquat_rf[3]))
                last_c_palm_rf = np.hstack((last_c_palm_posquat_rf[:3], Rotation.from_quat(_quat_rf).as_rotvec()))

                c_name_th = tacperception.get_contact_taxel_name(sim, model, 'th')
                last_c_palm_posquat_th = ug.get_relative_posquat(sim, "palm_link", c_name_th)  # wxyz
                _quat_th = np.hstack((last_c_palm_posquat_th[4:], last_c_palm_posquat_th[3]))
                last_c_palm_th = np.hstack((last_c_palm_posquat_th[:3], Rotation.from_quat(_quat_th).as_rotvec()))
                continue  # pass the first round

            x_state = np.ravel(x_state)
            gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
            gd_state = qg.posquat2posrotvec(gd_posquat)
            # Prediction step in EKF
            # x_bar, P_state_cov, ju, angles, G_pinv = ekf_grasping.state_predictor(sim, model, hand_param, object_param,
            #                                                                       x_state, tacperception, P_state_cov,
            #                                                               last_angle)
            x_bar, P_state_cov, ju, angles, G_pinv = ekf_grasping.state_predictor(sim, model, hand_param, object_param,
                                                                                  last_state, tacperception,
                                                                                  P_state_cov,
                                                                                  last_angle)
            last_angle = angles
            print("+++xbar, xstate:", x_bar, "\n>>", x_state, "\n>>", x_bar[:6] - x_state[:6])
            #
            # h_t_position, h_t_nv, = ekf_grasping.observe_computation(x_bar, tacperception, sim)
            # #
            # z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, \
            #                                                x_bar, tacperception)
            # #
            # z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
            # h_t = np.concatenate((h_t_position, h_t_nv), axis=None)
            # print("++++z_t:", z_t)
            # print("++++h_t:", h_t)
            """ z_t and h_t visualization """
            posquat_palm_world = ug.get_relative_posquat(sim, "world", "palm_link")
            T_palm_world = ug.posquat2trans(posquat_palm_world)
            posquat_cup_palm = ug.get_relative_posquat(sim, "palm_link", "cup")
            T_obj_palm = ug.posquat2trans(posquat_cup_palm)
            # for i in range(4):
            #     if tacperception.fin_tri[i] == 1:
            #         pos_zt_palm = z_t[3 * i:3 * i + 3]
            #         pos_zt_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_zt_palm.T)
            #         pos_zt_world = np.ravel(pos_zt_world.T)
            #         rot_zt_palm = qg.vec2rot(z_t[3 * i + 12:3 * i + 15])
            #         rot_zt_world = np.matmul(T_palm_world[:3, :3], rot_zt_palm)
            #         # rot_z = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            #         viewer.add_marker(pos=pos_zt_world, mat=rot_zt_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                           label="z", size=np.array([0.001, 0.001, 0.1]),
            #                           rgba=np.array([1.0, 0.0, 0.0, 1.0]))
            # for i in range(4):
            #     if tacperception.fin_tri[i] == 1:
            #         pos_ht_palm = h_t[3 * i:3 * i + 3]
            #         print("&&&&pos_ht_palm:", pos_ht_palm)
            #         pos_ht_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_ht_palm.T)
            #         pos_ht_world = np.ravel(pos_ht_world.T)
            #         rot_ht_palm = qg.vec2rot(h_t[3 * i + 12:3 * i + 15])
            #         rot_ht_world = np.matmul(T_palm_world[:3, :3], rot_ht_palm)
            #         # rot_h = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            #         viewer.add_marker(pos=pos_ht_world, mat=rot_ht_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                           label="h", size=np.array([0.001, 0.001, 0.1]),
            #                           rgba=np.array([0.34, 0.98, 1., 1.0]))
            """ Test G_pinv """
            c1 = tacperception.get_contact_taxel_name(sim, model, 'ff')
            c2 = tacperception.get_contact_taxel_name(sim, model, 'mf')
            c3 = tacperception.get_contact_taxel_name(sim, model, 'rf')
            c4 = tacperception.get_contact_taxel_name(sim, model, 'th')
            cur_c1_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c1)  # wxyz
            _quat = np.hstack((cur_c1_palm_posquat[4:], cur_c1_palm_posquat[3]))
            cur_c1_palm_posrotvec = np.hstack((cur_c1_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            cur_c2_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c2)  # wxyz
            _quat = np.hstack((cur_c2_palm_posquat[4:], cur_c2_palm_posquat[3]))
            cur_c2_palm_posrotvec = np.hstack((cur_c2_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            cur_c3_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c3)  # wxyz
            _quat = np.hstack((cur_c3_palm_posquat[4:], cur_c3_palm_posquat[3]))
            cur_c3_palm_posrotvec = np.hstack((cur_c3_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            cur_c4_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c4)  # wxyz
            _quat = np.hstack((cur_c4_palm_posquat[4:], cur_c4_palm_posquat[3]))
            cur_c4_palm_posrotvec = np.hstack((cur_c4_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))

            test_cur = np.hstack((cur_c1_palm_posrotvec, cur_c2_palm_posrotvec))
            test_cur = np.hstack((test_cur, cur_c3_palm_posrotvec))
            test_cur = np.hstack((test_cur, cur_c4_palm_posrotvec))
            # print("test_G all c:", test_cur)

            real_prediction = np.matmul(G_pinv, test_cur)
            real_prediction[:3] = (T_obj_palm[:3, 3] + np.matmul(T_obj_palm[:3, :3], real_prediction[:3].T)).T
            real_prediction[3:6] = np.matmul(T_obj_palm[:3, :3], real_prediction[3:6].T).T
            to_vis = np.ravel(1 / tacperception.fin_num * real_prediction + last_state[:6])
            to_vis_pos_palm = to_vis[:3]
            to_vis_rot_palm = Rotation.from_rotvec(to_vis[3:]).as_matrix()
            to_vis_pos = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], to_vis_pos_palm.T)
            to_vis_rot = np.matmul(T_palm_world[:3, :3], to_vis_rot_palm)
            viewer.add_marker(pos=to_vis_pos, mat=to_vis_rot, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="test_G", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))

            last_state = ug.get_relative_posquat(sim, "palm_link", "cup")
            # attention, here orientation we use the axis angle representation.
            last_state = np.array([ug.pos_quat2axis_angle(last_state)])

            """ Test pos_c_obj in x_bar """
            # pos1_c_obj = x_bar[6:9]
            # # posquat_obj_world = ug.get_relative_posquat(sim, "world", "cup")
            # # T_obj_world = ug.posquat2trans(posquat_obj_world)
            # posquat_p_world = ug.get_relative_posquat(sim, "world", "palm_link")
            # posquat_obj_p = ug.get_relative_posquat(sim, "palm_link", "cup")
            # T_p_world = ug.posquat2trans(posquat_p_world)
            # T_obj_p = ug.posquat2trans(posquat_obj_p)
            # pos1_c_palm = T_obj_p[:3, 3] + np.matmul(T_obj_p[:3, :3], pos1_c_obj.T)
            # print("&&&&pos_c_palm:", pos1_c_palm)
            # T_obj_world = np.matmul(T_p_world, T_obj_p)
            # # pos1_c_world = T_obj_world[:3, 3] + np.matmul(T_obj_world[:3, :3], pos1_c_obj.T)
            # pos1_c_world = T_p_world[:3, 3] + np.matmul(T_p_world[:3, :3], pos1_c_palm.T)
            # rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            # viewer.add_marker(pos=pos1_c_world, mat=rot, type=tactile_allegro_mujo_const.GEOM_ARROW, label="xbar_c", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            """ Test cup and palm in world """
            posquat_obj_world = ug.get_relative_posquat(sim, "world", "cup")
            T_obj_world = ug.posquat2trans(posquat_obj_world)
            pos_obj_world = T_obj_world[:3, 3].T
            rot_obj_world = T_obj_world[:3, :3]
            viewer.add_marker(pos=pos_obj_world, mat=rot_obj_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="obj", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            # posquat_p_world = ug.get_relative_posquat(sim, "world", "palm_link")
            # posquat_obj_p = ug.get_relative_posquat(sim, "palm_link", "cup")
            # T_p_world = ug.posquat2trans(posquat_p_world)
            # T_obj_p = ug.posquat2trans(posquat_obj_p)
            # T_obj_world = np.matmul(T_p_world, T_obj_p)
            # pos_obj_world = T_obj_world[:3, 3].T
            # rot_obj_world = T_obj_world[:3, :3]
            # viewer.add_marker(pos=pos_obj_world, mat=rot_obj_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="TTobj", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            #
            # pos_p_world = T_p_world[:3, 3].T
            # rot_p_world = T_p_world[:3, :3]
            # viewer.add_marker(pos=pos_p_world, mat=rot_p_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="palm", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            """ Test ju ff (contact pose in palm frame)"""
            c_tmp = Rotation.from_rotvec(last_c_palm_ff[3:]).as_matrix()
            c_tmp = np.matmul(c_tmp, np.array([[0, 0, 1],
                                               [0, 1, 0],
                                               [1, 0, 0]]))
            last_c_palm_ff[3:] = Rotation.from_matrix(c_tmp).as_rotvec()
            ju_tmp = Rotation.from_rotvec(ju[3:6]).as_matrix()
            ju_tmp = np.matmul(ju_tmp, np.array([[0, 0, 1],
                                               [0, 1, 0],
                                               [1, 0, 0]]))
            ju[3:6] = Rotation.from_matrix(ju_tmp).as_rotvec()
            _ju_ff = last_c_palm_ff + ju[:6]
            pos_c_palm = _ju_ff[:3]
            rotvec_c_palm = _ju_ff[3:6]
            pos_c_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_c_palm.T)
            rot_c_palm = Rotation.from_rotvec(rotvec_c_palm).as_matrix()
            # rot_c_palm = np.matmul(rot_c_palm, np.array([[0, 0, 1],
            #                                              [0, 1, 0],
            #                                              [1, 0, 0]]))
            rot_c_world = np.matmul(T_palm_world[:3, :3], rot_c_palm)
            viewer.add_marker(pos=pos_c_world, mat=rot_c_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="ju_ff", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            c_name_ff = tacperception.get_contact_taxel_name(sim, model, 'ff')
            # c_name_ff = 'touch_0_6_12'
            # print(">> >>c_name_ff:", c_name_ff)
            last_c_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c_name_ff)  # wxyz
            _quat = np.hstack((last_c_palm_posquat[4:], last_c_palm_posquat[3]))  # xyzw
            last_c_palm_ff = np.hstack((last_c_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            # pos_last_c = last_c_palm_ff[:3]
            # rot_last_c = Rotation.from_rotvec(last_c_palm_ff[3:]).as_matrix()
            # pos_last_c_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_last_c)
            # rot_last_c_world = np.matmul(T_palm_world[:3, :3], rot_last_c)
            # # rot_last_c_world = np.matmul(rot_last_c_world, np.array([[0, 0, 1],
            # #                                                          [0, 1, 0],
            # #                                                          [1, 0, 0]]))
            # viewer.add_marker(pos=pos_last_c_world, mat=rot_last_c_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="last_ff", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.9, 1., 1.0]))

            """ Test ju mf (contact pose in palm frame)"""
            # _ju_mf = last_c_palm_mf + ju[6:12]
            # pos_c_palm = _ju_mf[:3]
            # rotvec_c_palm = _ju_mf[3:6]
            # pos_c_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_c_palm.T)
            # rot_c_palm = Rotation.from_rotvec(rotvec_c_palm).as_matrix()
            # rot_c_world = np.matmul(T_palm_world[:3, :3], rot_c_palm)
            # viewer.add_marker(pos=pos_c_world, mat=rot_c_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="ju_mf", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            # c_name_mf = tacperception.get_contact_taxel_name(sim, model, 'mf')
            # last_c_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c_name_mf)  # wxyz
            # _quat = np.hstack((last_c_palm_posquat[4:], last_c_palm_posquat[3]))
            # last_c_palm_mf = np.hstack((last_c_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            #
            # """ Test ju rf (contact pose in palm frame)"""
            # _ju_rf = last_c_palm_rf + ju[12:18]
            # pos_c_palm = _ju_rf[:3]
            # rotvec_c_palm = _ju_rf[3:6]
            # pos_c_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_c_palm.T)
            # rot_c_palm = Rotation.from_rotvec(rotvec_c_palm).as_matrix()
            # rot_c_world = np.matmul(T_palm_world[:3, :3], rot_c_palm)
            # viewer.add_marker(pos=pos_c_world, mat=rot_c_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="ju_rf", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            # c_name_rf = tacperception.get_contact_taxel_name(sim, model, 'rf')
            # last_c_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c_name_rf)  # wxyz
            # _quat = np.hstack((last_c_palm_posquat[4:], last_c_palm_posquat[3]))
            # last_c_palm_rf = np.hstack((last_c_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))
            #
            # """ Test ju th (contact pose in palm frame)"""
            # _ju_th = last_c_palm_th + ju[18:]
            # pos_c_palm = _ju_th[:3]
            # rotvec_c_palm = _ju_th[3:6]
            # pos_c_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_c_palm.T)
            # rot_c_palm = Rotation.from_rotvec(rotvec_c_palm).as_matrix()
            # rot_c_world = np.matmul(T_palm_world[:3, :3], rot_c_palm)
            # viewer.add_marker(pos=pos_c_world, mat=rot_c_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
            #                   label="ju_th", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            # c_name_th = tacperception.get_contact_taxel_name(sim, model, 'th')
            # last_c_palm_posquat = ug.get_relative_posquat(sim, "palm_link", c_name_th)  # wxyz
            # _quat = np.hstack((last_c_palm_posquat[4:], last_c_palm_posquat[3]))
            # last_c_palm_th = np.hstack((last_c_palm_posquat[:3], Rotation.from_quat(_quat).as_rotvec()))

            # # posterior estimation
            # if tactile_allegro_mujo_const.posteriori_FLAG:
            #     x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, \
            #                                                        P_state_cov, tacperception)
            # else:
            #     x_state = x_bar
            # tacperception.fin_num = 0
            # tacperception.fin_tri = np.zeros(4)

            # """ Save data in one loop """
            # x_all = np.vstack((x_all, x_state))
            # gd_all = np.vstack((gd_all, gd_state))
            # """ Save to txt """
            # np.savetxt('../x_data.txt', x_all)
            # np.savetxt('../gd_data.txt', gd_all)
            # """ Visualization h_t and z_t """
        end = time.time()
        print('time cost in one loop ', end - start)
        # if not np.all(sim.data.sensordata == 0):
        #     viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
        viewer.render()


contact_flag = False
gd_all = np.empty(0)
x_all = np.empty(0)
