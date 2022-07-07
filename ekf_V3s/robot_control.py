import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import viz
# import tactile_perception as tacperception
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
                P_state_cov = 0.01 * np.identity(6 + 4 * 3)
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
                x_state = augmented_state(sim, model, hand_param, tacperception, x_state)
                __rot = Rotation.from_rotvec(x_state[3:6]).as_matrix()
                print("}}}}}rot0:", __rot)
                x_all = x_state
                gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
                gd_state = qg.posquat2posrotvec(gd_posquat)
                gd_all = gd_state
                contact_flag = True
                continue  # pass the first round

            x_state = np.ravel(x_state)
            gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
            gd_state = qg.posquat2posrotvec(gd_posquat)
            # Prediction step in EKF
            x_bar, P_state_cov, ju, delta_angles = ekf_grasping.state_predictor(sim, model, hand_param, object_param, \
                                                              x_state, tacperception, P_state_cov)
            print("+++xbar, xstate:", x_bar, "\n>>", x_state, "\n>>", x_bar[:6]-x_state[:6])
            #
            h_t_position, h_t_nv, = ekf_grasping.observe_computation(x_bar, tacperception, sim)
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
            for i in range(4):
                if tacperception.fin_tri[i] == 1:
                    pos_zt_palm = z_t[3*i:3*i+3]
                    pos_zt_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_zt_palm.T)
                    pos_zt_world = np.ravel(pos_zt_world.T)
                    rot_zt_palm = qg.vec2rot(z_t[3*i+12:3*i+15])
                    rot_zt_world = np.matmul(T_palm_world[:3, :3], rot_zt_palm)
                    # rot_z = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                    viewer.add_marker(pos=pos_zt_world, mat=rot_zt_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="z", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
            for i in range(4):
                if tacperception.fin_tri[i] == 1:
                    pos_ht_palm = h_t[3 * i:3 * i + 3]
                    pos_ht_world = T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_ht_palm.T)
                    pos_ht_world = np.ravel(pos_ht_world.T)
                    rot_ht_palm = qg.vec2rot(h_t[3 * i + 12:3 * i + 15])
                    rot_ht_world = np.matmul(T_palm_world[:3, :3], rot_ht_palm)
                    # rot_h = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                    viewer.add_marker(pos=pos_ht_world, mat=rot_ht_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="h", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            """ Test pos_c_obj in x_bar """
            pos1_c_obj = x_bar[6:9]
            posquat_obj_world = ug.get_relative_posquat(sim, "world", "cup")
            T_obj_world = ug.posquat2trans(posquat_obj_world)
            pos1_c_world = T_obj_world[:3, 3] + np.matmul(T_obj_world[:3, :3], pos1_c_obj.T)
            rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            viewer.add_marker(pos=pos1_c_world, mat=rot, type=tactile_allegro_mujo_const.GEOM_ARROW, label="xbar_c", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            """ Test cup and palm in world """
            posquat_obj_world = ug.get_relative_posquat(sim, "world", "cup")
            T_obj_world = ug.posquat2trans(posquat_obj_world)
            pos_obj_world = T_obj_world[:3, 3].T
            rot_obj_world = T_obj_world[:3, :3]
            viewer.add_marker(pos=pos_obj_world, mat=rot_obj_world, type=tactile_allegro_mujo_const.GEOM_ARROW, label="obj", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))
            posquat_p_world = ug.get_relative_posquat(sim, "world", "palm_link")
            T_p_world = ug.posquat2trans(posquat_p_world)
            pos_p_world = T_p_world[:3, 3].T
            rot_p_world = T_p_world[:3, :3]
            viewer.add_marker(pos=pos_p_world, mat=rot_p_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
                              label="palm", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))

            # # posterior estimation
            if tactile_allegro_mujo_const.posteriori_FLAG:
                x_state, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, \
                                                               P_state_cov, tacperception)
            else:
                x_state = x_bar
            tacperception.fin_num = 0
            tacperception.fin_tri = np.zeros(4)

            """ Save data in one loop """
            x_all = np.vstack((x_all, x_state))
            gd_all = np.vstack((gd_all, gd_state))
            u_all = np.vstack((u_all, delta_angles))
            ju_all = np.vstack((ju_all, ju))
            """ Save to txt """
            np.savetxt('../x_data.txt', x_all)
            np.savetxt('../gd_data.txt', gd_all)
            np.savetxt('../u_data.txt', u_all)
            np.savetxt('../ju_data.txt', ju_all)
            """ Visualization h_t and z_t """
        end = time.time()
        print('time cost in one loop ', end - start)
        # if not np.all(sim.data.sensordata == 0):
        #     viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
        viewer.render()


contact_flag = False
gd_all = np.empty(0)
x_all = np.empty(0)
