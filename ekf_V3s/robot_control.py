import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import viz
# import tactile_perception as tacperception
import util_geometry as ug


def robot_init(sim):
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3


def index_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN:\
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
    if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN:\
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
    if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any():  # 中指
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
    else:
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2


def ring_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN:\
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
    if not (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
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


def config_robot():
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

def augmented_state(sim, model, hand_param, \
                tacperception,x_state):
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
    global contact_flag
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

        if (tacperception.is_finger_contact(sim, hand_param[1][0]) == True) \
                or (tacperception.is_finger_contact(sim, hand_param[2][0]) == True) \
                or (tacperception.is_finger_contact(sim, hand_param[3][0]) == True) \
                or (tacperception.is_finger_contact(sim, hand_param[4][0]) == True):
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
                init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]),\
                                                    (1, 3)),\
                                np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))

                x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                # attention, here orientation we use the axis angle representation.
                x_state = np.array([ug.pos_quat2axis_angle(x_state)])
                x_state += init_e
                # argumented state with the contact position on the object surface described in the object frame
                x_state = augmented_state(sim, model, hand_param, tacperception, x_state)
                contact_flag = True

            x_state = np.ravel(x_state)
            # Prediction step in EKF
            x_bar, P_state_cov = ekf_grasping.state_predictor(sim, model, hand_param, object_param, \
                                                 x_state, tacperception, P_state_cov)

            h_t_position, h_t_nv, = ekf_grasping.observe_computation(x_bar, tacperception)

            z_t_position, z_t_nv = ekf_grasping.measure_fb(sim, model, hand_param, object_param, \
                                                       x_bar, tacperception)

            z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
            h_t = np.concatenate((h_t_position, h_t_nv), axis=None)
            # posterior estimation
            x_update, P_state_cov = ekf_grasping.ekf_posteriori(sim, model, viewer, x_bar, z_t, h_t, P_state_cov, tacperception)
            tacperception.fin_num = 0
            tacperception.fin_tri = np.zeros(4)

        # if not np.all(sim.data.sensordata == 0):
        #     viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
        viewer.render()

contact_flag = False