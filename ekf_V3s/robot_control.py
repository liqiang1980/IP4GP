import numpy as np
import tactile_allegro_mujo_const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import viz
import tactile_perception as tacperception
import util_geometry as ug


def robot_init(sim):
    sim.data.ctrl[0] = 0.8
    sim.data.ctrl[1] = -0.78
    sim.data.ctrl[2] = 1.13
    sim.data.ctrl[3] = -1.
    sim.data.ctrl[4] = 0
    sim.data.ctrl[5] = -0.3


def index_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[0:72]) > 0.0).any():  # 食指
        sim.data.ctrl[7] = sim.data.ctrl[7] + input1
        sim.data.ctrl[8] = sim.data.ctrl[8] + input1
        sim.data.ctrl[9] = sim.data.ctrl[9] + input1
    else:
        sim.data.ctrl[7] = sim.data.ctrl[7] + input2
        sim.data.ctrl[8] = sim.data.ctrl[8] + input2
        sim.data.ctrl[9] = sim.data.ctrl[9] + input2


def middle_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[144:216]) > 0.0).any():  # 中指
        sim.data.ctrl[11] = sim.data.ctrl[11] + input1
        sim.data.ctrl[12] = sim.data.ctrl[12] + input1
        sim.data.ctrl[13] = sim.data.ctrl[13] + input1
    else:
        sim.data.ctrl[11] = sim.data.ctrl[11] + input2
        sim.data.ctrl[12] = sim.data.ctrl[12] + input2
        sim.data.ctrl[13] = sim.data.ctrl[13] + input2


def middle_finger_vel(sim, input1, input2):
    print(sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1])
    if not (np.array(sim.data.sensordata[144:216]) > 0.0).any():  # 中指
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input1
    else:
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2
        sim.data.qvel[tactile_allegro_mujo_const.MF_MEA_1] = input2


def ring_finger(sim, input1, input2):
    if not (np.array(sim.data.sensordata[288:360]) > 0.0).any():  # 小拇指
        sim.data.ctrl[15] = sim.data.ctrl[15] + input1
        sim.data.ctrl[16] = sim.data.ctrl[16] + input1
        sim.data.ctrl[17] = sim.data.ctrl[17] + input1
    else:
        sim.data.ctrl[15] = sim.data.ctrl[15] + input2
        sim.data.ctrl[16] = sim.data.ctrl[16] + input2
        sim.data.ctrl[17] = sim.data.ctrl[17] + input2


def pre_thumb(sim, viewer):
    for _ in range(1000):
        sim.data.ctrl[18] = sim.data.ctrl[18] + 0.05
        sim.step()
        viewer.render()


def thumb(sim, input1, input2):
    if not (np.array(sim.data.sensordata[432:504]) > 0.0).any():  # da拇指
        sim.data.ctrl[19] = sim.data.ctrl[19] + input1
        sim.data.ctrl[20] = sim.data.ctrl[20] + input1
        sim.data.ctrl[21] = sim.data.ctrl[21] + input1 * 5
    else:
        sim.data.ctrl[19] = sim.data.ctrl[19] + input2
        sim.data.ctrl[20] = sim.data.ctrl[20] + input2
        sim.data.ctrl[21] = sim.data.ctrl[21] + input2 * 5


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


def interaction(sim, model, viewer, hand_param, object_param, alg_param, \
                ekf_grasping):
    global contact_flag
    # number of triggered fingers
    fin_num = 0
    # Which fingers are triggered? Mark them with "1"
    fin_tri = np.zeros(4)

    err_all = np.loadtxt("./err_inHand_v3bi.txt")
    pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
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
            # detect the first contact and initialize y_t_update with noise
            if not contact_flag:  # get the state of cup in the first round
                # noise +-5 mm, +-0.08 rad(4.5 deg)
                # prepare object pose and relevent noise
                init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]),\
                                                    (1, 3)),\
                                np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))

                x_state = ug.get_relative_posquat(sim, "palm_link", "cup")
                x_state = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(x_state)])
                x_state += init_e

                # compute the contact position on the object surface described in the object frame

                contact_flag = True
            x_state = np.ravel(x_state)

            if tacperception.is_finger_contact(sim, hand_param[1][0]) == True:
                fin_num += 1
                fin_tri[0] = 1
            if tacperception.is_finger_contact(sim, hand_param[2][0]) == True:
                fin_num += 1
                fin_tri[1] = 1
            if tacperception.is_finger_contact(sim, hand_param[3][0]) == True:
                fin_num += 1
                fin_tri[2] = 1
            if tacperception.is_finger_contact(sim, hand_param[4][0]) == True:
                fin_num += 1
                fin_tri[3] = 1

            # Prediction step in EKF
            x_bar = ekf_grasping.state_predictor(sim, model, hand_param, object_param, x_state, fin_num, fin_tri)
            # y_bar: predicted object pose
            x_bar = np.ravel(x_bar)
            # # joints' velocity
            # u_t = np.ravel(u_t)
            # # combined vector
            # x_t = np.hstack((x_bar, u_t))  # splice to 6+4n
            print("!!!!!!!!y_t:", x_bar)

            h_t = ekf_grasping.observation_computation(x_bar, fin_num)

            # FCL give out z_t
            # z_t = collision_test(fin_tri)
            # h_t: observing value from measurement equation
            z_t = h_t + np.random.uniform(-0.1, 0.1, 3 * fin_num)
            z_t = ug.normalization(z_t)
            # print("new z_t:", z_t)

            # err_all = np.vstack((err_all, err))
            # posterior estimation
            x_update = ekf_grasping.ekf_posteriori(sim, model, viewer, z_t, h_t)
            # x_state = x_update[:6]  # Remove control variables
            x_state = x_update  # Remove control variables
            fin_num = 0
            fin_tri = np.zeros(4)

        if not np.all(sim.data.sensordata == 0):
            viz.touch_visual(sim, model, viewer, np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
        viewer.render()

contact_flag = False