import math
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import fcl
import fcl_python
import func as f
import func2 as f2
import storeQR as sQR
import surfaceFitting as sf
import tactile_allegro_mujo_const
import test_Plot_plus as plt_plus

#used for configure parameters
import sys
from xml.dom.minidom import parseString


def read_xml(xml_file):
        with open(xml_file, 'r') as f:
                data = f.read()
        return parseString(data)

if (len(sys.argv) < 2):
        print ("Error: Missing parameter.")
else:
        dom = read_xml(sys.argv[1])
        hand_name = dom.getElementsByTagName('name')[0].firstChild.data
        hand_param = []
        hand_param.append(hand_name)

        #parse fingers' parameters
        #the parameters will be organized in list type in the following way
        #['allegro', ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}]]
        fingers = dom.getElementsByTagName('finger')
        for finger in fingers:
                finger_name = finger.getAttribute("name")
                is_used = finger.getElementsByTagName("used")[0]
                print(is_used.firstChild.data)
                js = finger.getElementsByTagName('init_posture')
                for jnt in js:
                        j_init_dic = {
                            "j1":jnt.getElementsByTagName("j1")[0].firstChild.data,
                            "j2":jnt.getElementsByTagName("j2")[0].firstChild.data,
                            "j3":jnt.getElementsByTagName("j3")[0].firstChild.data,
                            "j4":jnt.getElementsByTagName("j4")[0].firstChild.data
                        }
                finger_param = [finger_name, is_used.firstChild.data, j_init_dic]
                hand_param.append(finger_param)
        print(hand_param)
        print(hand_param[1][1])
        print(hand_param[2][1])
        print(hand_param[3][1])
        print(hand_param[4][1])
        #access to data in configure file
        #hand_param[0]: name of the hand
        #hand_param[1]: parameter of "ff" finger
        #hand_param[1][0]: name of "ff" finger
        #hand_param[1][1]: is "ff" finger used for grasping
        #hand_param[1][2]["j1"]: init of j1 of "ff" finger

        #parse object info
        object_param = []
        object_name = dom.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        object_position_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_position')[0].firstChild.data
        object_orientation_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_orientation')[0].firstChild.data
        object_param.append(object_name)
        object_param.append(object_position_noise)
        object_param.append(object_orientation_noise)
        print(object_param)

######################################
# v3b: Gaussian noise add to h() to become z_t
#
#
#######################################
#########################################   GLOBAL VARIABLES   #########################################################
xml_path = "../../robots/UR5_tactile_allegro_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

pose_cup = f.get_body_posquat(sim, object_param[0])
trans_cup = f.posquat2trans(pose_cup)
trans_pregrasp = np.array([[0, 0, 1, 0.1],  # cup参考系
                           [0, 1, 0, -0.23],
                           [-1, 0, 0, 0.07],
                           [0, 0, 0, 1]])
posequat = f.get_prepose_posequat(trans_cup, trans_pregrasp)  # 转为世界参考系
print("INIT:", posequat)
ctrl_wrist_pos = posequat[:3]
ctrl_wrist_quat = posequat[3:]
max_size = 0
flag = False  # 首次接触判断flag，False表示尚未锁定第一个接触点
c_point_name = ""
q_pos_pre = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],\
                      sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
y_t = np.array([0, 0, 0, 0, 0, 0])
n_o = np.array([0, 0, 0, 0, 0, 0])
pos_contact1 = np.array([0, 0, 0])
pos_contact_last = np.array([0, 0, 0])
S1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
y_t_pre = np.array([0, 0, 0, 0, 0, 0])
posquat_contact_pre = np.array([0, 0, 0, 0, 0, 0, 0])
surface_x = np.empty([1, 0])
surface_y = np.empty([1, 0])
surface_z = np.empty([1, 0])
count_time = 0  # 时间记录
# nor_in_p = np.zeros(3)
# G_big = np.zeros([6, 6])
save_model = 0
P = np.eye(6)
u_t = np.empty([1, 4])
# fin_num = 0
trans_palm2cup = f.posquat2trans(f.get_relative_posquat(sim, "cup", "palm_link"))
trans_cup2palm = f.posquat2trans(f.get_relative_posquat(sim, "palm_link", "cup"))
z0 = np.zeros(3)
z1 = np.zeros(3)
z2 = np.zeros(3)
z3 = np.zeros(3)
Pmodel = 1
conver_rate = 40

# 以下为EKF后验过程的变量
P_ori = 1000 * np.ones([22, 22])

y_t_update = np.array([np.zeros(10)])

# kinematic chain for all fingers
robot = URDF.from_xml_file('../../robots/UR5_allegro_hand_right.urdf')
#first finger
kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
#middle finger
kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
#ring finger
kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
#thumb
kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")
kdl_tree = kdl_tree_from_urdf_model(robot)

# 记录用变量
save_count_time = np.array([0])
save_pose_y_t_xyz = np.array([0, 0, 0])
save_pose_y_t_rpy = np.array([0, 0, 0])
save_pose_GD_xyz = np.array([0, 0, 0])
save_pose_GD_rpy = np.array([0, 0, 0])
save_error_xyz = np.array([0, 0, 0])
save_error_rpy = np.array([0, 0, 0])

# fcl库加载cup 的 BVH模型
obj_cup = fcl_python.OBJ("cup_1.obj")
verts_cup = obj_cup.get_vertices()
tris_cup = obj_cup.get_faces()

# Create mesh geometry
mesh_cup = fcl.BVHModel()
mesh_cup.beginModel(len(verts_cup), len(tris_cup))
mesh_cup.addSubModel(verts_cup, tris_cup)
mesh_cup.endModel()
print("len_verts_cup:", len(verts_cup))

# fcl库加载finger_tip 的 BVH模型
obj_fingertip = fcl_python.OBJ("fingertip_part.obj")
verts_fingertip = obj_fingertip.get_vertices()
tris_fingertip = obj_fingertip.get_faces()
print("len_verts_fingertip:", len(verts_fingertip))
print("len_tris_fingertip:", len(tris_fingertip))

mesh_fingertip = fcl.BVHModel()
mesh_fingertip.beginModel(len(verts_fingertip), len(tris_fingertip))
mesh_fingertip.addSubModel(verts_fingertip, tris_fingertip)
mesh_fingertip.endModel()

err_all = np.zeros(6)
err = np.zeros(6)

########################################   FUNCTIONS DEFINITION   ######################################################
# 触觉点可视化
def touch_visual(a):
    global max_size
    truth = f.get_relative_posquat(sim, "base_link", "cup")

    save_point_use = np.array([[0, 0, 0, 0, 0, 0, 0]])
    save_point_use = np.append(save_point_use, np.array([truth]), axis=0)
    for i in a:
        for k, l in enumerate(i):
            s_name = model._sensor_id2name[i[k]]
            sensor_pose = f.get_body_posquat(sim, s_name)
            relative_pose = f.get_relative_posquat(sim, "base_link", s_name)
            save_point_use = np.append(save_point_use, np.array([relative_pose]), axis=0)

            rot_sensor = f.as_matrix(np.hstack((sensor_pose[4:], sensor_pose[3])))
            # 用于控制方向，触觉传感器的方向问题
            test_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            viewer.add_marker(pos=sensor_pose[:3], mat=test_rot, type=const.GEOM_ARROW, label="contact",
                              size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

    if save_point_use.shape[0] > max_size:
        save_point_output = save_point_use
        np.save("output.npy", save_point_output)
        where_a = np.where(np.array(sim.data.sensordata) > 0.0)

    max_size = max(save_point_use.shape[0], max_size)
    viewer.render()


def ekf_predictor(sim, model, viewer, y_t_update):
    global flag, q_pos_pre, c_point_name, y_t, pos_contact_last, S1, y_t_pre, posquat_contact_pre
    global trans_palm2cup, nor_in_p, G_big, q_pos_pre, P
    global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
    global save_error_xyz, save_error_rpy, save_model
    global u_t, trans_cup2palm, conver_rate

    # print("*" * 30)
    ##############   refresh parameters  ###################
    delta_t = 0.1
    fin_num = 0  # number of triggered fingers
    fin_tri = np.zeros(4)  # Which fingers are triggered? Mark them with "1"

    G_pinv = np.zeros([6, 6])
    G_big = np.zeros([6, 6])
    J_finger_3 = np.zeros([6, 4])
    nor_in_p = np.zeros(3)

    G_contact = np.zeros([4, 6, 6])  # store G, the number of G is uncertain, this number between 1~4
    J = np.zeros([4, 6, 4])  # store J, the number of J is uncertain, this number between 1~4
    u_t_tmp = np.zeros([4, 4])  # store u_t, the number of u_t is uncertain, this number between 1~4
    nor_tmp = np.zeros([4, 3])  # store normal, the number of normal is uncertain, this number between 1~4
    coe_tmp = np.zeros([4, 6])  # store coefficients, the number of coefficients is uncertain, this number between 1~4

    y_t_update = np.ravel(y_t_update)  # flatten
    if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any() \
            or (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any() \
            or (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any() \
            or (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN:\
            tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any():
        if not flag:  # get the state of cup in the first round
            init_e = np.hstack((np.random.uniform(-0.005, 0.005, (1, 3)), np.random.uniform(-0.08, 0.08, (1, 3))))  # +-5 mm, +-0.08 rad(4.5 deg)
            # init_e = np.array([-0.005, -0.005, -0.005, -0.08, -0.08, -0.08])
            # init_e = np.array([0.005, 0.005, 0.005, 0.08, 0.08, 0.08])
            # init_e = np.array([-0.0012456, 0.00474746, -0.00015647, -0.06027362, 0.02768316, -0.01743691])
            y_t_update = f.get_relative_posquat(sim, "palm_link", "cup")  # x y z w
            y_t_update = np.array([f.pos_quat2pos_XYZ_RPY_xyzw(y_t_update)])
            y_t_update += init_e
            flag = True
        y_t_update = np.ravel(y_t_update)

        ################################################################################################################
        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J (index finger)">
        # todo I guess here you can use a unified function which different parameters to
        # compute Grasp matrix etc.
        if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any():
            a = np.where(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX] > 0.0)  # The No. of tactile sensor (index finger)
            c_points0 = a[0]
            print("fin0")
            #todo the pos_contact0 should be the mean value of all actived taxels.
            c_point_name0 = f2.get_c_point_name(model, c_points0)
            pos_contact0 = f.get_relative_posquat(sim, "palm_link", c_point_name0)[:3]  # get the position

            nor0, res0 = f2.get_normal(sim, model, c_points0, trans_cup2palm)  # get normal_in_cup
            nor_tmp[fin_num] = nor0  # save to tmp nor
            coe_tmp[fin_num] = res0  # save to tmp coe

            #the G is the estimated matrix because the noised object pose
            G_contact0 = f2.get_G(sim, c_point_name0, pos_contact0, y_t_update)  # get G
            G_contact[fin_num] = G_contact0  # save to tmp G

            # Get joint angle velocity--u_t
            u_t0 = np.array([sim.data.qvel[tactile_allegro_mujo_const.FF_MEA_1],\
                             sim.data.qvel[tactile_allegro_mujo_const.FF_MEA_2], \
                             sim.data.qvel[tactile_allegro_mujo_const.FF_MEA_3], \
                             sim.data.qvel[tactile_allegro_mujo_const.FF_MEA_4]])
            u_t_tmp[fin_num] = u_t0  # save to tmp u_t

            J0 = kdl_kin0.jacobian(u_t0)  # Get Jacobi J
            J[fin_num] = J0  # save to tmp J

            fin_num += 1
            fin_tri[0] = 1
        # </editor-fold>

        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J  (middle finger)">
        if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any():
            a1 = np.where(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX] > 0.0)  # The No. of tactile sensor (middle finger)
            c_points1 = a1[0] + 144
            print("fin1")
            c_point_name1 = f2.get_c_point_name(model, c_points1)

            pos_contact1 = f.get_relative_posquat(sim, "palm_link", c_point_name1)[:3]  # get the position

            nor1, res1 = f2.get_normal(sim, model, c_points1, trans_cup2palm)  # get normal_in_cup
            nor_tmp[fin_num] = nor1  # save to tmp nor
            coe_tmp[fin_num] = res1  # save to tmp coe

            G_contact1 = f2.get_G(sim, c_point_name1, pos_contact1, y_t_update)  # get G
            G_contact[fin_num] = G_contact1  # save to tmp G

            u_t1 = np.array([sim.data.qvel[274], sim.data.qvel[275], sim.data.qvel[312], sim.data.qvel[349]])  # Get u_t
            u_t_tmp[fin_num] = u_t1  # save to tmp u_t

            J1 = kdl_kin1.jacobian(u_t1)  # Get Jacobi J
            J[fin_num] = J1  # save to tmp J

            fin_num += 1
            fin_tri[1] = 1
        # </editor-fold>

        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J (little finger)">
        if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any():
            a2 = np.where(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX] > 0.0)  # The No. of tactile sensor (middle finger)
            c_points2 = a2[0] + 288
            print("fin2")
            c_point_name2 = f2.get_c_point_name(model, c_points2)

            pos_contact2 = f.get_relative_posquat(sim, "palm_link", c_point_name2)[:3]  # get the position

            nor2, res2 = f2.get_normal(sim, model, c_points2, trans_cup2palm)  # get normal_in_cup
            nor_tmp[fin_num] = nor2  # save to tmp nor
            coe_tmp[fin_num] = res2  # save to tmp coe

            G_contact2 = f2.get_G(sim, c_point_name2, pos_contact2, y_t_update)  # get G
            G_contact[fin_num] = G_contact2  # save to tmp G

            u_t2 = np.array([sim.data.qvel[422], sim.data.qvel[423], sim.data.qvel[460], sim.data.qvel[497]])  # Get u_t
            u_t_tmp[fin_num] = u_t2  # save to tmp u_t

            J2 = kdl_kin2.jacobian(u_t2)  # Get Jacobi J
            J[fin_num] = J2  # save to tmp J

            fin_num += 1
            fin_tri[2] = 1
        # </editor-fold>

        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J (thumb)">
        if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any():
            a3 = np.where(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX] > 0.0)  # The No. of tactile sensor (middle finger)
            c_points3 = a3[0] + 432
            print("fin3")
            c_point_name3 = f2.get_c_point_name(model, c_points3)

            pos_contact3 = f.get_relative_posquat(sim, "palm_link", c_point_name3)[:3]  # get the position

            nor3, res3 = f2.get_normal(sim, model, c_points3, trans_cup2palm)  # get normal_in_cup
            nor_tmp[fin_num] = nor3  # save to tmp nor
            coe_tmp[fin_num] = res3  # save to tmp coe

            G_contact3 = f2.get_G(sim, c_point_name3, pos_contact3, y_t_update)  # get G
            G_contact[fin_num] = G_contact3  # save to tmp G

            u_t3 = np.array([sim.data.qvel[570], sim.data.qvel[571], sim.data.qvel[572], sim.data.qvel[573]])  # Get u_t
            u_t_tmp[fin_num] = u_t3  # save to tmp u_t

            J3 = kdl_kin3.jacobian(u_t3)  # Get Jacobi J
            J[fin_num] = J3  # save to tmp J

            fin_num += 1
            fin_tri[3] = 1
        # </editor-fold>
        ################################################################################################################
        ########## Splice Big G ##########
        G_big = np.zeros([6 * fin_num, 6])  # dim: 6n * 6
        for i in range(fin_num):
            G_big[0 + i * 6: 6 + i * 6, :] = G_contact[i]
        # print("CHek:", G_big)

        ############### Splice Big J #################
        J_finger_3 = np.zeros([6 * fin_num, 4 * fin_num])
        for i in range(fin_num):
            J_finger_3[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = J[i]

        ############# Splice Big u_t #################
        u_t = np.zeros(4 * fin_num)
        for i in range(fin_num):
            u_t[0 + i * 4: 4 + i * 4] = u_t_tmp[i]

        ########## Splice Big normal_in_palm #########
        nor_in_p = np.zeros(3 * fin_num)
        for i in range(fin_num):
            nor_in_p[0 + i * 3: 3 + i * 3] = nor_tmp[i]

        G_pinv = np.linalg.pinv(G_big)  # Get G_pinv
        #todo could you please comment the formula the computation,
        # e.g. give the reference (to paper/book, pages, eq id)?
        prediction = np.matmul(np.matmul(G_pinv, J_finger_3), u_t * delta_t)  # Predict

        ###############################
        y_t = y_t_update + prediction
        #todo what does revel here?
        y_t = np.ravel(y_t)
        ###############################

        # cup真值的获取
        ground_truth = np.array([f.pos_quat2pos_XYZ_RPY_xyzw(f.get_relative_posquat(sim, "palm_link", "cup"))])
        error = np.abs(y_t - ground_truth)

        count_time += 1
        # Save
        save_count_time = np.hstack((save_count_time, count_time))
        save_pose_y_t_xyz = np.vstack((save_pose_y_t_xyz, np.array(y_t[:3])))
        save_pose_y_t_rpy = np.vstack((save_pose_y_t_rpy, np.array(y_t[3:])))
        save_pose_GD_xyz = np.vstack((save_pose_GD_xyz, np.array(ground_truth[0, :3])))
        save_pose_GD_rpy = np.vstack((save_pose_GD_rpy, np.array(ground_truth[0, 3:])))
        save_error_xyz = np.vstack((save_error_xyz, np.array(error[0, :3])))
        save_error_rpy = np.vstack((save_error_rpy, np.array(error[0, 3:])))
        # if save_model == 0:
        #     np.save("save_date/WithoutInitE/save_f/save_count_time.npy", save_count_time)
        #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
        #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
        #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_xyz.npy", save_pose_GD_xyz)
        #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_rpy.npy", save_pose_GD_rpy)
        #     np.save("save_date/WithoutInitE/save_f/save_error_xyz.npy", save_error_xyz)
        #     np.save("save_date/WithoutInitE/save_f/save_error_rpy.npy", save_error_rpy)
        if save_model == 1:
            np.save("save_i/save_count_time.npy", save_count_time)
            np.save("save_i/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
            np.save("save_i/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
            np.save("save_i/save_pose_GD_xyz.npy", save_pose_GD_xyz)
            np.save("save_i/save_pose_GD_rpy.npy", save_pose_GD_rpy)
            np.save("save_i/save_error_xyz.npy", save_error_xyz)
            np.save("save_i/save_error_rpy.npy", save_error_rpy)

    return nor_in_p, G_big, y_t, u_t, np.matmul(G_pinv, J_finger_3) * delta_t, fin_num, fin_tri, coe_tmp, err


def ekf_posteriori(sim, model, viewer, z_t, h_t, G_big, y_t, F_part, fin_num, fin_tri, coe, err):
    global P_ori, y_t_update, err_all
    global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time

    #todo add reference to equation paper/book pages, eq id
    #######################################    F    ###########################################
    F = np.eye(6 + 4 * fin_num)
    F[:6, 6:] = F_part

    #######################################    R    ###########################################
    #todo where does it come?
    R_tmp = np.array([[2.37024061e-001, -3.06343629e-001, 6.14938815e-001, 1.17960413e-001,
                       3.05049539e-001, -1.68124732e-003],
                      [-3.06343629e-001, 6.93113033e-001, -9.19982570e-001, -3.85185662e-001,
                       -9.70974182e-002, 1.42353856e-001],
                      [6.14938815e-001, -9.19982570e-001, 2.07220530e+000, 7.14798278e-001,
                       1.04838975e+000, -1.37061610e-001],
                      [1.17960413e-001, -3.85185662e-001, 7.14798278e-001, 1.36401953e+000,
                       4.32821898e-002, 1.60474548e-001],
                      [3.05049539e-001, -9.70974182e-002, 1.04838975e+000, 4.32821898e-002,
                       1.29434881e+000, -7.21019125e-002],
                      [-1.68124732e-003, 1.42353856e-001, -1.37061610e-001, 1.60474548e-001,
                       -7.21019125e-002, 5.31751383e-001]])
    R_ori = np.zeros([12, 12])
    R_ori[:6, :6] = R_tmp
    R = R_ori[:3 * fin_num, :3 * fin_num]

    #######################################    Q    ###########################################
    Q_ori = sQR.get_Qt()
    Q = Q_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]

    #######################################    H    ###########################################
    H_t = np.zeros([3 * fin_num, 6 + 4 * fin_num])
    H_t_tmp2 = np.linalg.pinv(G_big[:3].T)
    for i in range(fin_num):
        H_t_cup = np.array([[2 * coe[i][0], coe[i][1], 0],
                            [coe[i][1], 2 * coe[i][2], 0],
                            [0, 0, 0]])
        H_t_tmp1 = np.matmul(trans_cup2palm[:3, :3], H_t_cup)
        H_t[0 + 3 * i: 3 + 3 * i, :6] = np.matmul(H_t_tmp1, H_t_tmp2)

    #######################################    P    ###########################################
    P_tmp = P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]
    P_pre = np.matmul(np.matmul(F, P_tmp), F.T) + Q  # P:{6+4n}*{6+4n}  Q:{6+4n}*{6+4n}
    ############################################################################################

    error = z_t - h_t
    ###################################    Estimate    ########################################
    K_t = f.mul_three(P_pre, H_t.T, np.linalg.pinv(f.mul_three(H_t, P_pre, H_t.T) + R))  # K:{6+4n}*3n
    print("K_t", K_t.shape)
    P_t = np.matmul((np.eye(6 + 4 * fin_num) - np.matmul(K_t, H_t)), P_pre)  # P_t in current round
    P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num] = P_t  # update to P_original
    y_t = np.ravel(y_t)  # flatten
    err = err_all[0]
    err_all = err_all[1:]
    err = np.ravel(err)

    # y_t_update = y_t + np.matmul(K_t, error.T).T
    y_t_update = y_t + np.hstack((err, np.zeros(4 * fin_num)))

    return y_t_update


def EKF():
    global y_t_update, save_model, trans_cup2palm, Pmodel, err_all
    # save_model = 0
    save_model = 1

    trans_cup2palm = f.posquat2trans(f.get_relative_posquat(sim, "palm_link", "cup"))  # update T_cup2palm

    # Forward Predictor
    h_t, G_contact, y_t_ready, u_t, F_part, fin_num, fin_tri, coe, err = ekf_predictor(sim, model, viewer, y_t_update)
    #######################   Frame Loss: contact loss or get nan_normal   ############################
    if fin_num == 2:
        Pmodel = 2

    if fin_num != Pmodel or np.isnan(np.sum(h_t)):
        # if fin_num ==0 or np.isnan(np.sum(h_t)) or (fin_tri[0] == 0 and fin_tri[1] == 1):
        y_t_update = y_t_ready
        # print("pass")
        return 0
    ###################################################################################################

    y_t_ready = np.ravel(y_t_ready)
    u_t = np.ravel(u_t)
    y_t = np.hstack((y_t_ready, u_t))  # splice to 6+4n
    print("!!!!!!!!y_t:", y_t)

    # FCL give out z_t
    # z_t = collision_test(fin_tri)
    z_t = h_t + np.random.uniform(-0.1, 0.1, 3 * fin_num)
    # z_t = f2.normalization(z_t)
    print("new z_t:", z_t)

    err_all = np.vstack((err_all, err))
    ######<<<<<<<<<<< Switch fEKF / iEKF <<<<<<<<<<#########
    # y_t_update = y_t
    y_t_update = ekf_posteriori(sim, model, viewer, z_t, h_t, G_contact, y_t, F_part, fin_num, fin_tri, coe, err)
    ######>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#########
    y_t_update = y_t_update[:6]  # Remove control variables

    # 步进、渲染、可视化
    for _ in range(50):
        if not np.all(sim.data.sensordata == 0):
            touch_visual(np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
    viewer.render()


def interacting(hand_param):
    global err_all
    #todo what is the file?
    err_all = np.loadtxt("./err_inHand_v3bi.txt")
    f2.pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
    for ii in range(37):
        if hand_param[1][1] == '1':
            f2.index_finger(sim, 0.015, 0.00001)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.015, 0.00001)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.015, 0.001)
        EKF()
    # Slow Downt whether any array element along a given axis evaluates to True.
    for ij in range(30):
        if hand_param[1][1] == '1':
            f2.index_finger(sim, 0.0055, 0.004)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.0036, 0.003)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.0032, 0.0029)
        if hand_param[4][1] == '1':
            f2.thumb(sim, 0.003, 0.003)
        #todo EKF() already did the rendering, why here the sim step and rendering still needed?
        for i in range(4):
            for _ in range(50):
                sim.step()
            viewer.render()
        EKF()
    # Rotate
    for ij in range(30):
        # f2.index_finger(sim, 0.0055, 0.0038)
        if hand_param[2][1] == '1':
            f2.middle_finger(sim, 0.0003, 0.003)
        if hand_param[3][1] == '1':
            f2.little_thumb(sim, 0.0005, 0.005)
        if hand_param[4][1] == '1':
            f2.thumb(sim, 0.003, 0.003)
        for i in range(4):
            for _ in range(50):
                sim.step()
            viewer.render()
        EKF()

    plt_plus.plot_error("save_i")


############################>>>>>>>>>>>>>>>    MAIN LOOP    <<<<<<<<<<<<<###############################################

f2.robot_init(sim)
f2.Camera_set(viewer, model)
sim.model.eq_active[0] = True
for i in range(50):
    for _ in range(50):
        sim.step()
    viewer.render()

for i in range(4):
    sim.data.mocap_pos[0] = ctrl_wrist_pos  # mocap控制需要用世界参考系
    sim.data.mocap_quat[0] = ctrl_wrist_quat  # mocap控制需要用世界参考系
    for _ in range(50):
        sim.step()
    viewer.render()

interacting(hand_param)

# print(np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))
#
# save_tmp = np.load("save_date/iEKF_incremental.npy")
# np.save("save_date/iEKF_incremental.npy", np.vstack((save_tmp, np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))))
# print("Run times:", int(save_tmp.shape[0] / 45 + 1))
# print("over, shape:", save_tmp.shape)
# print("cHeCK GD:", save_pose_GD_xyz.shape)
#
# np.save("save_date/iEKF_incremental.npy", np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))
