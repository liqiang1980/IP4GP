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
import surfaceFitting as sf
import test_Plot_plus as plt_plus

######################################
# v3b: Gaussian noise add to h() to become z_t
#
#
#######################################
#########################################   GLOBAL VARIABLES   #########################################################
xml_path = "../../UR5/UR5_allegro_test.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

pose_cup = f.get_body_posquat(sim, "cup")
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
q_pos_pre = np.array([sim.data.qpos[126], sim.data.qpos[127], sim.data.qpos[164], sim.data.qpos[201]])
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
conver_rate = 30

# 以下为EKF后验过程的变量
P_ori = 1000 * np.ones([22, 22])

y_t_update = np.array([np.zeros(10)])

# 仅定义link_3.0_tip的测试用其他关节请重新定义kdl_kin
robot = URDF.from_xml_file('../../UR5/allegro_hand_tactile_right.urdf')
kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
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


def add_surface_xyz_palm(point_name):
    global surface_x, surface_y, surface_z
    posquat_contact = f.get_relative_posquat(sim, "palm_link", point_name)
    # xyz of the touched surface
    surface_x = np.append(surface_x, posquat_contact[0])
    surface_y = np.append(surface_y, posquat_contact[1])
    surface_z = np.append(surface_z, posquat_contact[2])


def add_surface_xyz_cup(point_name):
    global surface_x, surface_y, surface_z
    posquat_contact = f.get_relative_posquat(sim, "cup", point_name)
    # xyz of the touched surface
    surface_x = np.append(surface_x, posquat_contact[0])
    surface_y = np.append(surface_y, posquat_contact[1])
    surface_z = np.append(surface_z, posquat_contact[2])


def surface_forefinger_cup(cur_x, cur_y, cur_z):
    global surface_x, surface_y, surface_z
    surface_x = np.empty([1, 0])
    surface_y = np.empty([1, 0])
    surface_z = np.empty([1, 0])
    for i in range(72):
        forefinger_sensor_name = model._sensor_id2name[i]
        add_surface_xyz_cup(forefinger_sensor_name)  # 获取cup参考系下的xyz
    n = len(surface_x)
    s = sf.get_res(surface_x, surface_y, surface_z, n)  # 解得曲面方程的6个系数
    # sf.matching_3D(surface_x, surface_y, surface_z)
    return f2.get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s


def surface_forefinger_palm(cur_x, cur_y, cur_z):
    global surface_x, surface_y, surface_z
    surface_x = np.empty([1, 0])
    surface_y = np.empty([1, 0])
    surface_z = np.empty([1, 0])
    for i in range(72):
        forefinger_sensor_name = model._sensor_id2name[i]
        add_surface_xyz_palm(forefinger_sensor_name)  # 获取palm参考系下的xyz
    n = len(surface_x)
    s = sf.get_res(surface_x, surface_y, surface_z, n)  # 解得曲面方程的6个系数
    return f2.get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s


def ekf_predictor(sim, model, viewer, y_t_update):
    global flag, q_pos_pre, c_point_name, y_t, pos_contact_last, S1, y_t_pre, posquat_contact_pre
    global trans_palm2cup, nor_in_p, G_big, q_pos_pre, P
    global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
    global save_error_xyz, save_error_rpy, save_model
    global u_t, trans_cup2palm, conver_rate

    print("*" * 30)
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
    # print("yuuu:", y_t_update)
    if (np.array(sim.data.sensordata[0:72]) > 0.0).any() \
            or (np.array(sim.data.sensordata[144:216]) > 0.0).any() \
            or (np.array(sim.data.sensordata[288:360]) > 0.0).any() \
            or (np.array(sim.data.sensordata[432:504]) > 0.0).any():
        if not flag:  # get the state of cup in the first round
            init_e = np.hstack((np.random.uniform(-0.005, 0.005, (1, 3)), np.random.uniform(0.08, 0.08, (1, 3))))  # +-5 mm, +-0.08 rad(4.5 deg)
            y_t_update = f.get_relative_posquat(sim, "palm_link", "cup")  # x y z w
            y_t_update = np.array([f.pos_quat2pos_XYZ_RPY_xyzw(y_t_update)])
            y_t_update += init_e
            print("INIT_E:", init_e)
            flag = True
        y_t_update = np.ravel(y_t_update)

        ################################################################################################################
        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J (index finger)">
        if (np.array(sim.data.sensordata[0:72]) > 0.0).any():
            a = np.where(sim.data.sensordata[0:72] > 0.0)  # The No. of tactile sensor (index finger)
            c_points0 = a[0]
            print("fin0")
            c_point_name0 = f2.get_c_point_name(model, c_points0)

            pos_contact0 = f.get_relative_posquat(sim, "palm_link", c_point_name0)[:3]  # get the position

            nor0, res0 = f2.get_normal(sim, model, c_points0, trans_cup2palm)  # get normal_in_cup
            nor_tmp[fin_num] = nor0  # save to tmp nor
            coe_tmp[fin_num] = res0  # save to tmp coe

            G_contact0 = f2.get_G(sim, c_point_name0, pos_contact0, y_t_update)  # get G
            G_contact[fin_num] = G_contact0  # save to tmp G

            u_t0 = np.array([sim.data.qvel[126], sim.data.qvel[127], sim.data.qvel[164], sim.data.qvel[201]])  # Get u_t
            u_t_tmp[fin_num] = u_t0  # save to tmp u_t

            J0 = kdl_kin0.jacobian(u_t0)  # Get Jacobi J
            J[fin_num] = J0  # save to tmp J

            fin_num += 1
            fin_tri[0] = 1
        # </editor-fold>

        # <editor-fold desc=">>>>>get the contact names, normal, position, G and J  (middle finger)">
        if (np.array(sim.data.sensordata[144:216]) > 0.0).any():
            a1 = np.where(sim.data.sensordata[144:216] > 0.0)  # The No. of tactile sensor (middle finger)
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
        if (np.array(sim.data.sensordata[288:360]) > 0.0).any():
            a2 = np.where(sim.data.sensordata[288:360] > 0.0)  # The No. of tactile sensor (middle finger)
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
        if (np.array(sim.data.sensordata[432:504]) > 0.0).any():
            a3 = np.where(sim.data.sensordata[432:504] > 0.0)  # The No. of tactile sensor (middle finger)
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
        prediction = np.matmul(np.matmul(G_pinv, J_finger_3), u_t*delta_t)  # Predict

        ###############################
        y_t = y_t_update + prediction
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
        if save_model == 0:
            np.save("save_f/save_count_time.npy", save_count_time)
            np.save("save_f/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
            np.save("save_f/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
            np.save("save_f/save_pose_GD_xyz.npy", save_pose_GD_xyz)
            np.save("save_f/save_pose_GD_rpy.npy", save_pose_GD_rpy)
            np.save("save_f/save_error_xyz.npy", save_error_xyz)
            np.save("save_f/save_error_rpy.npy", save_error_rpy)
        elif save_model == 1:
            np.save("save_i/save_count_time.npy", save_count_time)
            np.save("save_i/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
            np.save("save_i/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
            np.save("save_i/save_pose_GD_xyz.npy", save_pose_GD_xyz)
            np.save("save_i/save_pose_GD_rpy.npy", save_pose_GD_rpy)
            np.save("save_i/save_error_xyz.npy", save_error_xyz)
            np.save("save_i/save_error_rpy.npy", save_error_rpy)

    return nor_in_p, G_big, y_t, u_t, np.matmul(G_pinv, J_finger_3) * delta_t, fin_num, fin_tri, coe_tmp


# fcl 碰撞结果显示
def print_collision_result(o1_name, o2_name, result):
    print('Collision between {} and {}:'.format(o1_name, o2_name))
    print('-' * 30)
    print('Collision?: {}'.format(result.is_collision))
    print('Number of contacts: {}'.format(len(result.contacts)))
    print('')


# fcl 碰撞最短距离显示
def print_distance_result(o1_name, o2_name, result):
    print('Distance between {} and {}:'.format(o1_name, o2_name))
    print('-' * 30)
    print('Distance: {}'.format(result.min_distance))
    print('Closest Points:')
    print(result.nearest_points[0])
    print(result.nearest_points[1])
    print('')


# 碰撞检测
def collision_part(pos_R_cup_global, pos_R_fingertip_global):
    global mesh_cup
    global mesh_fingertip
    R_cup_global = pos_R_cup_global[0:3, 0:3]
    pos_cup_global = pos_R_cup_global[0:3, 3] * 1000

    R_fingertip_global = pos_R_fingertip_global[0:3, 0:3]
    pos_fingertip_global = pos_R_fingertip_global[0:3, 3] * 1000

    t_cup_global = fcl.Transform(R_cup_global, pos_cup_global)

    t_fingertip_global = fcl.Transform(R_fingertip_global, pos_fingertip_global)

    o_cup = fcl.CollisionObject(mesh_cup, t_cup_global)
    o_fingertip = fcl.CollisionObject(mesh_fingertip, t_fingertip_global)

    req = fcl.CollisionRequest(enable_contact=True)
    res = fcl.CollisionResult()
    n_contacts = fcl.collide(o_cup, o_fingertip, req, res)

    return res


def collision_test(fin_tri):
    ######### Warning: the FCL library will output the normal vector triggered before， even if there is no contact.####
    z_tmp = np.zeros([4, 3])
    count = 0

    link_3_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_3.0_fcl")
    link_3_tip_trans_global = f.posquat2trans(link_3_tip_pos_global)

    link_7_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_7.0_fcl")
    link_7_tip_trans_global = f.posquat2trans(link_7_tip_pos_global)

    link_11_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_11.0_fcl")
    link_11_tip_trans_global = f.posquat2trans(link_11_tip_pos_global)

    link_15_tip_pos_global = f.get_relative_posquat(sim, "palm_link", "link_15.0_fcl")
    link_15_tip_trans_global = f.posquat2trans(link_15_tip_pos_global)

    cup_pose_global = f.get_relative_posquat(sim, "palm_link", "cup")
    cup_trans_global = f.posquat2trans(cup_pose_global)

    res0 = collision_part(cup_trans_global, link_3_tip_trans_global)
    res1 = collision_part(cup_trans_global, link_7_tip_trans_global)
    res2 = collision_part(cup_trans_global, link_11_tip_trans_global)
    res3 = collision_part(cup_trans_global, link_15_tip_trans_global)
    # output normal
    if res0.is_collision:
        contact = res0.contacts[0]
        normals = contact.normal
        print("normal0:", normals)
        z_tmp[count] = normals
        count += 1

    if res1.is_collision:
        contact = res1.contacts[0]
        normals = contact.normal
        print("normal1:", normals)
        z_tmp[count] = normals
        count += 1

    if res2.is_collision:
        contact = res2.contacts[0]
        normals = contact.normal
        print("normal2:", normals)
        z_tmp[count] = normals
        count += 1

    if res3.is_collision:
        contact = res3.contacts[0]
        normals = contact.normal
        print("normal3:", normals)
        z_tmp[count] = normals
        count += 1
    print("-" * 30)
    ############# Splice Big z #################
    z = np.zeros(int(3 * np.sum(fin_tri)))
    for i in range(int(np.sum(fin_tri))):
        z[0 + i * 3: 3 + i * 3] = z_tmp[np.where(fin_tri == 1)[0][i]]
    return z



def EKF():
    global y_t_update, save_model, trans_cup2palm, Pmodel
    save_model = 0
    # save_model = 1

    trans_cup2palm = f.posquat2trans(f.get_relative_posquat(sim, "palm_link", "cup"))  # update T_cup2palm

    # Forward Predictor
    h_t, G_contact, y_t_ready, u_t, F_part, fin_num, fin_tri, coe = ekf_predictor(sim, model, viewer, y_t_update)
    #######################   Frame Loss: contact loss or get nan_normal   ############################
    if fin_num == 2:
        Pmodel = 2

    if fin_num != Pmodel or np.isnan(np.sum(h_t)):
        # if fin_num ==0 or np.isnan(np.sum(h_t)) or (fin_tri[0] == 0 and fin_tri[1] == 1):
        y_t_update = y_t_ready
        print("pass")
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

    ######<<<<<<<<<<< Switch fEKF / iEKF <<<<<<<<<<#########
    y_t_update = y_t
    ######>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#########
    y_t_update = y_t_update[:6]  # Remove control variables

    # 步进、渲染、可视化
    for _ in range(50):
        if not np.all(sim.data.sensordata == 0):
            touch_visual(np.where(np.array(sim.data.sensordata) > 0.0))
        sim.step()
    viewer.render()


# 桌面食指平推
def index_finger_push():
    f2.pre_thumb(sim, viewer)  # Thumb root movement
    # Fast
    for ii in range(37):
        f2.index_finger(sim, 0.015, 0.00001)
        EKF()

    # Slow down
    for ij in range(23):
        f2.index_finger(sim, 0.005, 0.005)
        for i in range(1):
            for _ in range(50):
                sim.step()
            viewer.render()
        EKF()

    for ij in range(13):
        f2.index_finger(sim, 0.003, 0.004)
        f2.thumb(sim, 0.003, 0.004)
        for i in range(1):
            for _ in range(50):
                sim.step()
            viewer.render()

        EKF()

    # Slow down
    for ij in range(35):
        sim.data.ctrl[7] = sim.data.ctrl[7] + 0.002
        sim.data.ctrl[9] = sim.data.ctrl[9] + 0.008
        f2.thumb(sim, 0.005, 0.006)
        for i in range(1):
            for _ in range(50):
                sim.step()
            viewer.render()

        EKF()
    plt_plus.plot_error('save_f')  # iEKF
    # plt_plus.plot_error('save_i')  # iEKF


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

index_finger_push()
print(np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))

save_tmp = np.load("save_data/fEKF_incremental.npy")
np.save("save_data/fEKF_incremental.npy", np.vstack((save_tmp, np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))))
print("Run times:", int(save_tmp.shape[0] / 45 + 1))
print("over, shape:", save_tmp.shape)


# np.save("save_data/fEKF_incremental.npy", np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))

