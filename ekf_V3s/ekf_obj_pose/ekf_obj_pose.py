import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF

import storeQR as sQR
import tactile_allegro_mujo_const
import config_param
import util_geometry as ug
import robot_control as robcontrol
import mujoco_environment as mu_env
import ekf


hand_param, object_param, alg_param = config_param.pass_arg()
model, sim, viewer = mu_env.init_mujoco()
ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)
mu_env.config_fcl("cup_1.obj", "fingertip_part.obj")
grasping_ekf = ekf.EKF()
grasping_ekf.set_contact_flag(False)
grasping_ekf.set_store_flag(alg_param[0])

robcontrol.robot_init(sim)
mu_env.Camera_set(viewer, model)
sim.model.eq_active[0] = True

for i in range(500):
    sim.step()
    viewer.render()

sim.data.mocap_pos[0] = ctrl_wrist_pos  # mocap控制需要用世界参考系
sim.data.mocap_quat[0] = ctrl_wrist_quat  # mocap控制需要用世界参考系
for _ in range(50):
    sim.step()
    viewer.render()

robcontrol.interaction(sim, model, viewer, \
                       hand_param, object_param, alg_param, grasping_ekf)



#
# max_size = 0
# flag = False  # 首次接触判断flag，False表示尚未锁定第一个接触点
# c_point_name = ""
# q_pos_pre = np.array([sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_1],\
#                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_2],\
#                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_3],\
#                       sim.data.qpos[tactile_allegro_mujo_const.FF_MEA_4]])
# y_t = np.array([0, 0, 0, 0, 0, 0])
# n_o = np.array([0, 0, 0, 0, 0, 0])
# pos_contact1 = np.array([0, 0, 0])
# pos_contact_last = np.array([0, 0, 0])
# S1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# y_t_pre = np.array([0, 0, 0, 0, 0, 0])
# posquat_contact_pre = np.array([0, 0, 0, 0, 0, 0, 0])
# surface_x = np.empty([1, 0])
# surface_y = np.empty([1, 0])
# surface_z = np.empty([1, 0])
# count_time = 0  # 时间记录
# save_model = alg_param[0]
# P = np.eye(6)
# u_t = np.empty([1, 4])
# # number of contacted fingers
# fin_num = 0
# # Which fingers are triggered? Mark them with "1"
# fin_tri = np.zeros(4)
# trans_palm2cup = ug.posquat2trans(ug.get_relative_posquat(sim, "cup", "palm_link"))
# trans_cup2palm = ug.posquat2trans(ug.get_relative_posquat(sim, "palm_link", "cup"))
# z0 = np.zeros(3)
# z1 = np.zeros(3)
# z2 = np.zeros(3)
# z3 = np.zeros(3)
# Pmodel = 1
# conver_rate = 40
#
#
# G_contact = np.zeros([4, 6, 6])  # store G, the number of G is uncertain, this number between 1~4
# J = np.zeros([4, 6, 4])  # store J, the number of J is uncertain, this number between 1~4
# u_t_tmp = np.zeros([4, 4])  # store u_t, the number of u_t is uncertain, this number between 1~4
# nor_tmp = np.zeros([4, 3])  # store normal, the number of normal is uncertain, this number between 1~4
# coe_tmp = np.zeros([4, 6])  # store coefficients, the number of coefficients is uncertain, this number between 1~4
# Grasping_matrix = np.zeros([6, 6])
# G_pinv = np.zeros([6, 6])
# # 以下为EKF后验过程的变量
# P_ori = 1000 * np.ones([22, 22])
#
# y_t_update = np.array([np.zeros(10)])

# 记录用变量
# save_count_time = np.array([0])
# save_pose_y_t_xyz = np.array([0, 0, 0])
# save_pose_y_t_rpy = np.array([0, 0, 0])
# save_pose_GD_xyz = np.array([0, 0, 0])
# save_pose_GD_rpy = np.array([0, 0, 0])
# save_error_xyz = np.array([0, 0, 0])
# save_error_rpy = np.array([0, 0, 0])

# fcl库加载cup 的 BVH模型

# err_all = np.zeros(6)
# err = np.zeros(6)

########################################   FUNCTIONS DEFINITION   ######################################################
# 触觉点可视化


# def ekf_predictor(sim, y_t_update):
#     global flag, q_pos_pre, c_point_name, y_t, pos_contact_last, S1, y_t_pre, posquat_contact_pre, Grasping_matrix
#     global trans_palm2cup, q_pos_pre, P
#     global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
#     global save_error_xyz, save_error_rpy, save_model
#     global u_t, trans_cup2palm, conver_rate
#     global fin_num, fin_tri, J, u_t_tmp, nor_tmp, G_contact, G_pinv
#
#     # print("*" * 30)
#     ##############   refresh parameters  ###################
#     #todo why delta_t set 0.1
#     delta_t = 0.1
#     # number of triggered fingers
#     fin_num = 0
#     # Which fingers are triggered? Mark them with "1"
#     fin_tri = np.zeros(4)
#
#     # #grasping matrix of fingers
#     # G_pinv = np.zeros([6, 6])
#
#     #Jacobian matrix of fingers
#     J_fingers = np.zeros([6, 4])
#     nor_in_p = np.zeros(3)
#
#     y_t_update = np.ravel(y_t_update)  # flatten
#     if (is_finger_contact(hand_param[1][0]) == True ) \
#             or (is_finger_contact(hand_param[2][0]) == True ) \
#             or (is_finger_contact(hand_param[3][0]) == True ) \
#             or (is_finger_contact(hand_param[4][0]) == True ):
#         if not flag:  # get the state of cup in the first round
#             # noise +-5 mm, +-0.08 rad(4.5 deg)
#             print(object_param[1])
#             init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), \
#                                                   (1, 3)), \
#                                 np.random.uniform(-1*float(object_param[2]), float(object_param[2]), (1, 3))))
#             y_t_update = ug.get_relative_posquat(sim, "palm_link", "cup")  # x y z w
#             y_t_update = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(y_t_update)])
#             y_t_update += init_e
#             flag = True
#         y_t_update = np.ravel(y_t_update)
#
#         contact_compute(hand_param[1][0])
#         contact_compute(hand_param[2][0])
#         contact_compute(hand_param[3][0])
#         contact_compute(hand_param[4][0])
#
#         ################################################################################################################
#         ########## Splice Big G ##########
#         Grasping_matrix = np.zeros([6 * fin_num, 6])  # dim: 6n * 6
#         print(Grasping_matrix)
#         for i in range(fin_num):
#             Grasping_matrix[0 + i * 6: 6 + i * 6, :] = G_contact[i]
#         # print("CHek:", Grasping_matrix)
#
#         ############### Splice Big J #################
#         J_fingers = np.zeros([6 * fin_num, 4 * fin_num])
#         for i in range(fin_num):
#             J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = J[i]
#
#         ############# Splice Big u_t #################
#         u_t = np.zeros(4 * fin_num)
#         for i in range(fin_num):
#             u_t[0 + i * 4: 4 + i * 4] = u_t_tmp[i]
#
#         ########## Splice Big normal_in_palm #########
#         nor_in_p = np.zeros(3 * fin_num)
#         for i in range(fin_num):
#             nor_in_p[0 + i * 3: 3 + i * 3] = nor_tmp[i]
#
#         G_pinv = np.linalg.pinv(Grasping_matrix)  # Get G_pinv
#         #todo could you please comment the formula the computation,
#         # e.g. give the reference (to paper/book, pages, eq id)?
#         #todo u_t*delta_t should be replaced by delta_theta
#         prediction = np.matmul(np.matmul(G_pinv, J_fingers), u_t * delta_t)  # Predict
#
#         ###############################
#         y_t = y_t_update + prediction
#         #todo what does revel here?
#         y_t = np.ravel(y_t)
#         ###############################
#
#         # cup真值的获取
#         ground_truth = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(ug.get_relative_posquat(sim, "palm_link", "cup"))])
#         error = np.abs(y_t - ground_truth)
#
#         count_time += 1
#         # Save
#         save_count_time = np.hstack((save_count_time, count_time))
#         save_pose_y_t_xyz = np.vstack((save_pose_y_t_xyz, np.array(y_t[:3])))
#         save_pose_y_t_rpy = np.vstack((save_pose_y_t_rpy, np.array(y_t[3:])))
#         save_pose_GD_xyz = np.vstack((save_pose_GD_xyz, np.array(ground_truth[0, :3])))
#         save_pose_GD_rpy = np.vstack((save_pose_GD_rpy, np.array(ground_truth[0, 3:])))
#         save_error_xyz = np.vstack((save_error_xyz, np.array(error[0, :3])))
#         save_error_rpy = np.vstack((save_error_rpy, np.array(error[0, 3:])))
#         # if save_model == 0:
#         #     np.save("save_date/WithoutInitE/save_f/save_count_time.npy", save_count_time)
#         #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
#         #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
#         #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_xyz.npy", save_pose_GD_xyz)
#         #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_rpy.npy", save_pose_GD_rpy)
#         #     np.save("save_date/WithoutInitE/save_f/save_error_xyz.npy", save_error_xyz)
#         #     np.save("save_date/WithoutInitE/save_f/save_error_rpy.npy", save_error_rpy)
#         if save_model == 1:
#             np.save("save_i/save_count_time.npy", save_count_time)
#             np.save("save_i/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
#             np.save("save_i/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
#             np.save("save_i/save_pose_GD_xyz.npy", save_pose_GD_xyz)
#             np.save("save_i/save_pose_GD_rpy.npy", save_pose_GD_rpy)
#             np.save("save_i/save_error_xyz.npy", save_error_xyz)
#             np.save("save_i/save_error_rpy.npy", save_error_rpy)
#
#     return nor_in_p, Grasping_matrix, y_t, u_t, np.matmul(G_pinv, J_fingers) * delta_t, coe_tmp, err


# def ekf_posteriori(sim, model, viewer, z_t, h_t, Grasping_matrix, y_t, F_part, coe, err):
#     global P_ori, y_t_update, err_all
#     global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
#
#     #todo add reference to equation paper/book pages, eq id
#     #######################################    F    ###########################################
#     F = np.eye(6 + 4 * fin_num)
#     F[:6, 6:] = F_part
#
#     #######################################    R    ###########################################
#     #todo where does it come?
#     R_tmp = np.array([[2.37024061e-001, -3.06343629e-001, 6.14938815e-001, 1.17960413e-001,
#                        3.05049539e-001, -1.68124732e-003],
#                       [-3.06343629e-001, 6.93113033e-001, -9.19982570e-001, -3.85185662e-001,
#                        -9.70974182e-002, 1.42353856e-001],
#                       [6.14938815e-001, -9.19982570e-001, 2.07220530e+000, 7.14798278e-001,
#                        1.04838975e+000, -1.37061610e-001],
#                       [1.17960413e-001, -3.85185662e-001, 7.14798278e-001, 1.36401953e+000,
#                        4.32821898e-002, 1.60474548e-001],
#                       [3.05049539e-001, -9.70974182e-002, 1.04838975e+000, 4.32821898e-002,
#                        1.29434881e+000, -7.21019125e-002],
#                       [-1.68124732e-003, 1.42353856e-001, -1.37061610e-001, 1.60474548e-001,
#                        -7.21019125e-002, 5.31751383e-001]])
#     R_ori = np.zeros([12, 12])
#     R_ori[:6, :6] = R_tmp
#     R = R_ori[:3 * fin_num, :3 * fin_num]
#
#     #######################################    Q    ###########################################
#     Q_ori = sQR.get_Qt()
#     Q = Q_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]
#
#     #######################################    H    ###########################################
#     H_t = np.zeros([3 * fin_num, 6 + 4 * fin_num])
#     H_t_tmp2 = np.linalg.pinv(Grasping_matrix[:3].T)
#     for i in range(fin_num):
#         H_t_cup = np.array([[2 * coe[i][0], coe[i][1], 0],
#                             [coe[i][1], 2 * coe[i][2], 0],
#                             [0, 0, 0]])
#         H_t_tmp1 = np.matmul(trans_cup2palm[:3, :3], H_t_cup)
#         H_t[0 + 3 * i: 3 + 3 * i, :6] = np.matmul(H_t_tmp1, H_t_tmp2)
#
#     #######################################    P    ###########################################
#     P_tmp = P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]
#     P_pre = np.matmul(np.matmul(F, P_tmp), F.T) + Q  # P:{6+4n}*{6+4n}  Q:{6+4n}*{6+4n}
#     ############################################################################################
#
#     error = z_t - h_t
#     ###################################    Estimate    ########################################
#     K_t = ug.mul_three(P_pre, H_t.T, np.linalg.pinv(ug.mul_three(H_t, P_pre, H_t.T) + R))  # K:{6+4n}*3n
#     print("K_t", K_t.shape)
#     P_t = np.matmul((np.eye(6 + 4 * fin_num) - np.matmul(K_t, H_t)), P_pre)  # P_t in current round
#     P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num] = P_t  # update to P_original
#     y_t = np.ravel(y_t)  # flatten
#     err = err_all[0]
#     err_all = err_all[1:]
#     err = np.ravel(err)
#
#     # y_t_update = y_t + np.matmul(K_t, error.T).T
#     y_t_update = y_t + np.hstack((err, np.zeros(4 * fin_num)))
#
#     return y_t_update


# def EKF():
#     global y_t_update, save_model, trans_cup2palm, Pmodel, err_all, fin_num
#     #used for ground truth
#     trans_cup2palm = ug.posquat2trans(ug.get_relative_posquat(sim, "palm_link", "cup"))  # update T_cup2palm
#
#     # Forward Predictor
#     h_t, grasping_matrix, y_t_ready, u_t, F_part, coe, err = ekf_predictor(sim, y_t_update)
#     #######################   Frame Loss: contact loss or get nan_normal   ############################
#     if fin_num == 2:
#         Pmodel = 2
#
#     if fin_num != Pmodel or np.isnan(np.sum(h_t)):
#         # if fin_num ==0 or np.isnan(np.sum(h_t)) or (fin_tri[0] == 0 and fin_tri[1] == 1):
#         y_t_update = y_t_ready
#         print("pass")
#         return 0
#     ###################################################################################################
#
#     y_t_ready = np.ravel(y_t_ready)
#     u_t = np.ravel(u_t)
#     y_t = np.hstack((y_t_ready, u_t))  # splice to 6+4n
#     print("!!!!!!!!y_t:", y_t)
#
#     # FCL give out z_t
#     # z_t = collision_test(fin_tri)
#     z_t = h_t + np.random.uniform(-0.1, 0.1, 3 * fin_num)
#     # z_t = f2.normalization(z_t)
#     print("new z_t:", z_t)
#
#     err_all = np.vstack((err_all, err))
#     ######<<<<<<<<<<< Switch fEKF / iEKF <<<<<<<<<<#########
#     # y_t_update = y_t
#     y_t_update = ekf_posteriori(sim, model, viewer, z_t, h_t, grasping_matrix, y_t, F_part, coe, err)
#     ######>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#########
#     y_t_update = y_t_update[:6]  # Remove control variables
#
# def interacting(hand_param):
#     global err_all
#     #todo what is the file?
#     err_all = np.loadtxt("./err_inHand_v3bi.txt")
#     robcontrol.pre_thumb(sim, viewer)  # Thumb root movement
#     # Fast
#     for ii in range(1000):
#         if hand_param[1][1] == '1':
#             robcontrol.index_finger(sim, 0.0055, 0.00001)
#         if hand_param[2][1] == '1':
#             robcontrol.middle_finger(sim, 0.0016, 0.00001)
#         if hand_param[3][1] == '1':
#             robcontrol.ring_finger(sim, 0.002, 0.00001)
#         if hand_param[4][1] == '1':
#             robcontrol.thumb(sim, 0.0003, 0.00001)
#         EKF()
#         if not np.all(sim.data.sensordata == 0):
#             viz.touch_visual(sim, model. viewer, np.where(np.array(sim.data.sensordata) > 0.0))
#         sim.step()
#         viewer.render()
#     # plt_plus.plot_error("save_i")


############################>>>>>>>>>>>>>>>    MAIN LOOP    <<<<<<<<<<<<<###############################################

# robcontrol.robot_init(sim)
# mu_env.Camera_set(viewer, model)
# sim.model.eq_active[0] = True
#
# for i in range(500):
#     sim.step()
#     viewer.render()
#
# sim.data.mocap_pos[0] = ctrl_wrist_pos  # mocap控制需要用世界参考系
# sim.data.mocap_quat[0] = ctrl_wrist_quat  # mocap控制需要用世界参考系
# for _ in range(50):
#     sim.step()
#     viewer.render()

# interacting(hand_param)

# print(np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))
#
# save_tmp = np.load("save_date/iEKF_incremental.npy")
# np.save("save_date/iEKF_incremental.npy", np.vstack((save_tmp, np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))))
# print("Run times:", int(save_tmp.shape[0] / 45 + 1))
# print("over, shape:", save_tmp.shape)
# print("cHeCK GD:", save_pose_GD_xyz.shape)
#
# np.save("save_date/iEKF_incremental.npy", np.hstack((save_pose_y_t_xyz, save_pose_y_t_rpy)))
