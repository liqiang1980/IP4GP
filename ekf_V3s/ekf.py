import numpy as np
import tactile_perception as tacperception
import util_geometry as ug
import viz

class EKF:
    # def __init__(self, x_state):
    #     self.x_state = x_state

    def __init__(self):
        print('init ekf')


#     def predictor(self, hand_param, object_param, sim, y_t_update):
#       global flag, q_pos_pre, c_point_name, y_t, pos_contact_last, S1, y_t_pre, posquat_contact_pre, Grasping_matrix
#       global trans_palm2cup, q_pos_pre, P
#       global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
#       global save_error_xyz, save_error_rpy, save_model
#       global u_t, trans_cup2palm, conver_rate
#       global fin_num, fin_tri, J, u_t_tmp, nor_tmp, G_contact, G_pinv
#
#       ##############   refresh parameters  ###################
#       # todo why delta_t set 0.1
#       delta_t = 0.1
#       # number of triggered fingers
#       fin_num = 0
#       # Which fingers are triggered? Mark them with "1"
#       fin_tri = np.zeros(4)
#
#       # #grasping matrix of fingers
#       # G_pinv = np.zeros([6, 6])
#
#       # Jacobian matrix of fingers
#       J_fingers = np.zeros([6, 4])
#       nor_in_p = np.zeros(3)
#
#       y_t_update = np.ravel(y_t_update)  # flatten
#       if (tacperception.is_finger_contact(sim, hand_param[1][0]) == True) \
#               or (tacperception.is_finger_contact(sim, hand_param[2][0]) == True) \
#               or (tacperception.is_finger_contact(sim, hand_param[3][0]) == True) \
#               or (tacperception.is_finger_contact(sim, hand_param[4][0]) == True):
#         if not flag:  # get the state of cup in the first round
#           # noise +-5 mm, +-0.08 rad(4.5 deg)
#           print(object_param[1])
#           init_e = np.hstack((np.random.uniform((-1) * float(object_param[1]), float(object_param[1]),\
#                                                 (1, 3)),\
#                               np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))
#           y_t_update = ug.get_relative_posquat(sim, "palm_link", "cup")  # x y z w
#           y_t_update = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(y_t_update)])
#           y_t_update += init_e
#           flag = True
#         y_t_update = np.ravel(y_t_update)
#
#         ug.contact_compute(hand_param[1][0])
#         ug.contact_compute(hand_param[2][0])
#         ug.contact_compute(hand_param[3][0])
#         ug.contact_compute(hand_param[4][0])
#
#         ################################################################################################################
#         ########## Splice Big G ##########
#         Grasping_matrix = np.zeros([6 * fin_num, 6])  # dim: 6n * 6
#         print(Grasping_matrix)
#         for i in range(fin_num):
#           Grasping_matrix[0 + i * 6: 6 + i * 6, :] = G_contact[i]
#         # print("CHek:", Grasping_matrix)
#
#         ############### Splice Big J #################
#         J_fingers = np.zeros([6 * fin_num, 4 * fin_num])
#         for i in range(fin_num):
#           J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = J[i]
#
#         ############# Splice Big u_t #################
#         u_t = np.zeros(4 * fin_num)
#         for i in range(fin_num):
#           u_t[0 + i * 4: 4 + i * 4] = u_t_tmp[i]
#
#         ########## Splice Big normal_in_palm #########
#         nor_in_p = np.zeros(3 * fin_num)
#         for i in range(fin_num):
#           nor_in_p[0 + i * 3: 3 + i * 3] = nor_tmp[i]
#
#         G_pinv = np.linalg.pinv(Grasping_matrix)  # Get G_pinv
#         # todo could you please comment the formula the computation,
#         # e.g. give the reference (to paper/book, pages, eq id)?
#         # todo u_t*delta_t should be replaced by delta_theta
#         prediction = np.matmul(np.matmul(G_pinv, J_fingers), u_t * delta_t)  # Predict
#
#         ###############################
#         y_t = y_t_update + prediction
#         # todo what does revel here?
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
#           np.save("save_i/save_count_time.npy", save_count_time)
#           np.save("save_i/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
#           np.save("save_i/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
#           np.save("save_i/save_pose_GD_xyz.npy", save_pose_GD_xyz)
#           np.save("save_i/save_pose_GD_rpy.npy", save_pose_GD_rpy)
#           np.save("save_i/save_error_xyz.npy", save_error_xyz)
#           np.save("save_i/save_error_rpy.npy", save_error_rpy)
#
#       return nor_in_p, Grasping_matrix, y_t, u_t, np.matmul(G_pinv, J_fingers) * delta_t, coe_tmp, err
#
#
# def ekf_posteriori(sim, model, viewer, z_t, h_t, Grasping_matrix, y_t, F_part, coe, err):
#     global P_ori, y_t_update, err_all
#     global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
#
#     # todo add reference to equation paper/book pages, eq id
#     #######################################    F    ###########################################
#     F = np.eye(6 + 4 * fin_num)
#     F[:6, 6:] = F_part
#
#     #######################################    R    ###########################################
#     # todo where does it come?
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
#       H_t_cup = np.array([[2 * coe[i][0], coe[i][1], 0],
#                           [coe[i][1], 2 * coe[i][2], 0],
#                           [0, 0, 0]])
#       H_t_tmp1 = np.matmul(trans_cup2palm[:3, :3], H_t_cup)
#       H_t[0 + 3 * i: 3 + 3 * i, :6] = np.matmul(H_t_tmp1, H_t_tmp2)
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