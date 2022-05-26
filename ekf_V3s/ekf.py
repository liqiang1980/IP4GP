import numpy as np
import tactile_perception as tacperception
import util_geometry as ug
import viz
import storeQR as sQR


class EKF:
    # def __init__(self, x_state):
    #     self.x_state = x_state

    def __init__(self):
        print('init ekf')
        ##############   refresh parameters  ###################
        # todo why delta_t set 0.1
        self.delta_t = 0.1
        # number of triggered fingers
        self.fin_num = 0
        # Which fingers are triggered? Mark them with "1"
        self.fin_tri = np.zeros(4)
        #grasping matrix of fingers
        self.G_pinv = np.zeros([6, 6])
        # Jacobian matrix of fingers
        self.J_fingers = np.zeros([6, 4])
        self.nor_in_p = np.zeros(3)
        self.G_contact = np.zeros([4, 6, 6])
        self.count_time = 0
        self.save_count_time = []
        self.save_pose_y_t_xyz = []
        self.save_pose_y_t_rpy = []
        self.save_pose_GD_xyz = []
        self.save_pose_GD_rpy = []
        self.save_error_xyz = []
        self.save_error_rpy = []

    def set_store_flag(self, flag):
        self.save_model = flag

    def set_contact_flag(self, flag):
        self.flag = flag

    def state_predictor(self, sim, model, hand_param, object_param, y_t_update):
        print('state prediction')
        y_t_update = np.ravel(y_t_update)  # flatten
        self.fin_num, self.fin_tri, self.G_contact[self.fin_num], self.J[self.fin_num], self.u_t_tmp[self.fin_num], self.nor_tmp[self.fin_num] \
            = ug.contact_compute(sim, model, hand_param[1][0], self.fin_num, self.fin_tri)
        self.fin_num, self.fin_tri, self.G_contact[self.fin_num], self.J[self.fin_num], self.u_t_tmp[self.fin_num], self.nor_tmp[self.fin_num] \
            = ug.contact_compute(sim, model, hand_param[2][0], self.fin_num, self.fin_tri)
        self.fin_num, self.fin_tri, self.G_contact[self.fin_num], self.J[self.fin_num], self.u_t_tmp[self.fin_num], self.nor_tmp[self.fin_num] \
            = ug.contact_compute(sim, model, hand_param[3][0],self.fin_num, self.fin_tri )
        self.fin_num, self.fin_tri, self.G_contact[self.fin_num], self.J[self.fin_num], self.u_t_tmp[self.fin_num], self.nor_tmp[self.fin_num] \
            = ug.contact_compute(sim, model, hand_param[4][0], self.fin_num, self.fin_tri)

        ########## Splice Big G ##########
        self.Grasping_matrix = np.zeros([6 * self.fin_num, 6])  # dim: 6n * 6
        print(self.Grasping_matrix)
        for i in range(self.fin_num):
            self.Grasping_matrix[0 + i * 6: 6 + i * 6, :] = self.G_contact[i]
        # print("CHek:", Grasping_matrix)

        ############### Splice Big J #################
        self.J_fingers = np.zeros([6 * self.fin_num, 4 * self.fin_num])
        for i in range(self.fin_num):
            self.J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = self.J[i]

        ############# Splice Big u_t #################
        self.u_t = np.zeros(4 * self.fin_num)
        for i in range(self.fin_num):
            self.u_t[0 + i * 4: 4 + i * 4] = self.u_t_tmp[i]

        ########## Splice Big normal_in_palm #########
        self.nor_in_p = np.zeros(3 * self.fin_num)
        for i in range(self.fin_num):
            self.nor_in_p[0 + i * 3: 3 + i * 3] = self.nor_tmp[i]

        G_pinv = np.linalg.pinv(self.Grasping_matrix)  # Get G_pinv
        # todo could you please comment the formula the computation,
        # e.g. give the reference (to paper/book, pages, eq id)?
        # todo u_t*delta_t should be replaced by delta_theta
        prediction = np.matmul(np.matmul(G_pinv, self.J_fingers), self.u_t * self.delta_t)  # Predict

        ###############################
        y_t = y_t_update + prediction
        # todo what does revel here?
        y_t = np.ravel(y_t)
        ###############################

        # cup真值的获取
        ground_truth = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(ug.get_relative_posquat(sim, "palm_link", "cup"))])
        # error = np.abs(y_t - ground_truth)
        #
        # self.count_time += 1
        # # Save
        # self.save_count_time = np.hstack((self.save_count_time, self.count_time))
        # self.save_pose_y_t_xyz = np.vstack((self.save_pose_y_t_xyz, np.array(y_t[:3])))
        # self.save_pose_y_t_rpy = np.vstack((self.save_pose_y_t_rpy, np.array(y_t[3:])))
        # self.save_pose_GD_xyz = np.vstack((self.save_pose_GD_xyz, np.array(ground_truth[0, :3])))
        # self.save_pose_GD_rpy = np.vstack((self.save_pose_GD_rpy, np.array(ground_truth[0, 3:])))
        # self.save_error_xyz = np.vstack((self.save_error_xyz, np.array(error[0, :3])))
        # self.save_error_rpy = np.vstack((self.save_error_rpy, np.array(error[0, 3:])))
        # # if save_model == 0:
        # #     np.save("save_date/WithoutInitE/save_f/save_count_time.npy", save_count_time)
        # #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_xyz.npy", save_pose_y_t_xyz)
        # #     np.save("save_date/WithoutInitE/save_f/save_pose_y_t_rpy.npy", save_pose_y_t_rpy)
        # #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_xyz.npy", save_pose_GD_xyz)
        # #     np.save("save_date/WithoutInitE/save_f/save_pose_GD_rpy.npy", save_pose_GD_rpy)
        # #     np.save("save_date/WithoutInitE/save_f/save_error_xyz.npy", save_error_xyz)
        # #     np.save("save_date/WithoutInitE/save_f/save_error_rpy.npy", save_error_rpy)
        # if self.save_model == 1:
        #     np.save("save_i/save_count_time.npy", self.save_count_time)
        #     np.save("save_i/save_pose_y_t_xyz.npy", self.save_pose_y_t_xyz)
        #     np.save("save_i/save_pose_y_t_rpy.npy", self.save_pose_y_t_rpy)
        #     np.save("save_i/save_pose_GD_xyz.npy", self.save_pose_GD_xyz)
        #     np.save("save_i/save_pose_GD_rpy.npy", self.save_pose_GD_rpy)
        #     np.save("save_i/save_error_xyz.npy", self.save_error_xyz)
        #     np.save("save_i/save_error_rpy.npy", self.save_error_rpy)
          # return nor_in_p, Grasping_matrix, y_t, u_t, np.matmul(G_pinv, J_fingers) * delta_t, coe_tmp, err
        return y_t

    def observation_computation(self, y_bar):
        print('measurement equation computation')

    def ekf_posteriori(self, sim, model, viewer, z_t, h_t):
        print('posterior computation')
        # global P_ori, y_t_update, err_all
        # global save_count_time, save_pose_y_t_xyz, save_pose_y_t_rpy, save_pose_GD_xyz, save_pose_GD_rpy, count_time
        #
        # # todo add reference to equation paper/book pages, eq id
        # #######################################    F    ###########################################
        # F = np.eye(6 + 4 * fin_num)
        # F[:6, 6:] = F_part
        #
        # #######################################    R    ###########################################
        # # todo where does it come?
        # R_tmp = np.array([[2.37024061e-001, -3.06343629e-001, 6.14938815e-001, 1.17960413e-001,
        #                    3.05049539e-001, -1.68124732e-003],
        #                   [-3.06343629e-001, 6.93113033e-001, -9.19982570e-001, -3.85185662e-001,
        #                    -9.70974182e-002, 1.42353856e-001],
        #                   [6.14938815e-001, -9.19982570e-001, 2.07220530e+000, 7.14798278e-001,
        #                    1.04838975e+000, -1.37061610e-001],
        #                   [1.17960413e-001, -3.85185662e-001, 7.14798278e-001, 1.36401953e+000,
        #                    4.32821898e-002, 1.60474548e-001],
        #                   [3.05049539e-001, -9.70974182e-002, 1.04838975e+000, 4.32821898e-002,
        #                    1.29434881e+000, -7.21019125e-002],
        #                   [-1.68124732e-003, 1.42353856e-001, -1.37061610e-001, 1.60474548e-001,
        #                    -7.21019125e-002, 5.31751383e-001]])
        # R_ori = np.zeros([12, 12])
        # R_ori[:6, :6] = R_tmp
        # R = R_ori[:3 * fin_num, :3 * fin_num]
        #
        # #######################################    Q    ###########################################
        # Q_ori = sQR.get_Qt()
        # Q = Q_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]
        #
        # #######################################    H    ###########################################
        # H_t = np.zeros([3 * fin_num, 6 + 4 * fin_num])
        # H_t_tmp2 = np.linalg.pinv(Grasping_matrix[:3].T)
        # for i in range(fin_num):
        #   H_t_cup = np.array([[2 * coe[i][0], coe[i][1], 0],
        #                       [coe[i][1], 2 * coe[i][2], 0],
        #                       [0, 0, 0]])
        #   H_t_tmp1 = np.matmul(trans_cup2palm[:3, :3], H_t_cup)
        #   H_t[0 + 3 * i: 3 + 3 * i, :6] = np.matmul(H_t_tmp1, H_t_tmp2)
        #
        # #######################################    P    ###########################################
        # P_tmp = P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num]
        # P_pre = np.matmul(np.matmul(F, P_tmp), F.T) + Q  # P:{6+4n}*{6+4n}  Q:{6+4n}*{6+4n}
        # ############################################################################################
        #
        # error = z_t - h_t
        # ###################################    Estimate    ########################################
        # K_t = ug.mul_three(P_pre, H_t.T, np.linalg.pinv(ug.mul_three(H_t, P_pre, H_t.T) + R))  # K:{6+4n}*3n
        # print("K_t", K_t.shape)
        # P_t = np.matmul((np.eye(6 + 4 * fin_num) - np.matmul(K_t, H_t)), P_pre)  # P_t in current round
        # P_ori[:6 + 4 * fin_num, :6 + 4 * fin_num] = P_t  # update to P_original
        # y_t = np.ravel(y_t)  # flatten
        # err = err_all[0]
        # err_all = err_all[1:]
        # err = np.ravel(err)
        #
        # # y_t_update = y_t + np.matmul(K_t, error.T).T
        # y_t_update = y_t + np.hstack((err, np.zeros(4 * fin_num)))
        #
        # return y_t_update