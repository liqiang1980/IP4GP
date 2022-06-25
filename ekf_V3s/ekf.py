import numpy as np
import tactile_perception as tacperception
import util_geometry as ug
import object_geometry as og
import viz
import storeQR as sQR


# the concept of ekf algorithm can refer to one simplified example in
# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/

class EKF:

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
        self.J = np.zeros([4, 6, 4])
        self.nor_in_p = np.zeros(3)
        self.G_contact = np.zeros([4, 6, 6])
        self.u_t_tmp = np.zeros([4, 4])
        self.nor_tmp = np.zeros([4, 3])
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

    def state_predictor(self, sim, model, hand_param, object_param, x_state, tacperception, P_state_cov):
        print("state prediction")
        Transfer_Fun_Matrix = np.identity(6 + tacperception.fin_num * 3)
        Q_state_noise_cov = 0.001 * np.identity(6 + tacperception.fin_num * 3)

        x_state = np.ravel(x_state)
        if tacperception.fin_num == 2:
            print(np.nonzero(tacperception.fin_tri))
        contact_finger_id = np.nonzero(tacperception.fin_tri)[0]
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri
        for i in range(tacperception.fin_num):
            self.G_contact[i, :, :], self.J[i, :, :], self.u_t_tmp[i, :]\
                = ug.contact_compute(sim, model, hand_param[contact_finger_id[i] + 1][0], tacperception, x_state)
        ########## Splice Big G ##########
        self.Grasping_matrix = np.zeros([6 * self.fin_num, 6])  # dim: 6n * 6
        # print(self.Grasping_matrix)
        for i in range(self.fin_num):
            self.Grasping_matrix[0 + i * 6: 6 + i * 6, :] = self.G_contact[i, :, :]
        # print("CHek:", self.Grasping_matrix)

        ############### Splice Big J #################
        self.J_fingers = np.zeros([6 * self.fin_num, 4 * self.fin_num])
        for i in range(self.fin_num):
            self.J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = self.J[i, :, :]

        ############# Splice Big u_t #################
        self.u_t = np.zeros(4 * self.fin_num)
        for i in range(self.fin_num):
            self.u_t[0 + i * 4: 4 + i * 4] = self.u_t_tmp[i]

        G_pinv = np.linalg.pinv(self.Grasping_matrix)  # Get G_pinv
        prediction = np.matmul(np.matmul(G_pinv, self.J_fingers), self.u_t)
        #currently it is cheating that we only give the accurate contact position relevant to
        #the object frame
        if self.fin_num == 1:
            prediction = np.append(prediction, [0, 0, 0])
        if self.fin_num == 2:
            prediction = np.append(prediction, [0, 0, 0, 0, 0, 0])
        if self.fin_num == 3:
            prediction = np.append(prediction, [0, 0, 0, 0, 0, 0, 0, 0, 0])
        if self.fin_num == 4:
            prediction = np.append(prediction, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x_bar = x_state + prediction * \
                np.random.uniform(-0.001, 0.001, (1, 6 + self.fin_num * 3))
        x_bar = np.ravel(x_bar)

        P_state_cov = Transfer_Fun_Matrix * P_state_cov * \
                      Transfer_Fun_Matrix.transpose() + Q_state_noise_cov

        return x_bar, P_state_cov

        # cup真值的获取
        # ground_truth = np.array([ug.pos_quat2pos_XYZ_RPY_xyzw(ug.get_relative_posquat(sim, "palm_link", "cup"))])
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


    def observe_computation(self, x_bar, tacperception):
        print('measurement equation computation')
        contact_position = []
        contact_nv = []
        contact_finger_id = np.nonzero(tacperception.fin_tri)[0]
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri

        for i in range(tacperception.fin_num):
            obj_position = x_bar[:3]
            ct_relative_obj = x_bar[6:9]
            rot = ug.pos_euler_xyz_2_matrix(x_bar[3:6])
            contact_position.append(obj_position + np.matmul(rot, np.array((ct_relative_obj))))

            # get mean position:
            #############################  Get normal of contact point on the cup
            # ########################################
            nor_contact_in_cup, res = og.surface_cup(obj_position[0], obj_position[1],
                                obj_position[2])
            contact_nv.append(nor_contact_in_cup / np.linalg.norm(nor_contact_in_cup))
        return np.array(contact_position), np.array(contact_nv),

    def measure_fb(self, sim, model, hand_param, object_param, \
                                                       x_bar, tacperception):
        print('measurement feedback from sensing (ground truth + noise in simulation)')
        contact_position = []
        contact_nv = []
        contact_finger_id = np.nonzero(tacperception.fin_tri)[0]
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri
        for i in range(tacperception.fin_num):
            contact_position.append(tacperception.tuple_fin_ref_pose[3][:3] + np.random.uniform(-0.002, 0.002, 3))
            contact_nv.append(tacperception.get_contact_taxel_nv(sim, model, \
                                                       hand_param[contact_finger_id[i] + 1][0], "palm_link") \
                      + np.random.uniform(-0.01, 0.01, 3))
        return np.array(contact_position), np.array(contact_nv)

    def ekf_posteriori(self, sim, model, viewer, x_bar, z_t, h_t, P_state_cov, tacperception):
        print('posterior computation')
        # the jocobian matrix of measurement equation
        J_h = np.zeros([6, 6 + tacperception.fin_num * 3])
        # the covariance of measurement noise
        R_noi = np.random.normal(0, 0.01, size=(6, 6))
        K_t =  np.matmul(np.matmul(P_state_cov, J_h.transpose()), np.linalg.pinv(np.matmul(np.matmul(J_h, P_state_cov), J_h.transpose()) + R_noi))

        x_hat = x_bar + np.matmul(K_t, (z_t - h_t))
        P_state_cov = (np.zeros([6 + tacperception.fin_num * 3, 6 + tacperception.fin_num * 3]) \
                       - K_t @ J_h ) @ P_state_cov
        return x_hat, P_state_cov