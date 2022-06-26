import numpy as np
import tactile_perception as tacperception
import util_geometry as ug
import object_geometry as og
import viz
import storeQR as sQR
import time

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
        Transfer_Fun_Matrix = np.identity(6 + 4 * 3)
        Q_state_noise_cov = 0.001 * np.identity(6 + 4 * 3)

        x_state = np.ravel(x_state)
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri

        # start = time.time()

        for i in range(4):
            self.G_contact[i, :, :], self.J[i, :, :], self.u_t_tmp[i, :]\
                = ug.contact_compute(sim, model, hand_param[i + 1][0], tacperception, x_state)

        # end = time.time()
        # print("The time of execution of above program is :", end-start)

        ########## form Grasping matrix from contact information ##########
        # dim: 6n * 6, n is the number of fingers
        self.Grasping_matrix = np.zeros([6 * 4, 6])
        # print(self.Grasping_matrix)
        for i in range(4):
            self.Grasping_matrix[0 + i * 6: 6 + i * 6, :] = self.G_contact[i, :, :]
        # print("CHek:", self.Grasping_matrix)

        ############### Form hand Jacobian matrix from contact information #################
        self.J_fingers = np.zeros([6 * 4, 4 * 4])
        for i in range(4):
            self.J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = self.J[i, :, :]

        ############# form action - u_t #################
        self.u_t = np.zeros(4 * 4)
        for i in range(4):
            self.u_t[0 + i * 4: 4 + i * 4] = self.u_t_tmp[i]

        G_pinv = np.linalg.pinv(self.Grasping_matrix)  # Get G_pinv
        prediction = np.matmul(np.matmul(G_pinv, self.J_fingers), self.u_t)

        prediction = np.append(prediction, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x_bar = x_state + prediction + \
                np.random.normal(0, 0.001, (1, 6 + 4 * 3))
        x_bar = np.ravel(x_bar)

        P_state_cov = Transfer_Fun_Matrix * P_state_cov * \
                      Transfer_Fun_Matrix.transpose() + Q_state_noise_cov

        return x_bar, P_state_cov

    def observe_computation(self, x_bar, tacperception):
        print('measurement equation computation')
        contact_position = []
        contact_nv = []
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri
        obj_position = x_bar[:3]
        rot = ug.pos_euler_xyz_2_matrix(x_bar[3:6])
        for i in range(4):
            ct_relative_obj = x_bar[6:24]
            if tacperception.fin_tri[i] == 1:
                contact_position.append(obj_position + np.matmul(rot, np.array((x_bar[6 + i * 3:6 + (i + 1) * 3]))))
                ##########Get normal of contact point on the cup
                nor_contact_in_cup, res = og.surface_cup(x_bar[6 + i * 3], x_bar[6 + i * 3 + 1],
                                    x_bar[6 + i * 3 + 2])
                nor_contact_in_world = np.matmul(rot, nor_contact_in_cup)
                contact_nv.append(nor_contact_in_world / np.linalg.norm(nor_contact_in_world))
            else:
                contact_position.append([0, 0, 0])
                contact_nv.append([0, 0, 0])

        return np.array(contact_position), np.array(contact_nv),

    def measure_fb(self, sim, model, hand_param, object_param, \
                                                       x_bar, tacperception):
        print('measurement feedback from sensing (ground truth + noise in simulation)')
        contact_position = []
        contact_nv = []
        self.fin_num = tacperception.fin_num
        self.fin_tri = tacperception.fin_tri
        for i in range(4):
            if tacperception.fin_tri[i] == 1:
                contact_position.append((tacperception.get_contact_taxel_position(sim, model, \
                                                                     hand_param[i + 1][0], "palm_link"))[:3] \
                                        + np.random.normal(0, 0.001, 3))
                contact_nv.append(tacperception.get_contact_taxel_nv(sim, model, \
                                                           hand_param[i + 1][0], "palm_link") \
                          + np.random.normal(0, 0.001, 3))
            else:
                contact_position.append([0, 0, 0])
                contact_nv.append([0, 0, 0])

        return np.array(contact_position), np.array(contact_nv)

    def ekf_posteriori(self, sim, model, viewer, x_bar, z_t, h_t, P_state_cov, tacperception):
        print('posterior computation')
        # the jocobian matrix of measurement equation
        J_h = np.zeros([6 * 4, 6 + 4 * 3])
        # the covariance of measurement noise
        R_noi = np.random.normal(0, 0.01, size=(6 * 4, 6 * 4))
        # K_t =  np.matmul(np.matmul(P_state_cov, J_h.transpose()), \
        #                  np.linalg.pinv(np.matmul(np.matmul(J_h, P_state_cov), J_h.transpose()) + R_noi))
        K_t = P_state_cov @ J_h.transpose() @ \
              np.linalg.pinv(J_h @ P_state_cov @ J_h.transpose() + R_noi)
        x_hat = x_bar + np.matmul(K_t, (z_t - h_t))
        P_state_cov = (np.zeros([6 + 4 * 3, 6 + 4 * 3]) \
                       - K_t @ J_h ) @ P_state_cov
        return x_hat, P_state_cov