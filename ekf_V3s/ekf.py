import numpy as np
import copy
import qgFunc
import tactile_allegro_mujo_const as tac_const
import util_geometry as ug
import object_geometry as og
import math
import viz
import storeQR as sQR
import time
from scipy.spatial.transform import Rotation


# the concept of ekf algorithm can refer to one simplified example in
# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/

class EKF:
    def __init__(self):
        print('init ekf')
        ##############   refresh parameters  ###################
        # todo why delta_t set 0.1
        self.delta_t = 0.1
        # self.fin_num = 0
        # self.fin_tri = np.zeros(4)
        # self.G_pinv = np.zeros([6, 6])
        # self.J = np.zeros([4, 6, 4])
        self.nor_in_p = np.zeros(3)
        # self.G_contact = np.zeros([4, 6, 6])
        self.nor_tmp = np.zeros([4, 3])
        self.count_time = 0
        self.save_count_time = []
        self.save_pose_y_t_xyz = []
        self.save_pose_y_t_rpy = []
        self.save_pose_GD_xyz = []
        self.save_pose_GD_rpy = []
        self.save_error_xyz = []
        self.save_error_rpy = []
        self.delta_angle = []
        self.scale_kp = 0.3

    def set_store_flag(self, flag):
        self.save_model = flag

    def set_contact_flag(self, flag):
        self.flag = flag

    def state_predictor(self, xstate, P_state_cov, tacp, robctrl):
        # print("xstate_aug:", xstate)
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        # print('======================================')
        f_param = robctrl.f_param
        """ Matrix Initialization: F, P, G """
        F_Matrix = np.mat(np.zeros((6+3*robctrl.f_size, 6+3*robctrl.f_size)))
        Q_state_noise_cov = np.zeros((6+3*robctrl.f_size, 6+3*robctrl.f_size))
        for i in range(6):
            Q_state_noise_cov[i, i] = math.fabs(np.random.normal(0, 0.001))
        Grasping_matrix = np.zeros([6, 6 * robctrl.f_size])
        G_pinv = np.zeros([6, 6 * robctrl.f_size])
        ju = np.zeros(6 * robctrl.f_size)

        """ 
        G matrix calculate 
        and
        Get the twists of contact tacs: These are equivalent to Jac * delta_u
        """
        for i, f_part in enumerate(f_param):
            f_name = f_part[0]
            tac_name = tacp.cur_tac[f_name][0]  # If no contact, cur-tac info is updated by last-tac
            tac_posrotvec = tacp.cur_tac[f_name][1]
            if not tacp.is_contact[f_name]:  # this part is no-contact, pass
                continue
            pos_tac_palm = tac_posrotvec[:3]
            pos_cup_palm = xstate[:3]
            tmp_G = ug.get_Grasp_matrix(pos_tac_palm=pos_tac_palm, pos_cup_palm=pos_cup_palm)
            Grasping_matrix[:, 6*i: 6*i+6] = tmp_G
            ju[6*i: 6*i+6] = tacp.cur_tac[f_name][1] - tacp.last_tac[f_name][1]
            # print("  ", f_name, " cur:", tacp.cur_tac[f_name][1], "  last:", tacp.last_tac[f_name][1])
        # print("  >>ju:\n", ju)

        """ G_pinv calculate: 2 type """
        if tac_const.GT_FLAG == '1G':  # 4 G splice and calculate big GT_pinv
            G_pinv = np.linalg.pinv(Grasping_matrix.T)  # Get G_pinv
        elif tac_const.GT_FLAG == '4G':
            for i in range(robctrl.f_size):
                inv_tmp = Grasping_matrix[:, 0 + i * 6: 6 + i * 6]
                Y_RU = np.copy(inv_tmp[0:3, 3:6])
                Y_LD = np.copy(inv_tmp[3:6, 0:3])
                inv_tmp[0:3, 3:6] = Y_LD
                inv_tmp[3:6, 0:3] = Y_RU
                G_pinv[:, 0 + i * 6: 6 + i * 6] = inv_tmp  # 4 GT_inv splice a big G

        """ Prediction calculation """
        prediction = np.matmul(G_pinv, ju)
        prediction = np.append(prediction, [0] * (3 * robctrl.f_size))
        x_bar = xstate + prediction
        # x_bar = xstate
        # print("  >>prediction:\n", prediction)

        """ F_matrix and P_matrix update """
        if tac_const.GT_FLAG == '4G':
            F_Matrix[:6, :6] = np.mat(np.eye(6))
        P_state_cov = F_Matrix * P_state_cov * F_Matrix.transpose() + Q_state_noise_cov

        return x_bar, P_state_cov

    def observe_computation(self, tacp, robctrl):
        """
        Calculation of ht: position and normal.
        """
        pos_contact_palm = np.zeros(3 * robctrl.f_size)
        nv_contact_palm = np.zeros(3 * robctrl.f_size)
        # _obj_position = x_bar[:3]  # position of object in palm
        # _rot = ug.rotvec_2_Matrix(x_bar[3:6])  # rot of object in palm
        for i, f_part in enumerate(robctrl.f_param):
            f_name = f_part[0]
            ht_idx = [3 * i, 3 * i + 3]
            # if tacp.is_contact[f_name]:  # Keep at np.zeros(3) if no contact
            if tacp.is_first_contact[f_name]:  # Keep at np.zeros(3) if no contact
                # _pq_cup_palm = ug.get_relative_posquat(sim=sim, src="palm_link", tgt="cup")  # wxyz
                # _quat_cup_palm = np.hstack((_pq_cup_palm[4:], _pq_cup_palm[3]))
                # _R_cup_palm = Rotation.from_quat(_quat_cup_palm).as_matrix()
                pos_contact_palm[ht_idx[0]: ht_idx[1]] = np.ravel(
                    robctrl.pos_cup_palm + np.matmul(robctrl.R_cup_palm, robctrl.pos_contact_cup[f_name]))
                # pos_contact_palm[ht_idx[0]: ht_idx[1]] = np.ravel(_pq_cup_palm[:3] + np.matmul(_R_cup_palm, robctrl.pos_contact_cup[f_name]))
                """ Get normals from surface of object """
                # if int(object_param[3]) == 3:  # the object is a cylinder
                #     nv_contact_cup, res = og.surface_cylinder(cur_x=robctrl.pos_contact_cup[f_name][0],
                #                                               cur_y=robctrl.pos_contact_cup[f_name][1],
                #                                               cur_z=robctrl.pos_contact_cup[f_name][2])
                # else:  # the object is a cup
                #     nv_contact_cup, res = og.surface_cup(cur_x=robctrl.pos_contact_cup[f_name][0],
                #                                          cur_y=robctrl.pos_contact_cup[f_name][1],
                #                                          cur_z=robctrl.pos_contact_cup[f_name][2])
                nv_contact_cup = og.get_nv_contact_cup(obj_param=robctrl.obj_param,
                                                       pos_contact_cup=robctrl.pos_contact_cup[f_name])
                normal_contact_palm = np.matmul(robctrl.R_cup_palm, nv_contact_cup)
                nv_contact_palm[ht_idx[0]: ht_idx[1]] = normal_contact_palm / np.linalg.norm(normal_contact_palm)
        return np.array(pos_contact_palm), np.array(nv_contact_palm),

    def measure_fb(self, tacp, robctrl):
        """
        Calculation of zt: position and normal.
        """
        pos_tac_palm = np.zeros(3 * robctrl.f_size)
        nv_tac_palm = np.zeros(3 * robctrl.f_size)
        # self.fin_num = tacperception.fin_num
        # self.fin_tri = tacperception.fin_tri
        for i, f_part in enumerate(robctrl.f_param):
            f_name = f_part[0]
            zt_idx = [3 * i, 3 * i + 3]
            # if tacp.is_contact[f_name]:  # Keep at np.zeros(3) if no contact
            if tacp.is_first_contact[f_name]:  # Keep at np.zeros(3) if no contact
                # pos_tac_palm[zt_idx[0]: zt_idx[1]] = tacp.cur_tac[f_name][1][:3]
                pos_tac_palm[zt_idx[0]: zt_idx[1]] = tacp.cur_tac[f_name][1][:3] + np.random.normal(0.00, 0.0, 3)
                R_tac_palm = Rotation.from_rotvec(tacp.cur_tac[f_name][1][3:]).as_matrix()
                nv_tac_palm[zt_idx[0]: zt_idx[1]] = R_tac_palm[:, 0] + np.random.normal(0, 0., 3)
                # nv_tac_palm[zt_idx[0]: zt_idx[1]] = R_tac_palm[:, 0]
        return np.array(pos_tac_palm), np.array(nv_tac_palm)

    def ekf_posteriori(self, x_bar, z_t, h_t, P_state_cov, tacp, robctrl):
        """
        EKF posteriori estimation
        """
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        print('------------Posteriori----------------')
        pn_flag = tac_const.PN_FLAG
        [W1, W2, W3] = robctrl.rotvec_cup_palm  # rotvec of cup in palm frame
        step = 1
        if pn_flag == 'pn':  # use pos and normal as observation variable
            step = 2
        elif pn_flag == 'p':  # use pos only as observation variable
            step = 1
        J_h = np.mat(np.zeros([3*robctrl.f_size*step, 6+3*robctrl.f_size]))
        R_noi = np.mat(np.zeros([3*robctrl.f_size*step, 3*robctrl.f_size*step]))

        for i, f_part in enumerate(robctrl.f_param):
            f_name = f_part[0]
            # if tacp.is_contact[f_name]:
            if tacp.is_first_contact[f_name]:
                R_noi[(i*3)*step: (i*3+3)*step, (i*3)*step: (i*3+3)*step] = np.random.normal(0, 0.02) * np.identity(3*step)
                # print("W1, W2, W3, pos_CO:", W1, W2, W3, robctrl.pos_contact_cup[f_name])
                J_h[i*3:i*3+3, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3,
                                                     pos_CO_x=robctrl.pos_contact_cup[f_name][0],
                                                     pos_CO_y=robctrl.pos_contact_cup[f_name][1],
                                                     pos_CO_z=robctrl.pos_contact_cup[f_name][2])
        K_t = self.scale_kp * np.linalg.pinv(J_h)
        # K_t[0:3, :] = 0.08 * np.linalg.pinv(J_h)[0:3, :]
        # K_t[3:6, :] = 0.001 * np.linalg.pinv(J_h)[3:6, :]
        u, s, v = np.linalg.svd(J_h)
        delta_t = z_t - h_t
        # delta_t[12:24] = 0.0 * (z_t[12:24] - h_t[12:24])
        # delta_t[3*robctrl.f_size: 3*robctrl.f_size*2] = 0.0 * (z_t[3*robctrl.f_size*step: 3*robctrl.f_size*2] - h_t[3*robctrl.f_size: 3*robctrl.f_size*2])
        Update = np.ravel(np.matmul(K_t, delta_t))
        # print("delta_t:", delta_t, "\nUpdate:", Update)
        nonzeroind = np.nonzero(s)[0]
        b = []
        for i in range(len(nonzeroind)):
            if math.fabs(s[nonzeroind[i]] > 0.00001):
                b.append(s[nonzeroind[i]])
        # print("b:", b, "s:", s)
        c = np.array(b)
        # if len(b) and np.amin(c) > 0.01:
        if np.amin(c) > 0.01:
            x_hat = x_bar + Update
            P_state_cov = (np.eye(6 + 3 * robctrl.f_size) - K_t @ J_h) @ P_state_cov
        else:
            x_hat = x_bar
        return x_hat, P_state_cov