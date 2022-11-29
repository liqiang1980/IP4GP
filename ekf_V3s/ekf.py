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

    def state_predictor(self, sim, model, hand_param, object_param, xstate_aug,
                        P_state_cov, tacp, robctrl):
        f_param = hand_param[1:]
        dim_x_aug = len(xstate_aug)
        num_used_f = len(f_param)
        """ Matrix Initialization: F, P, G """
        F_Matrix = np.mat(np.zeros((dim_x_aug, dim_x_aug)))
        Q_state_noise_cov = np.zeros((dim_x_aug, dim_x_aug))
        for i in range(6):
            Q_state_noise_cov[i, i] = math.fabs(np.random.normal(0, 0.001))
        Grasping_matrix = np.zeros([6, 6 * num_used_f])
        G_pinv = np.zeros([6, 6 * num_used_f])
        ju = np.zeros(6 * num_used_f)

        """ 
        G matrix calculate 
        and
        Get the twists of contact tacs: These are equivalent to Jac * delta_u
        """
        for i, f_part in enumerate(f_param):
            f_name = f_part[0]
            tac_name = tacp.cur_tac[f_name][0]  # If no contact, cur-tac info is updated by last-tac
            tac_posrotvec = tacp.cur_tac[f_name][1]
            cur_s = tacp.contact_renew(sim=sim, idx=i, tac_name=tac_name, model="cur",
                                        T_tip_palm=T_tip_palm[i])
            if not tacp.is_contact[f_name]:  # this part is no-contact, pass
                continue
            pos_tac_palm = tac_posrotvec[:3]
            pos_cup_palm = xstate_aug[:3]
            tmp_G = ug.get_Grasp_matrix(pos_tac_palm=pos_tac_palm, pos_cup_palm=pos_cup_palm)
            Grasping_matrix[:, 0 + i * 6: 6 + i * 6] = tmp_G
            # self.J_fingers[0 + i * 6: 6 + i * 6, 0 + i * 4: 4 + i * 4] = self.J[i, :, :]
            ju[6 * i: 6 * i + 6] =

        if is_contact[i]:
            ju.extend(tacp.cur_contact[i][1] - tacp.last_contact[i][1])

        """ G_pinv calculate: 2 type """
        if tac_const.GT_FLAG == '1G':  # 4 G splice and calculate big GT_pinv
            G_pinv = np.linalg.pinv(Grasping_matrix.T)  # Get G_pinv
        elif tac_const.GT_FLAG == '4G':
            for i in range(num_used_f):
                inv_tmp = Grasping_matrix[:, 0 + i * 6: 6 + i * 6]
                Y_RU = np.copy(inv_tmp[0:3, 3:6])
                Y_LD = np.copy(inv_tmp[3:6, 0:3])
                inv_tmp[0:3, 3:6] = Y_LD
                inv_tmp[3:6, 0:3] = Y_RU
                G_pinv[:, 0 + i * 6: 6 + i * 6] = inv_tmp  # 4 GT_inv splice a big G


        prediction = np.matmul(G_pinv, ju)
        if tac_const.GT_FLAG == '4G':
            F_Matrix[:6, :6] = np.mat(np.eye(6))
        prediction = np.append(prediction, [0] * (12 * 3))
        x_bar = xstate_aug + prediction

        P_state_cov = F_Matrix * P_state_cov * F_Matrix.transpose() + Q_state_noise_cov

        return x_bar, P_state_cov

    def observe_computation(self, x_bar, tacperception, sim, object_param):
        # print('measurement equation computation')
        contact_position = []
        contact_nv = []
        # self.fin_num = tacperception.fin_num
        # self.fin_tri = tacperception.fin_tri
        _obj_position = x_bar[:3]  # position of object in palm
        _rot = ug.rotvec_2_Matrix(x_bar[3:6])  # rot of object in palm
        for i in range(4):
            if tacperception.fin_tri[i] == 1:
                contact_position.append(_obj_position.T + np.matmul(_rot, x_bar[6 + i * 3:6 + (i + 1) * 3]))
                ##########Get normal of contact point on the cup
                if int(object_param[3]) == 3:
                    nor_contact_in_cup, res = og.surface_cylinder(x_bar[6 + i * 3], x_bar[6 + i * 3 + 1],
                                                                  x_bar[6 + i * 3 + 2])
                else:
                    nor_contact_in_cup, res = og.surface_cup(x_bar[6 + i * 3], x_bar[6 + i * 3 + 1],
                                                             x_bar[6 + i * 3 + 2])
                nor_contact_in_world = np.matmul(_rot, nor_contact_in_cup)
                contact_nv.append(nor_contact_in_world / np.linalg.norm(nor_contact_in_world))
            else:
                contact_position.append([0, 0, 0])
                contact_nv.append([0, 0, 0])

        return np.array(contact_position), np.array(contact_nv),

    def measure_fb(self, sim, model, hand_param, object_param, x_bar, tacperception):
        # print('measurement feedback from sensing (ground truth + noise in simulation)')
        contact_position = []
        contact_nv = []
        # self.fin_num = tacperception.fin_num
        # self.fin_tri = tacperception.fin_tri

        for i in range(4):
            if tacperception.fin_tri[i] == 1:
                contact_position.append(
                    (tacperception.get_contact_taxel_position(sim, model, hand_param[i + 1][0], "palm_link", "z"))[:3] \
                    + np.random.normal(0.00, 0.0, 3))
                contact_nv.append(tacperception.get_contact_taxel_nv(sim, model,
                                                                     hand_param[i + 1][0], "palm_link") \
                                  + np.random.normal(0, 0., 3))
            else:
                contact_position.append([0, 0, 0])
                contact_nv.append([0, 0, 0])

        return np.array(contact_position), np.array(contact_nv)

    def ekf_posteriori(self, sim, model, viewer, x_bar, z_t, h_t, P_state_cov, tacperception, object_param):
        # print('posterior computation')
        # the jocobian matrix of measurement equation
        [W1, W2, W3] = x_bar[3:6]  # the rotvec of object in palm frame {P}
        pos_c1 = x_bar[6:9]  # the pos of contact point in object frame {O}
        pos_c2 = x_bar[9:12]  # the pos of contact point in object frame {O}
        pos_c3 = x_bar[12:15]  # the pos of contact point in object frame {O}
        pos_c4 = x_bar[15:18]  # the pos of contact point in object frame {O}
        if int(object_param[3]) == 3:
            normal_c1 = og.surface_cylinder(pos_c1[0], pos_c1[1], pos_c1[2])[0]  # the normal of contact point in {O}
            normal_c2 = og.surface_cylinder(pos_c2[0], pos_c2[1], pos_c2[2])[0]  # the normal of contact point in {O}
            normal_c3 = og.surface_cylinder(pos_c3[0], pos_c3[1], pos_c3[2])[0]  # the normal of contact point in {O}
            normal_c4 = og.surface_cylinder(pos_c4[0], pos_c4[1], pos_c4[2])[0]  # the normal of contact point in {O}
        else:
            normal_c1 = og.surface_cup(pos_c1[0], pos_c1[1], pos_c1[2])[0]  # the normal of contact point in {O}
            normal_c2 = og.surface_cup(pos_c2[0], pos_c2[1], pos_c2[2])[0]  # the normal of contact point in {O}
            normal_c3 = og.surface_cup(pos_c3[0], pos_c3[1], pos_c3[2])[0]  # the normal of contact point in {O}
            normal_c4 = og.surface_cup(pos_c4[0], pos_c4[1], pos_c4[2])[0]  # the normal of contact point in {O}
        # print("  normal_c1:", normal_c1)
        pn_flag = tac_const.PN_FLAG
        if pn_flag == 'pn':  # use pos and normal as observation variable
            J_h = np.zeros([6 * 4, 6 + 4 * 3])
            R_noi = np.zeros([24, 24])
            if tacperception.is_ff_contact:
                J_h[:3, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c1[0], pos_CO_y=pos_c1[1],
                                              pos_CO_z=pos_c1[2])
                J_h[12:15, :6] = ug.H_calculator_pn(W1=W1, W2=W2, W3=W3, normal_CO_x=normal_c1[0],
                                                    normal_CO_y=normal_c1[1],
                                                    normal_CO_z=normal_c1[2])
                J_h[12:15, :6] = np.zeros([3, 6])
                R_noi[:6, :6] = np.random.normal(0, 0.002) * np.identity(6)
            if tacperception.is_mf_contact:
                J_h[3:6, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c2[0], pos_CO_y=pos_c2[1],
                                               pos_CO_z=pos_c2[2])
                J_h[15:18, :6] = ug.H_calculator_pn(W1=W1, W2=W2, W3=W3, normal_CO_x=normal_c2[0],
                                                    normal_CO_y=normal_c2[1],
                                                    normal_CO_z=normal_c2[2])
                J_h[15:18, :6] = np.zeros([3, 6])
                R_noi[6:12, 6:12] = np.random.normal(0, 0.002) * np.identity(6)
            if tacperception.is_rf_contact:
                J_h[6:9, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c3[0], pos_CO_y=pos_c3[1],
                                               pos_CO_z=pos_c3[2])
                J_h[18:21, :6] = ug.H_calculator_pn(W1=W1, W2=W2, W3=W3, normal_CO_x=normal_c3[0],
                                                    normal_CO_y=normal_c3[1],
                                                    normal_CO_z=normal_c3[2])
                J_h[18:21, :6] = np.zeros([3, 6])
                R_noi[12:18, 12:18] = np.random.normal(0, 0.002) * np.identity(6)
            if tacperception.is_th_contact:
                J_h[9:12, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c4[0], pos_CO_y=pos_c4[1],
                                                pos_CO_z=pos_c4[2])
                J_h[21:24, :6] = ug.H_calculator_pn(W1=W1, W2=W2, W3=W3, normal_CO_x=normal_c4[0],
                                                    normal_CO_y=normal_c4[1],
                                                    normal_CO_z=normal_c4[2])
                J_h[21:24, :6] = np.zeros([3, 6])
                R_noi[18:24, 18:24] = np.random.normal(0, 0.002) * np.identity(6)
            # K_t = P_state_cov @ J_h.transpose() @ \
            #       np.linalg.pinv(J_h @ P_state_cov @ J_h.transpose() + R_noi)
            K_t = self.scale_kp * np.linalg.pinv(J_h)
            # K_t[0:3, :] = 0.08 * np.linalg.pinv(J_h)[0:3, :]
            # K_t[3:6, :] = 0.001 * np.linalg.pinv(J_h)[3:6, :]
            u, s, v = np.linalg.svd(J_h)
            delta_t = z_t - h_t
            delta_t[12:24] = 0.0 * (z_t[12:24] - h_t[12:24])
            Update = np.ravel(np.matmul(K_t, delta_t))
            # print('update is ', Update)
            nonzeroind = np.nonzero(s)[0]
            b = []
            for i in range(len(nonzeroind)):
                if math.fabs(s[nonzeroind[i]] > 0.00001):
                    b.append(s[nonzeroind[i]])
            # print(b)
            c = np.array(b)
            if np.amin(c) > 0.01:
                x_hat = x_bar + Update
                P_state_cov = (np.eye(6 + 4 * 3) - K_t @ J_h) @ P_state_cov
            else:
                x_hat = x_bar

        elif pn_flag == 'p':  # use pos only as observation variable
            J_h = np.zeros([3 * 4, 6 + 4 * 3])
            R_noi = np.zeros([12, 12])
            if tacperception.is_ff_contact == True:
                J_h[:3, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c1[0], pos_CO_y=pos_c1[1],
                                              pos_CO_z=pos_c1[2])
                R_noi[:3, :3] = np.random.normal(0, 0.002) * np.identity(3)
            if tacperception.is_mf_contact == True:
                J_h[3:6, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c2[0], pos_CO_y=pos_c2[1],
                                               pos_CO_z=pos_c2[2])
                R_noi[3:6, 3:6] = np.random.normal(0, 0.002) * np.identity(3)
            if tacperception.is_rf_contact == True:
                J_h[6:9, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c3[0], pos_CO_y=pos_c3[1],
                                               pos_CO_z=pos_c3[2])
                R_noi[6:9, 6:9] = np.random.normal(0, 0.002) * np.identity(3)
            if tacperception.is_th_contact == True:
                J_h[9:12, :6] = ug.H_calculator(W1=W1, W2=W2, W3=W3, pos_CO_x=pos_c4[0], pos_CO_y=pos_c4[1],
                                                pos_CO_z=pos_c4[2])
                R_noi[9:12, 9:12] = np.random.normal(0, 0.002) * np.identity(3)
            # K_t = P_state_cov @ J_h.transpose() @ \
            #       np.linalg.pinv(J_h @ P_state_cov @ J_h.transpose() + R_noi)
            K_t = self.scale_kp * np.linalg.pinv(J_h)
            u, s, v = np.linalg.svd(J_h)
            # print('s is ', s)
            # print('in posteria P_state_cov before', P_state_cov)
            # print('J_h ', J_h)
            # print('pinv ', np.linalg.pinv(J_h @ P_state_cov @ J_h.transpose() + R_noi))
            # h_t[:3] = -h_t[:3]
            # normal direction of the object is oposite to the contact tacxel
            # h_t[12:] = -h_t[12:]
            Update = np.ravel(np.matmul(K_t, (z_t - h_t)))
            # print('update is ', Update)
            nonzeroind = np.nonzero(s)[0]
            b = []
            for i in range(len(nonzeroind)):
                if math.fabs(s[nonzeroind[i]] > 0.00001):
                    b.append(s[nonzeroind[i]])
            # print(b)
            c = np.array(b)
            if np.amin(c) > 0.01:
                x_hat = x_bar + Update
                P_state_cov = (np.eye(6 + 4 * 3) - K_t @ J_h) @ P_state_cov
            else:
                x_hat = x_bar

            # K_t[0:3, :] = np.linalg.pinv(J_h)[0:3, :]
            # K_t[3:6, :] = 0.01 * np.linalg.pinv(J_h)[3:6, :]
        # the covariance of measurement noise
        # R_noi = np.random.normal(0, 0.01, size=(6 * 4, 6 * 4))
        # R_noi[:3, :3] = np.mat([[0.001, 0.15, 0.2],
        #                         [0.15, 0.001, 0.002],
        #                         [0.2, 0.002, 0.001]])
        # K_t =  np.matmul(np.matmul(P_state_cov, J_h.transpose()), \
        #                  np.linalg.pinv(np.matmul(np.matmul(J_h, P_state_cov), J_h.transpose()) + R_noi))

        # print('Kt is ', K_t)
        # print('in posteria P_state_cov after', P_state_cov)
        return x_hat, P_state_cov
