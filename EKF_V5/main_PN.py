import math
import re
import time
import pathlib
import numpy as np
import PyKDL as kdl
import copy

import scipy
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from mujoco_py import load_model_from_path
import xml.etree.ElementTree as ET
import qgFunc as qg
from qgFunc import if_match, get_poseuler_from_xml
import Plot_plus as pPlt


class ekfDataClass:
    """
    Save params for EKF
    xt: Object State in current round
    xt_pre: Object State in last round
    GD: Ground Truth
    F: State transfer matrix from current round to next round
    P: The covariance matrix of initial state, just give it appropriate large values and it will converge automatically.
    H: Observation matrix. That is, the matrix representation of the apparent measurement h().
    """

    def __init__(self):
        self.xt = np.zeros(6)
        self.xtLong = np.zeros(46)
        self.xt_pre = np.zeros(6)
        self.xt_posquat = np.zeros(7)
        self.xt_all = np.zeros(6)
        self.GD_all = np.zeros(6)
        self.F = np.mat(np.zeros([46, 46]))
        self.P = np.mat(np.ones([46, 46]) * 100)
        self.P_pre = self.P
        self.H = np.mat(np.zeros([24, 46]))
        # self.Q = np.mat(np.zeros([46, 46]))
        self.Q = np.mat(np.eye(46)) * 0.00001  # Noise parameter matrix of state xt. Parameter adjustment required
        # self.Q[:3, :3] = np.mat([[0.1, -15, -2],
        #                          [-15, 10, 15],
        #                          [-2, 15, -1.5]])
        # self.R = np.mat(np.zeros([24, 24]))
        self.R = np.mat(np.eye(24))  # Noise parameter matrix of sensors. Parameter adjustment required
        # self.R[:3, :3] = np.mat([[1, 4, 0],
        #                          [4, 3, 1],
        #                          [0, 1, 1]])
        # self.R[3:6, 3:6] = self.R[:3, :3]
        # self.R[6:9, 6:9] = self.R[:3, :3]
        # self.R[9:12, 9:12] = self.R[:3, :3]
        # self.R[12:15, 12:15] = self.R[:3, :3]
        # self.R[15:18, 15:18] = self.R[:3, :3]
        # self.R[18:21, 18:21] = self.R[:3, :3]
        # self.R[21:24, 21:24] = self.R[:3, :3]
        # self.R = self.R * 300
        self.pos_O0_pre = np.mat(np.zeros(3)).T
        self.pos_O1_pre = np.mat(np.zeros(3)).T
        self.pos_O2_pre = np.mat(np.zeros(3)).T
        self.pos_O3_pre = np.mat(np.zeros(3)).T
        self.normal_O0_pre = np.mat(np.ones(3)).T
        self.normal_O1_pre = np.mat(np.ones(3)).T
        self.normal_O2_pre = np.mat(np.ones(3)).T
        self.normal_O3_pre = np.mat(np.ones(3)).T

        # test Variables
        self.joint_vel_all = np.zeros(16)
        self.first_flag = 0
        self.test_pos_all = np.mat(np.zeros(3))
        self.test_pos_all2 = np.mat(np.zeros(3))
        self.test_pos_all3 = np.mat(np.zeros(3))
        self.test_normal_all = np.mat(np.zeros(3))
        self.test_normal_all2 = np.mat(np.zeros(3))


class MyClass:
    def __init__(self, txt_dir, dataClass, pass_taxel):
        self._dataClass = dataClass
        self.pass_taxel = pass_taxel
        self.this_file_dir = str(pathlib.Path(__file__).parent.absolute())
        self.all_data = np.loadtxt(self.this_file_dir + txt_dir + "all_data.txt")
        self.hand_pose = np.loadtxt(self.this_file_dir + txt_dir + "hand_pose.txt")
        self.xml_path = "./UR5/UR5_allegro_test.xml"
        self.model = load_model_from_path(self.xml_path)
        self.xml_tree = ET.parse(self.xml_path)
        self.xml_root = self.xml_tree.getroot()
        self.hand_description = URDF.from_xml_file(
            self.this_file_dir + "/allegro_tactile_description/allegro_hand_description_right_tactile_for_ik_viz.urdf")
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)
        # init kdl_JntArrays for each finger
        self.index_qpos = kdl.JntArray(4)
        self.mid_qpos = kdl.JntArray(4)
        self.ring_qpos = kdl.JntArray(4)
        self.thumb_qpos = kdl.JntArray(4)
        kdl.SetToZero(self.index_qpos)
        kdl.SetToZero(self.mid_qpos)
        kdl.SetToZero(self.ring_qpos)
        kdl.SetToZero(self.thumb_qpos)
        # init joint_vels
        self.index_vel = np.zeros(4)
        self.mid_vel = np.zeros(4)
        self.ring_vel = np.zeros(4)
        self.thumb_vel = np.zeros(4)
        # init kdl_Frames for each finger
        self.index_pos = kdl.Frame()  # Construct an identity frame
        self.mid_pos = kdl.Frame()  # Construct an identity frame
        self.ring_pos = kdl.Frame()  # Construct an identity frame
        self.thumb_pos = kdl.Frame()  # Construct an identity frame
        # chain
        self.index_chain = self.hand_tree.getChain("palm_link", "index_tip_tactile")
        self.mid_chain = self.hand_tree.getChain("palm_link", "mid_tip_tactile")
        self.ring_chain = self.hand_tree.getChain("palm_link", "ring_tip_tactile")
        self.thumb_chain = self.hand_tree.getChain("palm_link", "thumb_tip_tactile")
        # forward kinematicsallegro_jstates
        self.index_fk = kdl.ChainFkSolverPos_recursive(self.index_chain)
        self.mid_fk = kdl.ChainFkSolverPos_recursive(self.mid_chain)
        self.ring_fk = kdl.ChainFkSolverPos_recursive(self.ring_chain)
        self.thumb_fk = kdl.ChainFkSolverPos_recursive(self.thumb_chain)
        # KDLKinematics
        self.index_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "index_tip_tactile")
        self.mid_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "mid_tip_tactile")
        self.ring_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "ring_tip_tactile")
        self.thumb_kdl_kin = KDLKinematics(self.hand_description, "palm_link", "thumb_tip_tactile")
        # Other params
        self.T_palm_d = np.zeros([4, 4])

    def fk_dealer(self):
        """
        Get T (tips in palm) and J by FK method
        joint positions are updated in main_process()
        """
        M = np.mat(np.zeros((3, 3)))
        p = np.zeros([3, 1])
        self.T_palm_d = qg.posquat2T(self.hand_pose)  # T: palm in world frame {W}
        index_in_palm_T = np.mat(np.eye(4))
        mid_in_palm_T = np.mat(np.eye(4))
        ring_in_palm_T = np.mat(np.eye(4))
        thumb_in_palm_T = np.mat(np.eye(4))

        # forward kinematics
        qg.kdl_calc_fk(self.index_fk, self.index_qpos, self.index_pos)
        M[0, 0] = copy.deepcopy(self.index_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.index_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.index_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.index_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.index_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.index_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.index_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.index_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.index_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.index_pos.p[0])
        p[1, 0] = copy.deepcopy(self.index_pos.p[1])
        p[2, 0] = copy.deepcopy(self.index_pos.p[2])
        index_in_palm_T[:3, :3] = M
        index_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.mid_fk, self.mid_qpos, self.mid_pos)
        M[0, 0] = copy.deepcopy(self.mid_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.mid_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.mid_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.mid_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.mid_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.mid_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.mid_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.mid_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.mid_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.mid_pos.p[0])
        p[1, 0] = copy.deepcopy(self.mid_pos.p[1])
        p[2, 0] = copy.deepcopy(self.mid_pos.p[2])
        mid_in_palm_T[:3, :3] = M
        mid_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.ring_fk, self.ring_qpos, self.ring_pos)
        M[0, 0] = copy.deepcopy(self.ring_pos.M[0, 0])
        M[0, 1] = copy.deepcopy(self.ring_pos.M[0, 1])
        M[0, 2] = copy.deepcopy(self.ring_pos.M[0, 2])
        M[1, 0] = copy.deepcopy(self.ring_pos.M[1, 0])
        M[1, 1] = copy.deepcopy(self.ring_pos.M[1, 1])
        M[1, 2] = copy.deepcopy(self.ring_pos.M[1, 2])
        M[2, 0] = copy.deepcopy(self.ring_pos.M[2, 0])
        M[2, 1] = copy.deepcopy(self.ring_pos.M[2, 1])
        M[2, 2] = copy.deepcopy(self.ring_pos.M[2, 2])
        p[0, 0] = copy.deepcopy(self.ring_pos.p[0])
        p[1, 0] = copy.deepcopy(self.ring_pos.p[1])
        p[2, 0] = copy.deepcopy(self.ring_pos.p[2])
        ring_in_palm_T[:3, :3] = M
        ring_in_palm_T[:3, 3] = p

        qg.kdl_calc_fk(self.thumb_fk, self.thumb_qpos, self.thumb_pos)
        M[0, 0] = self.thumb_pos.M[0, 0]
        M[0, 1] = self.thumb_pos.M[0, 1]
        M[0, 2] = self.thumb_pos.M[0, 2]
        M[1, 0] = self.thumb_pos.M[1, 0]
        M[1, 1] = self.thumb_pos.M[1, 1]
        M[1, 2] = self.thumb_pos.M[1, 2]
        M[2, 0] = self.thumb_pos.M[2, 0]
        M[2, 1] = self.thumb_pos.M[2, 1]
        M[2, 2] = self.thumb_pos.M[2, 2]
        p[0, 0] = self.thumb_pos.p[0]
        p[1, 0] = self.thumb_pos.p[1]
        p[2, 0] = self.thumb_pos.p[2]
        thumb_in_palm_T[:3, :3] = M
        thumb_in_palm_T[:3, 3] = p

        return index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T

    def Jacobian_dealer(self, joint_pos, pos0, pos1, pos2, pos3):
        jstate = joint_pos
        """ Hand to finger tips Jacobians: """
        pos0 = np.ravel(pos0)
        pos1 = np.ravel(pos1)
        pos2 = np.ravel(pos2)
        pos3 = np.ravel(pos3)
        J_index = copy.deepcopy(self.index_kdl_kin.jacobian(q=jstate[0:4], pos=pos0))
        # J_index = copy.deepcopy(self.index_kdl_kin.jacobian(q=jstate[0:4]))
        J_index = np.mat(J_index)
        J_mid = copy.deepcopy(self.mid_kdl_kin.jacobian(q=jstate[4:8], pos=pos1))
        # J_mid = copy.deepcopy(self.mid_kdl_kin.jacobian(q=jstate[4:8]))
        J_mid = np.mat(J_mid)
        J_ring = copy.deepcopy(self.ring_kdl_kin.jacobian(q=jstate[8:12], pos=pos2))
        # J_ring = copy.deepcopy(self.ring_kdl_kin.jacobian(q=jstate[8:12]))
        J_ring = np.mat(J_ring)
        J_thumb = copy.deepcopy(self.thumb_kdl_kin.jacobian(q=jstate[12:16], pos=pos3))
        # J_thumb = copy.deepcopy(self.thumb_kdl_kin.jacobian(q=jstate[12:16]))
        J_thumb = np.mat(J_thumb)
        return J_index, J_mid, J_ring, J_thumb

    def main_process(self):
        self.all_data = self.all_data[120:]  # Discard invalid values
        print("Dim:", self.all_data.shape)
        time_last = 0
        time_now = 0
        joint_pos_now = np.zeros(16)
        joint_pos_last = np.zeros(16)
        for i in range(self.all_data.shape[0]):
            data_now = self.all_data[i]
            time_stamp = data_now[0]
            object_posquat = data_now[1:8]
            object_posrpy = qg.posquat2posrpy(object_posquat)
            joint_pos = data_now[8:24]
            joint_vel = data_now[24:40]
            taxel_data = data_now[40:]
            print("==" * 30, i, ":", time_stamp)
            time_now = time_stamp
            if i == 0:
                self._dataClass.xt = np.ravel(qg.posquat2posrpy(object_posquat))
                self._dataClass.xt_all = self._dataClass.xt
                self._dataClass.GD_all = object_posrpy
                continue

            """
            Set joint positions
            """
            self.index_qpos[0] = joint_pos[0]
            self.index_qpos[1] = joint_pos[1]
            self.index_qpos[2] = joint_pos[2]
            self.index_qpos[3] = joint_pos[3]
            self.mid_qpos[0] = joint_pos[4]
            self.mid_qpos[1] = joint_pos[5]
            self.mid_qpos[2] = joint_pos[6]
            self.mid_qpos[3] = joint_pos[7]
            self.ring_qpos[0] = joint_pos[8]
            self.ring_qpos[1] = joint_pos[9]
            self.ring_qpos[2] = joint_pos[10]
            self.ring_qpos[3] = joint_pos[11]
            self.thumb_qpos[0] = joint_pos[12]
            self.thumb_qpos[1] = joint_pos[13]
            self.thumb_qpos[2] = joint_pos[14]
            self.thumb_qpos[3] = joint_pos[15]
            joint_pos_now = joint_pos

            """
            Get joint_vels
            """
            delta_time = time_now - time_last
            time_last = time_now
            delta_pos = joint_pos_now - joint_pos_last
            joint_pos_last = joint_pos_now
            joint_vel = delta_pos / delta_time
            self.index_vel = joint_vel[:4]
            self.mid_vel = joint_vel[4:8]
            self.ring_vel = joint_vel[8:12]
            self.thumb_vel = joint_vel[12:]
            print("    Check joint vel (rad/s):\n    ", self.index_vel, "\n    ", self.mid_vel, "\n    ", self.ring_vel,
                  "\n    ", self.thumb_vel)

            """
            Get names of active taxel
            """
            num_taxel = np.where(taxel_data[:540] > 0)[0]
            name_taxel = np.empty(0)
            for j in range(num_taxel.shape[0]):  # Traverse all taxel in round i
                name_taxel = np.hstack((name_taxel, self.model._sensor_id2name[num_taxel[j]]))
            print("   all taxel  :", name_taxel)

            """
            Get pos, euler, R, T of active taxel.
            Number 0-3 refers to Index, Mid, Ring, Thumb.
            """
            count_taxel = np.zeros(4)
            increasing_pos0 = np.mat(np.zeros(3))
            increasing_pos1 = np.mat(np.zeros(3))
            increasing_pos2 = np.mat(np.zeros(3))
            increasing_pos3 = np.mat(np.zeros(3))
            increasing_euler0 = np.mat(np.zeros(3))
            increasing_euler1 = np.mat(np.zeros(3))
            increasing_euler2 = np.mat(np.zeros(3))
            increasing_euler3 = np.mat(np.zeros(3))
            tip_flag_outer = np.zeros(4)  # Mark which fingertip was triggered in round i
            for jj in range(name_taxel.shape[0]):  # Traverse all taxel in round i
                name_taxel_tmp = name_taxel[jj]
                tip_flag = np.zeros(4)  # Mark which fingertip the current Taxel is on
                print(">>>", name_taxel_tmp, "<<<")
                # Skip invalid taxels
                if (self.pass_taxel == name_taxel_tmp).any():
                    print("   PASS   ")
                    continue

                # Need to know the taxel in which tip
                if re.search("touch_0_", name_taxel_tmp) is not None:
                    print("Index tip touches !")
                    count_taxel[0] += 1
                    tip_flag[0] = 1
                    tip_flag_outer[0] = 1
                elif re.search('touch_7_', name_taxel_tmp) is not None:
                    print("Mid tip touches !!")
                    count_taxel[1] += 1
                    tip_flag[1] = 1
                    tip_flag_outer[1] = 1
                elif re.search('touch_11_', name_taxel_tmp) is not None:
                    print("Ring tip touches !!!")
                    count_taxel[2] += 1
                    tip_flag[2] = 1
                    tip_flag_outer[2] = 1
                elif re.search('touch_15_', name_taxel_tmp) is not None:
                    print("Thumb tip touches !!!!")
                    count_taxel[3] += 1
                    tip_flag[3] = 1
                    tip_flag_outer[3] = 1
                else:  # Skip taxels which is not in the tips
                    print("          Skip")
                    continue

                # Get pos, euler, R of taxels, in tip frame.
                pos, euler = self.get_taxel_poseuler(taxel_name=name_taxel_tmp)  # Taxel in tip
                pos = np.fromstring(pos, dtype=float, sep=' ')
                euler = np.fromstring(euler, dtype=float, sep=' ')
                print('TOUCH_poseuler_array:', pos, euler)

                # Stack in diff situation
                if tip_flag[0] == 1:
                    increasing_pos0 = np.vstack((increasing_pos0, pos))
                    increasing_euler0 = np.vstack((increasing_euler0, euler))
                elif tip_flag[1] == 1:
                    increasing_pos1 = np.vstack((increasing_pos1, pos))
                    increasing_euler1 = np.vstack((increasing_euler1, euler))
                elif tip_flag[2] == 1:
                    increasing_pos2 = np.vstack((increasing_pos2, pos))
                    increasing_euler2 = np.vstack((increasing_euler2, euler))
                elif tip_flag[3] == 1:
                    increasing_pos3 = np.vstack((increasing_pos3, pos))
                    increasing_euler3 = np.vstack((increasing_euler3, euler))

            total_taxel = np.sum(count_taxel)
            """
            Use ut(e.g. joint_vel) to exclude tips which is not contact
            """
            for k in range(4):
                if tip_flag_outer[k] == 0:
                    joint_vel[k * 4:k * 4 + 4] = np.zeros(4)
            print("     Check before T & J, joint vel:", joint_vel)

            """
            If any contact point exists, predict.
            """
            if total_taxel != 0:
                increasing_pos0 = increasing_pos0[1:]
                increasing_pos1 = increasing_pos1[1:]
                increasing_pos2 = increasing_pos2[1:]
                increasing_pos3 = increasing_pos3[1:]
                increasing_euler0 = increasing_euler0[1:]
                increasing_euler1 = increasing_euler1[1:]
                increasing_euler2 = increasing_euler2[1:]
                increasing_euler3 = increasing_euler3[1:]

                _mean_pos0 = qg.get_mean_3D(increasing_pos0)
                _mean_pos1 = qg.get_mean_3D(increasing_pos1)
                _mean_pos2 = qg.get_mean_3D(increasing_pos2)
                _mean_pos3 = qg.get_mean_3D(increasing_pos3)
                _mean_euler0 = qg.get_mean_3D(increasing_euler0)
                _mean_euler1 = qg.get_mean_3D(increasing_euler1)
                _mean_euler2 = qg.get_mean_3D(increasing_euler2)
                _mean_euler3 = qg.get_mean_3D(increasing_euler3)

                mean_pos0 = np.zeros(3)
                mean_pos1 = np.zeros(3)
                mean_pos2 = np.zeros(3)
                mean_pos3 = np.zeros(3)
                mean_euler0 = np.zeros(3)
                mean_euler1 = np.zeros(3)
                mean_euler2 = np.zeros(3)
                mean_euler3 = np.zeros(3)
                if increasing_pos0.size != 0:
                    mean_pos0, loc0 = qg.choose_taxel(_mean_pos0, increasing_pos0)
                    mean_euler0 = increasing_euler0[loc0]
                if increasing_pos1.size != 0:
                    mean_pos1, loc1 = qg.choose_taxel(_mean_pos1, increasing_pos1)
                    mean_euler1 = increasing_euler0[loc1]
                if increasing_pos2.size != 0:
                    mean_pos2, loc2 = qg.choose_taxel(_mean_pos2, increasing_pos2)
                    mean_euler2 = increasing_euler0[loc2]
                if increasing_pos3.size != 0:
                    mean_pos3, loc3 = qg.choose_taxel(_mean_pos3, increasing_pos3)
                    mean_euler3 = increasing_euler0[loc3]

                """
                FK: Get T (tips in palm frame {P})
                """
                index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T = self.fk_dealer()

                """
                Exclude T of tips which is not contact
                """
                if tip_flag_outer[0] == 0:
                    index_in_palm_T = np.mat(np.zeros([4, 4]))
                if tip_flag_outer[1] == 0:
                    mid_in_palm_T = np.mat(np.zeros([4, 4]))
                if tip_flag_outer[2] == 0:
                    ring_in_palm_T = np.mat(np.zeros([4, 4]))
                if tip_flag_outer[3] == 0:
                    thumb_in_palm_T = np.mat(np.zeros([4, 4]))

                T_taxel_in_palm0, T_taxel_in_W0 = self.get_T_taxel(pos=mean_pos0, euler=mean_euler0,
                                                                   T_tip_in_palm=index_in_palm_T)
                T_taxel_in_palm1, T_taxel_in_W1 = self.get_T_taxel(pos=mean_pos1, euler=mean_euler1,
                                                                   T_tip_in_palm=mid_in_palm_T)
                T_taxel_in_palm2, T_taxel_in_W2 = self.get_T_taxel(pos=mean_pos2, euler=mean_euler2,
                                                                   T_tip_in_palm=ring_in_palm_T)
                T_taxel_in_palm3, T_taxel_in_W3 = self.get_T_taxel(pos=mean_pos3, euler=mean_euler3,
                                                                   T_tip_in_palm=thumb_in_palm_T)
                pos_taxel_palm0 = np.ravel(T_taxel_in_palm0[:3, 3].T)
                pos_taxel_palm1 = np.ravel(T_taxel_in_palm1[:3, 3].T)
                pos_taxel_palm2 = np.ravel(T_taxel_in_palm2[:3, 3].T)
                pos_taxel_palm3 = np.ravel(T_taxel_in_palm3[:3, 3].T)
                J_index, J_mid, J_ring, J_thumb = self.Jacobian_dealer(joint_pos, pos_taxel_palm0, pos_taxel_palm1,
                                                                       pos_taxel_palm2, pos_taxel_palm3)

                """
                Get Grasp Matrix G in round i
                """
                big_G = np.mat(np.zeros([6, 24]))
                # Index Finger:
                R = T_taxel_in_W0[:3, :3]  # R: The rotation mat of taxel in world frame {W}
                p = self._dataClass.xt[:3]  # p: The position of object in {W}. Get it from prediction.
                ci = mean_pos0  # ci: The position of taxel in {W}.
                G = qg.get_G(R, p, ci)
                big_G[:, :6] = G
                # Mid Finger:
                R = T_taxel_in_W1[:3, :3]  # R: The rotation mat of taxel in world frame {W}
                p = self._dataClass.xt[:3]  # p: The position of object in {W}. Get it from prediction.
                ci = mean_pos1  # ci: The position of taxel in {W}.
                G = qg.get_G(R, p, ci)
                big_G[:, 6:12] = G
                # Ring Finger:
                R = T_taxel_in_W2[:3, :3]  # R: The rotation mat of taxel in world frame {W}
                p = self._dataClass.xt[:3]  # p: The position of object in {W}. Get it from prediction.
                ci = mean_pos2  # ci: The position of taxel in {W}.
                G = qg.get_G(R, p, ci)
                big_G[:, 12:18] = G
                # Thumb Finger:
                R = T_taxel_in_W3[:3, :3]  # R: The rotation mat of taxel in world frame {W}
                p = self._dataClass.xt[:3]  # p: The position of object in {W}. Get it from prediction.
                ci = mean_pos3  # ci: The position of taxel in {W}.
                G = qg.get_G(R, p, ci)
                big_G[:, 18:] = G

                """
                Get Jacobian Matrix J in round i
                (Temporarily use J_tips instead of J_taxels)
                """
                big_J = np.mat(np.zeros([24, 16]))
                big_J[:6, :4] = J_index
                big_J[6:12, 4:8] = J_mid
                big_J[12:18, 8:12] = J_ring
                big_J[18:, 12:] = J_thumb

                print("????????xt:", self._dataClass.xt)
                """
                Prediction
                """
                F_part = np.dot(np.linalg.pinv(big_G).T, big_J) * delta_time
                Prediction = np.dot(F_part, joint_vel)
                print("    >>Forward Prediction", Prediction)
                # Before this line, all xt is the state of last round
                self._dataClass.xt = np.ravel(self._dataClass.xt - Prediction)  # x_t: object poseuler in {W}
                self._dataClass.xtLong[:6] = self._dataClass.xt
                self._dataClass.xtLong[6:22] = joint_vel
                # print("   ??after predict???xt:", self._dataClass.xt, self._dataClass.xtLong)

                T_object_in_W = qg.posrpy2T(self._dataClass.xt)
                self.posterior(T_object_in_W, T_taxel_in_W0, T_taxel_in_W1, T_taxel_in_W2, T_taxel_in_W3, F_part,
                               tip_flag_outer)
            # round i is over

            # print("King1", self._dataClass.test_pos_all)
            # print("King2", self._dataClass.test_pos_all2)
            # print("King3 - King1", self._dataClass.test_pos_all3 - self._dataClass.test_pos_all)
            # print("normal_King2 - normal_King1", self._dataClass.test_normal_all2 - self._dataClass.test_normal_all)

            """
            Data save
            """
            self._dataClass.xt_all = np.vstack((self._dataClass.xt_all, self._dataClass.xt))
            self._dataClass.GD_all = np.vstack((self._dataClass.GD_all, object_posrpy))
            self._dataClass.joint_vel_all = np.vstack((self._dataClass.joint_vel_all, joint_vel))
        self._dataClass.xt_all = self._dataClass.xt_all[1:]
        self._dataClass.GD_all = self._dataClass.GD_all[1:]
        self._dataClass.joint_vel_all = self._dataClass.joint_vel_all[1:]
        """
        Plot
        """
        pPlt.plot_xt_GD_6in1(self._dataClass.xt_all, self._dataClass.GD_all,
                             label1='x[mm]', label2='y[mm]', label3='z[mm]',
                             label4='rx[deg]', label5='ry[deg]', label6='rz[deg]')
        # pPlt.plot_4_joint_vel(self._dataClass.joint_vel_all)

    def posterior(self, T_O_W, T0, T1, T2, T3, F_part, tip_flag_outer):
        """
        input: T_O_W, T_object in world frame {W}.
        input: T0~T3, T_taxel in world frame {W}.
        input: pos0~pos3, taxel positions in world frame {W}.
        EKF posterior update.
        """
        self._dataClass.F = qg.get_bigF_PN(F_part=F_part)

        self._dataClass.P = qg.matmul3mat(self._dataClass.F, self._dataClass.P_pre,
                                          self._dataClass.F.T) + self._dataClass.Q

        """
        Use taxels in last round and object pose in current round as h()
        Use taxels in current round and object pose in current round as z
        calculate error = z - h()
        """
        R_O_W = T_O_W[:3, :3]  # R_object in {W} in current round
        pos_object_W = T_O_W[:3, 3]
        normal_object_W = qg.R2normal(R_O_W)
        T_taxel_O0 = np.dot(np.linalg.inv(T_O_W), T0)  # T_taxel in {O} in current round
        T_taxel_O1 = np.dot(np.linalg.inv(T_O_W), T1)  # T_taxel in {O} in current round
        T_taxel_O2 = np.dot(np.linalg.inv(T_O_W), T2)  # T_taxel in {O} in current round
        T_taxel_O3 = np.dot(np.linalg.inv(T_O_W), T3)  # T_taxel in {O} in current round
        pos_taxel_O0 = T_taxel_O0[:3, 3]  # taxel pos in {O} in current round
        pos_taxel_O1 = T_taxel_O1[:3, 3]  # taxel pos in {O} in current round
        pos_taxel_O2 = T_taxel_O2[:3, 3]  # taxel pos in {O} in current round
        pos_taxel_O3 = T_taxel_O3[:3, 3]  # taxel pos in {O} in current round
        normal_O0 = qg.get_normal_O(pos_taxel_O0)  # taxel normal in {O} in current round
        normal_O1 = qg.get_normal_O(pos_taxel_O1)  # taxel normal in {O} in current round
        normal_O2 = qg.get_normal_O(pos_taxel_O2)  # taxel normal in {O} in current round
        normal_O3 = qg.get_normal_O(pos_taxel_O3)  # taxel normal in {O} in current round

        pos_taxel_W0_h = np.matmul(R_O_W, self._dataClass.pos_O0_pre)  # last taxel in current object, taxel pos in {W}
        pos_taxel_W1_h = np.matmul(R_O_W, self._dataClass.pos_O1_pre)  # last taxel in current object, taxel pos in {W}
        pos_taxel_W2_h = np.matmul(R_O_W, self._dataClass.pos_O2_pre)  # last taxel in current object, taxel pos in {W}
        pos_taxel_W3_h = np.matmul(R_O_W, self._dataClass.pos_O3_pre)  # last taxel in current object, taxel pos in {W}
        normal_W0_h = np.matmul(R_O_W,
                                self._dataClass.normal_O0_pre)  # last taxel in current object, taxel normal in {W}
        normal_W1_h = np.matmul(R_O_W,
                                self._dataClass.normal_O1_pre)  # last taxel in current object, taxel normal in {W}
        normal_W2_h = np.matmul(R_O_W,
                                self._dataClass.normal_O2_pre)  # last taxel in current object, taxel normal in {W}
        normal_W3_h = np.matmul(R_O_W,
                                self._dataClass.normal_O3_pre)  # last taxel in current object, taxel normal in {W}

        # Save pos & normal as last round pos & normal
        self._dataClass.pos_O0_pre = pos_taxel_O0
        self._dataClass.pos_O1_pre = pos_taxel_O1
        self._dataClass.pos_O2_pre = pos_taxel_O2
        self._dataClass.pos_O3_pre = pos_taxel_O3
        self._dataClass.normal_O0_pre = normal_O0
        self._dataClass.normal_O1_pre = normal_O1
        self._dataClass.normal_O2_pre = normal_O2
        self._dataClass.normal_O3_pre = normal_O3

        # Calculate measurement vector z
        pos_taxel_W0 = T0[:3, 3]  # taxel pos in {W} in current round
        pos_taxel_W1 = T1[:3, 3]  # taxel pos in {W} in current round
        pos_taxel_W2 = T2[:3, 3]  # taxel pos in {W} in current round
        pos_taxel_W3 = T3[:3, 3]  # taxel pos in {W} in current round
        normal_W0_a = np.matmul(R_O_W, normal_O0)  # taxel normal in {W} in current round
        normal_W1_a = np.matmul(R_O_W, normal_O1)  # taxel normal in {W} in current round
        normal_W2_a = np.matmul(R_O_W, normal_O2)  # taxel normal in {W} in current round
        normal_W3_a = np.matmul(R_O_W, normal_O3)  # taxel normal in {W} in current round
        """
        Make normal_z and normal_h on the same scale
        """
        normal_W0 = normal_W0_a * math.sqrt(normal_W0_a[0] ** 2 + normal_W0_a[1] ** 2 + normal_W0_a[2] ** 2) \
                    / math.sqrt(normal_W0_h[0] ** 2 + normal_W0_h[1] ** 2 + normal_W0_h[2] ** 2)
        normal_W1 = normal_W1_a * math.sqrt(normal_W1_a[0] ** 2 + normal_W1_a[1] ** 2 + normal_W1_a[2] ** 2) \
                    / math.sqrt(normal_W1_h[0] ** 2 + normal_W1_h[1] ** 2 + normal_W1_h[2] ** 2)
        normal_W2 = normal_W2_a * math.sqrt(normal_W2_a[0] ** 2 + normal_W2_a[1] ** 2 + normal_W2_a[2] ** 2) \
                    / math.sqrt(normal_W2_h[0] ** 2 + normal_W2_h[1] ** 2 + normal_W2_h[2] ** 2)
        normal_W3 = normal_W3_a * math.sqrt(normal_W3_a[0] ** 2 + normal_W3_a[1] ** 2 + normal_W3_a[2] ** 2) \
                    / math.sqrt(normal_W3_h[0] ** 2 + normal_W3_h[1] ** 2 + normal_W3_h[2] ** 2)
        z0 = qg.get_z_or_h('z0', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3,
                           normal_W0_a, normal_W1_a, normal_W2_a, normal_W3_a)
        z = qg.get_z_or_h('z', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3,
                          normal_W0, normal_W1, normal_W2, normal_W3)
        test_z = qg.get_z_or_h_PN('test_z', pos_taxel_W0, pos_taxel_W1, pos_taxel_W2, pos_taxel_W3, normal_W0,
                                  normal_W1, normal_W2, normal_W3)

        """
        Set pos_taxel in {O} just once. Assume that there is not any sliding.
        """
        if not self._dataClass.first_flag:
            self._dataClass.xtLong[22:25] = pos_taxel_O0.T
            self._dataClass.xtLong[25:28] = pos_taxel_O1.T
            self._dataClass.xtLong[28:31] = pos_taxel_O2.T
            self._dataClass.xtLong[31:34] = pos_taxel_O3.T
            self._dataClass.xtLong[34:37] = normal_O0.T
            self._dataClass.xtLong[37:40] = normal_O1.T
            self._dataClass.xtLong[40:43] = normal_O2.T
            self._dataClass.xtLong[43:46] = normal_O3.T
            self._dataClass.first_flag = 1
        self._dataClass.H = qg.get_bigH_PN(R_O_W)  # 24*46
        H = self._dataClass.H
        # print("     $$ mat_H:", H)

        # test H by index finger
        # pos_taxel_W, normal_taxel_W = qg.testH(self._dataClass.xtLong, pos_object_W, pos_taxel_W0_h, R_O_W)
        # pos_taxel_W = np.ravel(pos_taxel_W)
        # normal_taxel_W = np.ravel(normal_taxel_W)
        # self._dataClass.test_pos_all = np.vstack((self._dataClass.test_pos_all, pos_taxel_W))
        # self._dataClass.test_pos_all3 = np.vstack((self._dataClass.test_pos_all3, np.ravel(pos_taxel_W0_h)))
        # self._dataClass.test_normal_all = np.vstack((self._dataClass.test_normal_all, normal_taxel_W))
        # self._dataClass.test_normal_all2 = np.vstack((self._dataClass.test_normal_all2, normal_W0_h.T))

        test_h = np.matmul(H, self._dataClass.xtLong).T
        print("### test: h = H * xt (.T):", test_h.T)
        h = qg.get_z_or_h('h', pos_taxel_W0_h, pos_taxel_W1_h, pos_taxel_W2_h, pos_taxel_W3_h,
                          normal_W0_h, normal_W1_h, normal_W2_h, normal_W3_h)

        error = test_z - test_h
        """
        Exclude tips which is not contact
        """
        for k in range(4):
            if tip_flag_outer[k] == 0:
                error[k * 3:k * 3 + 3, 0] = np.mat(np.zeros(3)).T
                error[k * 3 + 12:k * 3 + 15, 0] = np.mat(np.zeros(3)).T
        print("$$ error:", error.T)

        """
        The remaining steps of posterior EKF
        """
        S = qg.matmul3mat(H, self._dataClass.P, H.T) + self._dataClass.R
        K = qg.matmul3mat(self._dataClass.P, H.T, np.linalg.inv(S))
        print("     $$ mat_S, mat_K:", S.shape, K.shape)
        print("     $$ mat_P:", self._dataClass.P.shape)

        Update = np.matmul(K, error).T
        # print("    >>Posterior Update", Update.shape)
        self._dataClass.xtLong = self._dataClass.xtLong - Update
        self._dataClass.xtLong = np.ravel(self._dataClass.xtLong)
        self._dataClass.xt = self._dataClass.xtLong[:6]
        # print("   ??after update???xt:", self._dataClass.xt, self._dataClass.xtLong)
        self._dataClass.P_pre = np.matmul((np.mat(np.eye(46)) - np.matmul(K, H)), self._dataClass.P)

    def get_T_taxel(self, pos, euler, T_tip_in_palm):
        """
        Translate poseuler to T
        """
        R_taxl_in_tip = qg.euler2R(euler)  # Taxel in tip
        T_taxel_in_tip = np.mat(np.eye(4))
        T_taxel_in_tip[:3, :3] = R_taxl_in_tip
        T_taxel_in_tip[:3, 3] = np.mat(pos).T
        # print("T_taxel_in_tip:", T_taxel_in_tip)
        T_taxel_in_palm = np.dot(T_tip_in_palm, T_taxel_in_tip)
        # print("T_taxel_in_palm:", T_taxel_in_palm)
        T_taxel_in_W = np.matmul(self.T_palm_d, T_taxel_in_palm)
        # print("T_taxel_in_W:", T_taxel_in_W)
        return T_taxel_in_palm, T_taxel_in_W

    def get_taxel_poseuler(self, taxel_name):
        """
        Input taxel_name string, return pos and euler.
        """
        pos = np.zeros(3)
        euler = np.zeros(3)
        nodes = self.xml_root.findall('worldbody/body')
        for child in nodes:
            # print(child.tag, ":", child.attrib)
            if child.get('name') == {'name': 'box_link'}.get('name'):  # Get 'box_link' from all worldbody/body
                # If there is only one <body>, use find() or findall()
                # print("OK", child.tag, ":", child.attrib)
                childnodes = child.findall('body')[0]
                # print(childnodes.tag, ':', childnodes.attrib)
                childnodes1 = childnodes.findall('body')[0]
                # print(childnodes1.tag, ':', childnodes1.attrib)
                childnodes2 = childnodes1.findall('body')[0]
                # print(childnodes2.tag, ':', childnodes2.attrib)
                childnodes3 = childnodes2.findall('body')[0]
                # print(childnodes3.tag, ':', childnodes3.attrib)
                childnodes4 = childnodes3.findall('body')[0]
                # print(childnodes4.tag, ':', childnodes4.attrib)
                childnodes5 = childnodes4.findall('body')[0]
                # print(childnodes5.tag, ':', childnodes5.attrib)
                childnodes6 = childnodes5.findall('body')[0]
                # print(childnodes6.tag, ':', childnodes6.attrib)
                childnodes7 = childnodes6.findall('body')[0]
                # print(childnodes7.tag, ':', childnodes7.attrib)
                childnodes8 = childnodes7.findall('body')[0]
                # print("c8:", childnodes8.tag, ':', childnodes8.attrib)
                # childnodes9 = childnodes8.findall('body')[0]
                for childnodes9 in childnodes8:
                    # print("c9:", childnodes9.tag, ':', childnodes9.attrib)
                    # childnodes10 = childnodes9.findall('body')[0]
                    for childnodes10 in childnodes9.findall('body'):
                        # print("c10", childnodes10.tag, ':', childnodes10.attrib)
                        for childnodes11 in childnodes10.findall('body'):
                            # print("  c11:", childnodes11.tag, ":", childnodes11.attrib)
                            if if_match(nodelist=childnodes11, name=taxel_name):
                                pos, euler = get_poseuler_from_xml(nodelist=childnodes11)
                                break
                            for childnodes12 in childnodes11.findall('body'):
                                # print("    c12", childnodes12.tag, ":", childnodes12.attrib)
                                for childnodes13 in childnodes12.findall('body'):
                                    # print("      @13", childnodes13.tag, ":", childnodes13.attrib)
                                    for childnodes14 in childnodes13.findall('body'):
                                        # print("        @@14", childnodes14.tag, ":", childnodes14.attrib)
                                        if if_match(nodelist=childnodes14, name=taxel_name):
                                            pos, euler = get_poseuler_from_xml(childnodes14)
                                            break
                                        for childnodes15 in childnodes14.findall('body'):
                                            # print("          @@@15", childnodes15.tag, ":", childnodes15.attrib)
                                            if if_match(nodelist=childnodes15, name=taxel_name):
                                                pos, euler = get_poseuler_from_xml(childnodes15)
                                                break
                                            for childnodes16 in childnodes15.findall('body'):
                                                # print("            $16", childnodes16.tag, ":", childnodes16.attrib)
                                                if if_match(nodelist=childnodes16, name=taxel_name):
                                                    pos, euler = get_poseuler_from_xml(childnodes16)
                                                    break
                break
        return pos, euler


if __name__ == '__main__':
    invalid_taxel = np.array(
        ['touch_0_4_9', 'touch_0_4_8', 'touch_0_4_7', 'touch_0_5_6', 'touch_2_1_1', 'touch_9_1_1', 'touch_13_1_1',
         'touch_16_1_1'])
    txt_dir = "/data/largeshaker/One_fin/1/"
    # txt_dir = "/data/largeshaker/One_fin/2/"
    # txt_dir = "/data/largeshaker/Two_fin/1/"
    # txt_dir = "/data/largeshaker/Two_fin/2/"
    # txt_dir = "/data/largeshaker/Four_fin/1/"
    # txt_dir = "/data/largeshaker/Four_fin/2/"
    # txt_dir = "/data/largeshaker/Four_fin/3/"
    # txt_dir = "/data/bottle/One_fin/1/"
    # txt_dir = "/data/bottle/One_fin/2/"
    # txt_dir = "/data/bottle/One_fin/3/"
    # txt_dir = "/data/bottle/Two_fin/1/"
    # txt_dir = "/data/bottle/Two_fin/2/"
    # txt_dir = "/data/bottle/Two_fin/3/"
    # txt_dir = "/data/bottle/Four_fin/1/"
    # txt_dir = "/data/bottle/Four_fin/2/"
    # txt_dir = "/data/bottle/Four_fin/3/"

    ekf_data = ekfDataClass()
    my_class = MyClass(txt_dir=txt_dir, dataClass=ekf_data, pass_taxel=invalid_taxel)
    my_class.main_process()
