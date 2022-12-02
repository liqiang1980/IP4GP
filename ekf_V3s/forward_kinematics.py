import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import numpy as np
import copy
import tactile_allegro_mujo_const as tac_const
import qgFunc as qg
from scipy.spatial.transform import Rotation


class ForwardKinematics:
    def __init__(self, hand_param):
        self.f_param = hand_param[1:]
        self.hand_description = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)  # for kinematic chain
        self.qpos_pos = {"ff": [kdl.JntArray(4), kdl.Frame()], "ffd": [kdl.JntArray(3), kdl.Frame()], "ffq": [kdl.JntArray(2), kdl.Frame()],
                         "mf": [kdl.JntArray(4), kdl.Frame()], "mfd": [kdl.JntArray(3), kdl.Frame()], "mfq": [kdl.JntArray(2), kdl.Frame()],
                         "rf": [kdl.JntArray(4), kdl.Frame()], "rfd": [kdl.JntArray(3), kdl.Frame()], "rfq": [kdl.JntArray(2), kdl.Frame()],
                         "th": [kdl.JntArray(4), kdl.Frame()], "thd": [kdl.JntArray(3), kdl.Frame()], "palm": [kdl.JntArray(0), kdl.Frame()]}
        self.jnt_num = tac_const.FULL_FINGER_JNTS_NUM
        self.cur_jnt = np.zeros(self.jnt_num)
        self.kdl_chain = {}
        self.kdl_fk = {}
        self.T_tip_palm = {}
        self.R_tip_palm = {}
        self.rotvec_tip_palm = {}
        """ Initialization """
        for f_part in self.f_param:
            f_name = f_part[0]
            kdl.SetToZero(self.qpos_pos[f_name][0])
            self.kdl_chain[f_name] = self.hand_tree.getChain(chain_root="palm_link", chain_tip=f_part[5])
            self.kdl_fk[f_name] = kdl.ChainFkSolverPos_recursive(self.kdl_chain[f_name])
            self.T_tip_palm[f_name] = np.mat(np.eye(4))
            self.R_tip_palm[f_name] = np.mat(np.eye(3))
            self.rotvec_tip_palm[f_name] = np.zeros(3)

        # self.index_qpos = kdl.JntArray(4)
        # self.mid_qpos = kdl.JntArray(4)
        # self.ring_qpos = kdl.JntArray(4)
        # self.thumb_qpos = kdl.JntArray(4)
        # kdl.SetToZero(self.index_qpos)
        # kdl.SetToZero(self.mid_qpos)
        # kdl.SetToZero(self.ring_qpos)
        # kdl.SetToZero(self.thumb_qpos)
        # self.index_chain = self.hand_tree.getChain("palm_link", "link_3.0_tip")
        # self.mid_chain = self.hand_tree.getChain("palm_link", "link_7.0_tip")
        # self.ring_chain = self.hand_tree.getChain("palm_link", "link_11.0_tip")
        # self.thumb_chain = self.hand_tree.getChain("palm_link", "link_15.0_tip")
        # forward kinematics: allegro_jstates
        # self.index_fk = kdl.ChainFkSolverPos_recursive(self.index_chain)
        # self.mid_fk = kdl.ChainFkSolverPos_recursive(self.mid_chain)
        # self.ring_fk = kdl.ChainFkSolverPos_recursive(self.ring_chain)
        # self.thumb_fk = kdl.ChainFkSolverPos_recursive(self.thumb_chain)
        # init kdl_Frames for each finger
        # self.index_pos = kdl.Frame()  # Construct an identity frame
        # self.mid_pos = kdl.Frame()  # Construct an identity frame
        # self.ring_pos = kdl.Frame()  # Construct an identity frame
        # self.thumb_pos = kdl.Frame()  # Construct an identity frame
        # T results from FK_dealer
        # self.T_index_palm = np.mat(np.eye(4))
        # self.T_middle_palm = np.mat(np.eye(4))
        # self.T_ring_palm = np.mat(np.eye(4))
        # self.T_thumb_palm = np.mat(np.eye(4))

    def get_cur_jnt(self, sim):
        # jnt_num = tac_const.FULL_FINGER_JNTS_NUM
        # cur_jnt = np.zeros(jnt_num)
        jnt_id = [tac_const.FF_MEA_1, tac_const.FF_MEA_2, tac_const.FF_MEA_3, tac_const.FF_MEA_4,
                  tac_const.MF_MEA_1, tac_const.MF_MEA_2, tac_const.MF_MEA_3, tac_const.MF_MEA_4,
                  tac_const.RF_MEA_1, tac_const.RF_MEA_2, tac_const.RF_MEA_3, tac_const.RF_MEA_4,
                  tac_const.TH_MEA_1, tac_const.TH_MEA_2, tac_const.TH_MEA_3, tac_const.TH_MEA_4
                  ]
        for i in range(self.jnt_num):
            self.cur_jnt[i] = sim.data.qpos[jnt_id[i]]
        return self.cur_jnt
        # cur_jnt[0:4] = np.array([sim.data.qpos[],
        #                          sim.data.qpos[tac_const.FF_MEA_2],
        #                          sim.data.qpos[tac_const.FF_MEA_3],
        #                          sim.data.qpos[tac_const.FF_MEA_4]])
        #
        # cur_jnt[4:8] = np.array([sim.data.qpos[tac_const.MF_MEA_1],
        #                          sim.data.qpos[tac_const.MF_MEA_2],
        #                          sim.data.qpos[tac_const.MF_MEA_3],
        #                          sim.data.qpos[tac_const.MF_MEA_4]])
        #
        # cur_jnt[8:12] = np.array([sim.data.qpos[tac_const.RF_MEA_1],
        #                           sim.data.qpos[tac_const.RF_MEA_2],
        #                           sim.data.qpos[tac_const.RF_MEA_3],
        #                           sim.data.qpos[tac_const.RF_MEA_4]])

        # cur_jnt[12:16] = np.array([sim.data.qpos[tac_const.TH_MEA_1],
        #                            sim.data.qpos[tac_const.TH_MEA_2],
        #                            sim.data.qpos[tac_const.TH_MEA_3],
        #                            sim.data.qpos[tac_const.TH_MEA_4]])
        # return cur_jnt

    def joint_update(self):
        _all_joint = self.cur_jnt
        self.qpos_pos["ff"][0][0] = _all_joint[0]
        self.qpos_pos["ff"][0][1] = _all_joint[1]
        self.qpos_pos["ff"][0][2] = _all_joint[2]
        self.qpos_pos["ff"][0][3] = _all_joint[3]

        self.qpos_pos["mf"][0][0] = _all_joint[4]
        self.qpos_pos["mf"][0][1] = _all_joint[5]
        self.qpos_pos["mf"][0][2] = _all_joint[6]
        self.qpos_pos["mf"][0][3] = _all_joint[7]

        self.qpos_pos["rf"][0][0] = _all_joint[8]
        self.qpos_pos["rf"][0][1] = _all_joint[9]
        self.qpos_pos["rf"][0][2] = _all_joint[10]
        self.qpos_pos["rf"][0][3] = _all_joint[11]

        self.qpos_pos["th"][0][0] = _all_joint[12]
        self.qpos_pos["th"][0][1] = _all_joint[13]
        self.qpos_pos["th"][0][2] = _all_joint[14]
        self.qpos_pos["th"][0][3] = _all_joint[15]

        self.qpos_pos["ffd"][0][0] = _all_joint[0]
        self.qpos_pos["ffd"][0][1] = _all_joint[1]
        self.qpos_pos["ffd"][0][2] = _all_joint[2]

        self.qpos_pos["mfd"][0][0] = _all_joint[4]
        self.qpos_pos["mfd"][0][1] = _all_joint[5]
        self.qpos_pos["mfd"][0][2] = _all_joint[6]

        self.qpos_pos["rfd"][0][0] = _all_joint[8]
        self.qpos_pos["rfd"][0][1] = _all_joint[9]
        self.qpos_pos["rfd"][0][2] = _all_joint[10]

        self.qpos_pos["thd"][0][0] = _all_joint[12]
        self.qpos_pos["thd"][0][1] = _all_joint[13]
        self.qpos_pos["thd"][0][2] = _all_joint[14]

        self.qpos_pos["ffq"][0][0] = _all_joint[0]
        self.qpos_pos["ffq"][0][1] = _all_joint[1]

        self.qpos_pos["mfq"][0][0] = _all_joint[4]
        self.qpos_pos["mfq"][0][1] = _all_joint[5]

        self.qpos_pos["rfq"][0][0] = _all_joint[8]
        self.qpos_pos["rfq"][0][1] = _all_joint[9]


    def fk_dealer(self):
        """
        Get T (tips in palm) and J by FK method
        joint positions are updated in main_process()
        """
        M = np.mat(np.zeros((3, 3)))
        p = np.zeros([3, 1])
        # index_in_palm_T = np.mat(np.eye(4))
        # mid_in_palm_T = np.mat(np.eye(4))
        # ring_in_palm_T = np.mat(np.eye(4))
        # thumb_in_palm_T = np.mat(np.eye(4))

        for f_part in self.f_param:
            f_name = f_part[0]
            if f_name == "palm":
                break
            # print(f_name)
            qg.kdl_calc_fk(fk=self.kdl_fk[f_name], q=self.qpos_pos[f_name][0], pos=self.qpos_pos[f_name][1])
            for i in range(3):
                p[i, 0] = copy.deepcopy(self.qpos_pos[f_name][1].p[i])
                for j in range(3):
                    M[i, j] = copy.deepcopy(self.qpos_pos[f_name][1].M[i, j])
            self.T_tip_palm[f_name][:3, :3] = M
            self.T_tip_palm[f_name][:3, 3] = p
            self.R_tip_palm[f_name] = M
            self.rotvec_tip_palm[f_name] = Rotation.from_matrix(self.R_tip_palm[f_name]).as_rotvec()
        return self.T_tip_palm

        # forward kinematics
        # qg.kdl_calc_fk(self.index_fk, self.index_qpos, self.index_pos)
        # M[0, 0] = copy.deepcopy(self.index_pos.M[0, 0])
        # M[0, 1] = copy.deepcopy(self.index_pos.M[0, 1])
        # M[0, 2] = copy.deepcopy(self.index_pos.M[0, 2])
        # M[1, 0] = copy.deepcopy(self.index_pos.M[1, 0])
        # M[1, 1] = copy.deepcopy(self.index_pos.M[1, 1])
        # M[1, 2] = copy.deepcopy(self.index_pos.M[1, 2])
        # M[2, 0] = copy.deepcopy(self.index_pos.M[2, 0])
        # M[2, 1] = copy.deepcopy(self.index_pos.M[2, 1])
        # M[2, 2] = copy.deepcopy(self.index_pos.M[2, 2])
        # p[0, 0] = copy.deepcopy(self.index_pos.p[0])
        # p[1, 0] = copy.deepcopy(self.index_pos.p[1])
        # p[2, 0] = copy.deepcopy(self.index_pos.p[2])
        # index_in_palm_T[:3, :3] = M
        # index_in_palm_T[:3, 3] = p
        #
        # qg.kdl_calc_fk(self.mid_fk, self.mid_qpos, self.mid_pos)
        # M[0, 0] = copy.deepcopy(self.mid_pos.M[0, 0])
        # M[0, 1] = copy.deepcopy(self.mid_pos.M[0, 1])
        # M[0, 2] = copy.deepcopy(self.mid_pos.M[0, 2])
        # M[1, 0] = copy.deepcopy(self.mid_pos.M[1, 0])
        # M[1, 1] = copy.deepcopy(self.mid_pos.M[1, 1])
        # M[1, 2] = copy.deepcopy(self.mid_pos.M[1, 2])
        # M[2, 0] = copy.deepcopy(self.mid_pos.M[2, 0])
        # M[2, 1] = copy.deepcopy(self.mid_pos.M[2, 1])
        # M[2, 2] = copy.deepcopy(self.mid_pos.M[2, 2])
        # p[0, 0] = copy.deepcopy(self.mid_pos.p[0])
        # p[1, 0] = copy.deepcopy(self.mid_pos.p[1])
        # p[2, 0] = copy.deepcopy(self.mid_pos.p[2])
        # mid_in_palm_T[:3, :3] = M
        # mid_in_palm_T[:3, 3] = p
        #
        # qg.kdl_calc_fk(self.ring_fk, self.ring_qpos, self.ring_pos)
        # M[0, 0] = copy.deepcopy(self.ring_pos.M[0, 0])
        # M[0, 1] = copy.deepcopy(self.ring_pos.M[0, 1])
        # M[0, 2] = copy.deepcopy(self.ring_pos.M[0, 2])
        # M[1, 0] = copy.deepcopy(self.ring_pos.M[1, 0])
        # M[1, 1] = copy.deepcopy(self.ring_pos.M[1, 1])
        # M[1, 2] = copy.deepcopy(self.ring_pos.M[1, 2])
        # M[2, 0] = copy.deepcopy(self.ring_pos.M[2, 0])
        # M[2, 1] = copy.deepcopy(self.ring_pos.M[2, 1])
        # M[2, 2] = copy.deepcopy(self.ring_pos.M[2, 2])
        # p[0, 0] = copy.deepcopy(self.ring_pos.p[0])
        # p[1, 0] = copy.deepcopy(self.ring_pos.p[1])
        # p[2, 0] = copy.deepcopy(self.ring_pos.p[2])
        # ring_in_palm_T[:3, :3] = M
        # ring_in_palm_T[:3, 3] = p
        #
        # qg.kdl_calc_fk(self.thumb_fk, self.thumb_qpos, self.thumb_pos)
        # M[0, 0] = self.thumb_pos.M[0, 0]
        # M[0, 1] = self.thumb_pos.M[0, 1]
        # M[0, 2] = self.thumb_pos.M[0, 2]
        # M[1, 0] = self.thumb_pos.M[1, 0]
        # M[1, 1] = self.thumb_pos.M[1, 1]
        # M[1, 2] = self.thumb_pos.M[1, 2]
        # M[2, 0] = self.thumb_pos.M[2, 0]
        # M[2, 1] = self.thumb_pos.M[2, 1]
        # M[2, 2] = self.thumb_pos.M[2, 2]
        # p[0, 0] = self.thumb_pos.p[0]
        # p[1, 0] = self.thumb_pos.p[1]
        # p[2, 0] = self.thumb_pos.p[2]
        # thumb_in_palm_T[:3, :3] = M
        # thumb_in_palm_T[:3, 3] = p

        # return index_in_palm_T, mid_in_palm_T, ring_in_palm_T, thumb_in_palm_T

    def fk_update_all(self, sim):
        self.get_cur_jnt(sim=sim)
        self.joint_update()
        self.fk_dealer()
        # print("--fk update:", self.T_tip_palm)

