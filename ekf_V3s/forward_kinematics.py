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
        self.hand_description = URDF.from_xml_file('/home/manipulation-vnc/Code/IP4GP/robots/allegro_hand_right_with_tactile.urdf')
        # self.hand_description = URDF.from_xml_file('/home/lqg/robotics_data/qg_ws/src/fpt_vis/robot/allegro_hand_right_with_tactile.urdf')
        self.hand_tree = kdl_tree_from_urdf_model(self.hand_description)  # for kinematic chain
        self.qpos_pos = {"ff": [kdl.JntArray(4), kdl.Frame()], "ffd": [kdl.JntArray(3), kdl.Frame()],
                         "ffq": [kdl.JntArray(2), kdl.Frame()],
                         "mf": [kdl.JntArray(4), kdl.Frame()], "mfd": [kdl.JntArray(3), kdl.Frame()],
                         "mfq": [kdl.JntArray(2), kdl.Frame()],
                         "rf": [kdl.JntArray(4), kdl.Frame()], "rfd": [kdl.JntArray(3), kdl.Frame()],
                         "rfq": [kdl.JntArray(2), kdl.Frame()],
                         "th": [kdl.JntArray(4), kdl.Frame()], "thd": [kdl.JntArray(3), kdl.Frame()],
                         "palm": [kdl.JntArray(0), kdl.Frame()]}
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

    def get_cur_jnt(self, basicData):
        jnt_state = basicData.joint_states.position
        for i, jnt in enumerate(jnt_state):
            self.cur_jnt[i] = jnt
        return self.cur_jnt

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

    def get_relative_posrot(self, tac_name, f_name, xml_path):
        """
        Get pos & rpy of tac-in-tip from xml.
        Translate to tac-in-palm.
        """
        pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name, xml_path=xml_path)
        # print(tac_name, pos_tac_tip, rpy_tac_tip)
        T_tac_tip = qg.posrpy2trans(pos=pos_tac_tip, rpy=rpy_tac_tip)
        T_tip_palm = self.T_tip_palm[f_name]
        T_tac_palm = np.matmul(T_tip_palm, T_tac_tip)
        pos_tac_palm = np.ravel(T_tac_palm[:3, 3].T)
        rotvec_tac_palm = self.rotvec_tip_palm[f_name]  # Use default-jnt rotvec instead of tac rotvec
        return pos_tac_palm, rotvec_tac_palm, T_tac_palm

    def fk_update_all(self, basicData):
        self.get_cur_jnt(basicData=basicData)
        self.joint_update()
        self.fk_dealer()
