import numpy as np
from scipy.spatial.transform import Rotation
import util_geometry as ug


class TransComponent:
    def __init__(self):
        self.pos_cup_palm = [0]*3
        self.rotvec_cup_palm = [0]*3
        self.quat_cup_palm = [0]*3
        self.R_cup_palm = np.mat(np.eye(3))
        self.T_cup_palm = np.mat(np.eye(4))

    def components_update(self, x_state):
        self.pos_cup_palm = x_state[:3]
        self.rotvec_cup_palm = x_state[3:]
        self.quat_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_quat()
        self.R_cup_palm = Rotation.from_rotvec(self.rotvec_cup_palm).as_matrix()
        self.T_cup_palm[:3, :3] = self.R_cup_palm
        self.T_cup_palm[:3, 3] = np.mat(self.pos_cup_palm).T
