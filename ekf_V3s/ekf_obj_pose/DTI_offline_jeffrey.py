import config_param
import robot_control
import ekf
import tactile_perception
import numpy as np
import forward_kinematics
from mujoco_py import load_model_from_path
from scipy.spatial.transform import Rotation
import tactile_allegro_mujo_const as tac_const
import forward_kinematics
import qgFunc as qg

class BasicDataClass():
    def __init__(self, xml_path):
        self.data_path = tac_const.txt_dir[2]
        self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[60:]
        self.hand_pose = np.loadtxt(self.data_path + 'hand_pose.txt') 
        self.no_working_tac = {22: 'touch_0_5_9', 62: 'touch_0_3_2', 201: 'touch_7_4_3', 516: 'touch_16_3_1',
                               518: 'touch_16_3_3', 504: 'touch_16_1_1', 108: 'touch_2_1_1', 252: 'touch_9_1_1',
                               396: 'touch_13_1_1',}
        self.model = load_model_from_path(xml_path)

        '''Trans: hand to world'''
        self.hand_world_posquat = self.hand_pose
        self.hand_world_posquat[:3] += np.array([0.01, 0.01, 0])
        self.hand_world_R = Rotation.from_quat(self.hand_world_posquat[3:]).as_matrix()
        self.hand_world_T = np.mat(np.eye(4))
        self.hand_world_T[:3, :3] = self.hand_world_R
        self.hand_world_T[:3, 3] = np.mat(self.hand_world_posquat[:3]).T
        self.world_hand_T = np.linalg.pinv(self.hand_world_T)

        '''parameters for saving one data in temporarily'''
        self.time_stamp = 0
        self.obj_palm_posrotvec = [0.0] * 6
        self.joint_pos = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        self.joint_vel = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        self.taxel_data = [0.0] * tac_const.TAC_TOTAL_NUM
        self.tac_tip_pos = {}


if __name__ == "__main__":
    '''Loading Parameters'''
    hand_param, object_param, alg_param = config_param.pass_arg()

    '''Loading XML file'''
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand.xml"
    if int(object_param[3]) == 1:
        xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_obj_frozen.xml"
    elif int(object_param[3]) == 2:
        xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_obj_upsidedown.xml"
    elif int(object_param[3]) == 3:
        xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_cylinder.xml"
    elif int(object_param[3]) == 4:
        xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_cylinder_frozen.xml"
    
    '''Instance basic data class '''
    basic_data = BasicDataClass(xml_path=xml_path)

    '''Instance FK Class'''
    fk = forward_kinematics.ForwardKinematics(hand_param=hand_param)

    '''Instance EKF Class'''
    grasping_ekf = ekf.EKF()
    grasping_ekf.set_contact_flag(False)
    grasping_ekf.set_store_flag(alg_param[0])

    '''Instance tac-perception Class'''
    tac_perception = tactile_perception.cls_tactile_perception(xml_path=xml_path, fk=fk)

    '''Instance Robot Class'''
    rob_ctrl = robot_control.ROBCTRL(obj_param=object_param, hand_param=hand_param, model=..., xml_path=xml_path, fk=fk)

    '''Tac_in_tip Initialization'''
    f_param = hand_param[1:]
    print("Initialing...")
    for f_part in f_param:
        f_name = f_part[0]
        tac_id = f_part[3]
        basic_data.tac_tip_pos[f_name] = []
        for tid in range(tac_id[0], tac_id[1], 1):
            tac_name = basic_data.model._sensor_id2name[tid]
            pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name, xml_path=xml_path)
            basic_data.tac_tip_pos[f_name].append(pos_tac_tip)
    
    print("All Tac Parts are Ready!")

    first_contact_flag = False
    for i, _data in enumerate(basic_data.data_all):
        print('Round:', i)
        basic_data.time_stamp = _data[0]
        
        '''posquat of the object in world (xyzw)'''
        obj_world_posquat = _data[1:8]
        obj_world_R = Rotation.from_quat(obj_world_posquat[3:]).as_matrix()
        '''#左乘？還是右乘？'''
        obj_palm_R = np.matmul(basic_data.world_hand_T[:3, :3], obj_world_R)
        obj_palm_rotvec = Rotation.from_matrix(obj_palm_R).as_rotvec()
        obj_palm_pos = np.ravel(basic_data.world_hand_T[:3, 3].T) + np.ravel(np.matmul(basic_data.world_hand_T[:3, :3], obj_world_posquat[:3]))
        basic_data.obj_palm_posrotvec[:3] = obj_palm_pos
        basic_data.obj_palm_posrotvec[3:] = obj_palm_rotvec

        '''Joint Position'''
        basic_data.joint_pos = _data[8:24]
        
        '''Joint Velocity'''
        basic_data.joint_vel = _data[24:40]

        '''Tactile Data'''
        _taxel_data = _data[40: 40 + 540]
        basic_data.taxel_data = (_taxel_data.astype('int32')).tolist()

        for key in basic_data.no_working_tac:
            basic_data.taxel_data[key] = 0
        
        '''EKF Processing'''
        if not first_contact_flag and (np.array(basic_data.taxel_data) > 0.0).any():
            first_contact_flag = True
        
        '''EKF Start'''
        if first_contact_flag:
            print(rob_ctrl.cnt_test, "EKF Round:")
            rob_ctrl.interaction(object_param=object_param,
                                ekf_grasping=grasping_ekf,
                                tacp=tac_perception,
                                basicData=basic_data)

