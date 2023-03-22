#!/usr/bin/env python
from sensor_msgs.msg import JointState
from allegro_tactile_sensor.msg import tactile_msgs
from geometry_msgs.msg import TransformStamped
from mujoco_py import load_model_from_path
from scipy.spatial.transform import Rotation
from urdf_parser_py.urdf import URDF
import rospy
import message_filters
import config_param
import robot_control
import ekf
import tactile_perception
import numpy as np
import forward_kinematics
import tactile_allegro_mujo_const as tac_const
import forward_kinematics
import qgFunc as qg

class BasicDataClass():

    def __init__(self, xml_path):
        '''Allegrohand Data init'''
        self.joint_pos = np.zeros(16)
        self.joint_vel = np.zeros(16)
        
        '''Tactile Data Init'''
        self.index_tip_value = np.zeros([12, 6])
        self.index_mid_value = np.zeros([6, 6])
        self.index_end_value = np.zeros([6, 6])

        self.middle_tip_value = np.zeros([12, 6])
        self.middle_mid_value = np.zeros([6, 6])
        self.middle_end_value = np.zeros([6, 6])

        self.ring_tip_value = np.zeros([12, 6])
        self.ring_mid_value = np.zeros([6, 6])
        self.ring_end_value = np.zeros([6, 6])

        self.thumb_tip_value = np.zeros([12, 6])
        self.thumb_mid_value = np.zeros([6, 6])
        self.palm_value = np.zeros(113)

        '''Object Pose from Vicon Data init'''
        self.object_pose = np.zeros(7)
        # self.hand_pose = np.ones(7)


        '''Inherit from QG'''
        self.data_path = tac_const.txt_dir[2]
        # self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[60:]
        
        '''To be Done! Please update the hand_pose from the Vicon, although it is not be moved.'''
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
        # self.joint_pos = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        # self.joint_vel = [0.0] * tac_const.FULL_FINGER_JNTS_NUM
        # '''I use the specfic tactile part instead of taxel_data'''
        self.taxel_data = [0.0] * tac_const.TAC_TOTAL_NUM
        self.tac_tip_pos = {}

        rospy.init_node('DTI', anonymous=True)
        self.rate = rospy.Rate(50)
    

    def listener(self):
        sub1 = message_filters.Subscriber('/allegroHand_0/joint_states', JointState, queue_size=1)
        sub2 = message_filters.Subscriber('/allegro_tactile', tactile_msgs, queue_size=1)
        # sub3 = message_filters.Subscriber('/vicon/jeffrey_wan/jeffrey_wan', TransformStamped, queue_size=1)
        # For testing.
        sub3 = message_filters.Subscriber('/vicon/fjl/fjl', TransformStamped, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2, sub3], 100, 0.1, allow_headerless=True)
        ts.registerCallback(self.data_callback)
        rospy.spin()
    
    def data_callback(self, joint_msg, tac_msg, vicon_msg):
        self.joint_pos = joint_msg.position
        self.joint_vel = joint_msg.velocity
        print('joint position:\n', self.joint_pos)
        print('joint velocity:\n', self.joint_vel)
        '''To be checked again!'''
        self.index_tip_value = np.reshape(tac_msg.index_tip_Value, [12, 6])
        self.index_mid_value = np.reshape(tac_msg.index_mid_Value, [6, 6])
        self.index_end_value = np.reshape(tac_msg.index_end_Value, [6, 6])

        self.middle_tip_value = np.reshape(tac_msg.middle_tip_Value, [12, 6])
        self.middle_mid_value = np.reshape(tac_msg.middle_mid_Value, [6, 6])
        self.middle_end_value = np.reshape(tac_msg.middle_end_Value, [6, 6])

        self.ring_tip_value = np.reshape(tac_msg.ring_tip_Value, [12, 6])
        self.ring_mid_value = np.reshape(tac_msg.ring_mid_Value, [6, 6])
        self.ring_end_value = np.reshape(tac_msg.ring_end_Value, [6, 6])

        self.thumb_tip_value = np.reshape(tac_msg.thumb_tip_Value, [12, 6])
        self.thumb_mid_value = np.reshape(tac_msg.thumb_mid_Value, [6, 6])
        self.palm_value = np.array(tac_msg.palm_Value)

        print('index_tip_value:\n', self.index_tip_value)
        print('index_mid_value:\n', self.index_mid_value)
        print('index_end_value:\n', self.index_end_value)
        print('palm_value:\n', self.palm_value)
    
        self.object_pose[0] = vicon_msg.transform.translation.x
        self.object_pose[1] = vicon_msg.transform.translation.y
        self.object_pose[2] = vicon_msg.transform.translation.z
        self.object_pose[3] = vicon_msg.transform.rotation.x
        self.object_pose[4] = vicon_msg.transform.rotation.y
        self.object_pose[5] = vicon_msg.transform.rotation.z
        self.object_pose[6] = vicon_msg.transform.rotation.w
        print('Object Pose:\n', self.object_pose)

        '''combine the tactile data to taxel_data'''
        self.taxel_data[:72] = self.index_tip_value  # 72
        self.taxel_data[72:108] = self.index_mid_value  # 36
        self.taxel_data[108:144] = self.index_end_value  # 36
        self.taxel_data[144:216] = self.middle_tip_value  # 72
        self.taxel_data[216:252] = self.middle_mid_value  # 36
        self.taxel_data[252:288] = self.middle_end_value  # 36
        self.taxel_data[288:360] = self.ring_tip_value  # 72
        self.taxel_data[360:396] = self.ring_mid_value  # 36
        self.taxel_data[396:432] = self.ring_end_value  # 36
        self.taxel_data[432:504] = self.thumb_tip_value  # 72
        self.taxel_data[504:540] = self.thumb_mid_value  # 36
        self.taxel_data[540:653] = self.palm_value  # 113



def finger_control_func(joint_cmd, taxel_data, input_1, input_2, f_part):
    f_name = f_part[0]
    tac_id = f_part[3]
    ctrl_id = {"ff": [tac_const.FF_CTRL_2, tac_const.FF_CTRL_3, tac_const.FF_CTRL_4],
               "mf": [tac_const.MF_CTRL_2, tac_const.MF_CTRL_3, tac_const.MF_CTRL_4],
               "rf": [tac_const.RF_CTRL_2, tac_const.RF_CTRL_3, tac_const.RF_CTRL_4],
               "th": [tac_const.TH_CTRL_3, tac_const.TH_CTRL_4]
               }
    _input = 0
    if not(np.array(taxel_data[tac_id[0]: tac_id[1]]) > 0.0).any():
        _input = input_1
    else:
        _input = input_2
    
    for cid in ctrl_id[f_name]:
        joint_cmd[cid] += _input


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
    
    '''Instance online data Class'''
    basic_data = BasicDataClass(xml_path=xml_path)
    try:
        basic_data.listener()
    except rospy.ROSInternalException:
        pass


    '''Instance FK Class'''
    fk = forward_kinematics.ForwardKinematics(hand_param=hand_param)

    '''Instance EKF Class'''
    grasping_ekf = ekf.EKF()
    grasping_ekf.set_contact_flag(False)
    grasping_ekf.set_store_flag(alg_param[0])

    '''Instance tac-perception Class'''
    tac_perception = tactile_perception.cls_tactile_perception(xml_path=xml_path, fk=fk)

    '''Instance Robot Class'''
    rob_ctrl = robot_control.ROBCTRL(obj_param=object_param, hand_param=hand_param, model=basic_data.model, xml_path=xml_path, fk=fk)

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

    ctrl_order = tac_const.CTRL_ORDER
    ctrl_val = ctrl_order[object_param[3]]
    # tran_cnt = [0, 0, 0, 0, 0, 150, 0, 0, 100]
    # tran_cnt0 = [0, 0, 0, 0, 0, 350, 0, 0, 300]
    round_choose = [610, 1200, 0, 0, 0, 1200, 0, 0, 1200]

    for i in range(round_choose[object_param[3]]):
        print('Round:', i)
        
        '''Fingers Contral'''
        for f_part in f_param:
            f_name = f_part[0]
            tac_id = f_part[3]
            if f_name in ctrl_val:
                finger_control_func(joint_cmd= , taxel_data= basic_data.taxel_data, input_1=ctrl_val[f_name][0], input_2=ctrl_val[f_name][1], f_part=f_part)
                        
        '''posquat of the object in world (xyzw)'''
        obj_world_posquat = basic_data.object_pose
        obj_world_R = Rotation.from_quat(obj_world_posquat[3:]).as_matrix()
        '''#左乘？還是右乘？'''
        obj_palm_R = np.matmul(basic_data.world_hand_T[:3, :3], obj_world_R)
        obj_palm_rotvec = Rotation.from_matrix(obj_palm_R).as_rotvec()
        obj_palm_pos = np.ravel(basic_data.world_hand_T[:3, 3].T) + np.ravel(np.matmul(basic_data.world_hand_T[:3, :3], obj_world_posquat[:3]))
        basic_data.obj_palm_posrotvec[:3] = obj_palm_pos
        basic_data.obj_palm_posrotvec[3:] = obj_palm_rotvec

        '''Joint Position'''
        print('Current joint position', basic_data.joint_pos)
        
        '''Joint Velocity'''
        print('Current joint Velocity', basic_data.joint_vel)

        '''Tactile Data'''
        _taxel_data = basic_data.taxel_data[0: 540]
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

