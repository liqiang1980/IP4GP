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

'''To be fixed:
The fingers do not move.'''


class AllegroHandController():
    def __init__(self, xml_path):
        # Initialize ROS node
        rospy.init_node('allegro_hand_controller')

        # Set up publishers for joint commands
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)

        # Set up subscribers for joint states and tactile data
        self.joint_angles_sub = rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.tactile_sub = rospy.Subscriber('/allegro_tactile', tactile_msgs, self.tactile_callback, queue_size= 1)
        self.vicon_sub = rospy.Subscriber('/vicon/jeffrey_cup/jeffrey_cup', TransformStamped, self.vicon_callback,queue_size=1)
        # Initialize joint states and tactile data

        '''Allegrohand Data Init'''
        self.joint_states = JointState()
        self.joint_states.position = np.zeros(16)
        # self.joint_states.velocity = np.zeros(16)

        '''Tactile Data Init'''
        self.tactile_data = np.zeros(540)


        '''Object Pose from Vicon Data init'''
        self.object_pose = np.zeros(7)


        # Define the index of each joint in 'joint_states'
        self.ctrl_id = {
            'ff': [0, 1, 2, 3],
            'mf': [4, 5, 6, 7],
            'rf': [8, 9, 10, 11],
            'th': [12, 13, 14, 15]
        }

        '''Hand Pose Collected from Vicon.'''
        self.hand_pose = np.array([-2.52357038e-0, 6.44453981e-01, 5.03844975e+00, 7.26140712e-03, -1.84615072e-03, 2.17123351e-04, 9.99971908e-01])

        '''Model from XML'''
        self.model = load_model_from_path(xml_path)

        '''No Working Tactile from QG's file.'''
        self.no_working_tac = {22: 'touch_0_5_9', 62: 'touch_0_3_2', 201: 'touch_7_4_3', 516: 'touch_16_3_1',
                               518: 'touch_16_3_3', 504: 'touch_16_1_1', 108: 'touch_2_1_1', 252: 'touch_9_1_1',
                               396: 'touch_13_1_1',}
        
        
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
        # self.taxel_data = [0.0] * tac_const.TAC_TOTAL_NUM
        self.tac_tip_pos = {}

    def get_tactile_id(self, f_name):
        if f_name == 'ff':
            return (0, 114)
        elif f_name == 'mf':
            return (114, 288)
        elif f_name == 'rf':
            return (288, 432)
        elif f_name == 'th':
            return (432, 540)

    def joint_states_callback(self, joint_msg):
        self.joint_states.position = joint_msg.position
        # print('Callback Joint States. The joint states are: \n', self.joint_states.position)
    
    def tactile_callback(self, tac_msg):
        '''combine the tactile data to taxel_data'''
        self.tactile_data[:72] = tac_msg.index_tip_Value  # 72
        self.tactile_data[72:108] = tac_msg.index_mid_Value  # 36
        self.tactile_data[108:144] = tac_msg.index_end_Value  # 36
        self.tactile_data[144:216] = tac_msg.middle_tip_Value  # 72
        self.tactile_data[216:252] = tac_msg.middle_mid_Value  # 36
        self.tactile_data[252:288] = tac_msg.middle_mid_Value  # 36
        self.tactile_data[288:360] = tac_msg.ring_tip_Value  # 72
        self.tactile_data[360:396] = tac_msg.ring_mid_Value  # 36
        self.tactile_data[396:432] = tac_msg.ring_end_Value  # 36
        self.tactile_data[432:504] = tac_msg.thumb_tip_Value  # 72
        self.tactile_data[504:540] = tac_msg.thumb_mid_Value  # 36
        # self.tactile_data[540:653] = tac_msg.palm_Value  # 113
        # print('Callback Tactile Data. The tactile data is: \n', self.tactile_data)

    def vicon_callback(self, vicon_msg):
        self.object_pose[0] = vicon_msg.transform.translation.x
        self.object_pose[1] = vicon_msg.transform.translation.y
        self.object_pose[2] = vicon_msg.transform.translation.z
        self.object_pose[3] = vicon_msg.transform.rotation.x
        self.object_pose[4] = vicon_msg.transform.rotation.y
        self.object_pose[5] = vicon_msg.transform.rotation.z
        self.object_pose[6] = vicon_msg.transform.rotation.w
        # print('Callback Vicon Data. The Object Pose is:\n', self.object_pose)


    def finger_control_func(self, input_1, input_2, f_part):
        rate = rospy.Rate(50) # Hz

        '''Construct joint command message'''
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.name = ['joint_{}'.format(i) for i in range(16)]
        joint_cmd.position = [0.0] * 16

        f_name = f_part[0]
        tac_id = f_part[3]
        ctrl_id = {"ff": [1, 2, 3],
                    "mf": [5, 6, 7],
                    "rf": [9, 10, 11],
                    "th": [14, 15]}

        _input = 0

        # Check if any tactile sensor is being pressed
        if not(np.array(self.tactile_data[tac_id[0]: tac_id[1]]) > 0.0).any():
            _input = input_1
        else:
            _input = input_2

        for cid in ctrl_id[f_name]:
            joint_cmd.position[cid] += _input
        

        '''Publish joint command message'''
        self.joint_cmd_pub.publish(joint_cmd)

        # Sleep to maintain loop rate
        rate.sleep()



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
    
    '''Instance AllegroHand Finger Control Class'''
    basic_data = AllegroHandController(xml_path=xml_path)

    '''Instance FK Class'''
    fk = forward_kinematics.ForwardKinematics(hand_param=hand_param)

    '''Instance EKF Class'''
    grasping_ekf = ekf.EKF()
    grasping_ekf.set_contact_flag(False)
    grasping_ekf.set_store_flag(alg_param[0])

    '''Instance tac-perception Class'''
    tac_perception = tactile_perception.Cls_tactile_perception(xml_path=xml_path, fk=fk)

    '''Instance Robot Class'''
    rob_ctrl = robot_control.Robctrl(obj_param=object_param, hand_param=hand_param, model=basic_data.model, xml_path=xml_path, fk=fk)

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
    ctrl_val = ctrl_order[int(object_param[3])]
    # tran_cnt = [0, 0, 0, 0, 0, 150, 0, 0, 100]
    # tran_cnt0 = [0, 0, 0, 0, 0, 350, 0, 0, 300]
    round_choose = [610, 1200, 0, 0, 0, 1200, 0, 0, 1200]
    

    '''===============================Itertivate Process============================='''
    '''=============================================================================='''

    for i in range(round_choose[int(object_param[3])]):
        print('Round:', i)
        
        '''Fingers Control'''
        for f_part in f_param:
            f_name = f_part[0]
            tac_id = f_part[3]
            if f_name in ctrl_val:
                basic_data.finger_control_func(input_1=ctrl_val[f_name][0], input_2=ctrl_val[f_name][1], f_part=f_part)
                        
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
        print('Current joint position', basic_data.joint_states.position)

        '''Tactile Data'''
        _taxel_data = basic_data.tactile_data
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

