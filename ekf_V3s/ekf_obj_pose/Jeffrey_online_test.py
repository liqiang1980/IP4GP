#!/usr/bin/env python
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from allegro_tactile_sensor.msg import tactile_msgs
from geometry_msgs.msg import TransformStamped
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import tactile_allegro_mujo_const as tac_const
from mujoco_py import load_model_from_path
from scipy.spatial.transform import Rotation
import rospy
import numpy as np
import message_filters



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


        # self.data_all = np.loadtxt(self.data_path + 'all_data.txt')[60:]
        
        '''To be Done! Please update the hand_pose from the Vicon, although it is not be moved.'''
        self.data_path = tac_const.txt_dir[2]
        self.hand_pose = np.loadtxt(self.data_path + 'hand_pose.txt')
        # self.hand_pose = np.empty(7)
        # # 7*1
        # self.hand_pose[0] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.translation.x
        # self.hand_pose[1] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.translation.y
        # self.hand_pose[2] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.translation.z
        # self.hand_pose[3] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.rotation.x
        # self.hand_pose[4] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.rotation.y
        # self.hand_pose[5] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.rotation.z
        # self.hand_pose[6] = rospy.wait_for_message("/vicon/Allegro_hand/Allegro_hand", TransformStamped).transform.rotation.w



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

        '''parameters for ROS'''

        # rospy.init_node('DTI', anonymous=True)
        self.rate = rospy.Rate(50)
        self.sub1 = message_filters.Subscriber('/allegroHand_0/joint_states', JointState, queue_size=1)
        self.sub2 = message_filters.Subscriber('/allegro_tactile', tactile_msgs, queue_size=1)
        # sub3 = message_filters.Subscriber('/vicon/jeffrey_wan/jeffrey_wan', TransformStamped, queue_size=1)
        # For testing.
        # self.sub3 = message_filters.Subscriber('/vicon/fjl/fjl', TransformStamped, queue_size=1)
        # ts = message_filters.ApproximateTimeSynchronizer([self.sub1, self.sub2, self.sub3], 100, 0.1, allow_headerless=True)
        # ts.registerCallback(self.data_callback)
        # rospy.spin() 

        '''parameters for ROS Control'''
        self.pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size = 1)
        self.hand_ctrl = JointState()
        self.hand_ctrl.header = Header()
        self.hand_ctrl.name = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
                                    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
                                    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
                                    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']
        self.hand_ctrl.header.stamp = rospy.Time.now()
        self.hand_ctrl.position = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0])
        self.hand_ctrl.velocity = []
        for i in range(8):
            print('Control Init.')
            self.pub.publish(self.hand_ctrl)
            self.rate.sleep()    

    def listener(self):
        # sub1 = message_filters.Subscriber('/allegroHand_0/joint_states', JointState, queue_size=1)
        # sub2 = message_filters.Subscriber('/allegro_tactile', tactile_msgs, queue_size=1)
        # sub3 = message_filters.Subscriber('/vicon/jeffrey_wan/jeffrey_wan', TransformStamped, queue_size=1)
        # For testing.
        # sub3 = message_filters.Subscriber('/vicon/fjl/fjl', TransformStamped, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([self.sub1, self.sub2, self.sub3], 100, 0.1, allow_headerless=True)
        ts.registerCallback(self.data_callback)
        rospy.spin()
    
    def talker(self):
        theta_2 = 90 * np.pi / 180  
        for i in range (100):               
            self.hand_ctrl.position = [0, theta_2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.hand_ctrl.velocity = []
            self.pub.publish(self.hand_ctrl)  
            self.rate.sleep()

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
    rospy.init_node('DTI', anonymous=True)
    xml_path = "/home/manipulation-vnc/Code/IP4GP/robots/UR5_tactile_allegro_hand_obj_frozen.xml"
    assemble_data = BasicDataClass(xml_path=xml_path)
    # t1 = OnefinAction()

    try:
        # assemble_data.talker()
        # assemble_data.listener()
        assemble_data.talker()

    except rospy.ROSInternalException:
        pass