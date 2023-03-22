#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from allegro_tactile_sensor.msg import tactile_msgs
from geometry_msgs.msg import TransformStamped

'''
To be test!
'''

class AllegroHandController():
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('allegro_hand_controller')

        # Set up publishers for joint commands
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)

        # Set up subscribers for joint states and tactile data
        self.joint_angles_sub = rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.tactile_sub = rospy.Subscriber('/allegro_tactile', tactile_msgs, self.tactile_callback, queue_size= 1)
        self.vicon_sub = rospy.Subscriber('/vicon/jeffrey_cup/jeffrey_cup', TransformStamped, self.vicon_callback,queue_size=1)
        # Initialize joint states and tactile data
        self.joint_states = JointState()
        self.joint_states.position = np.zeros(16)
        self.tactile_data = np.zeros(540)
        self.object_pose = np.zeros(7)

        # Set up control parameters
        self.input_1 = 0.1
        self.input_2 = 0.01

        # Define the index of each joint in 'joint_states'
        self.ctrl_id = {
            'ff': [0, 1, 2, 3],
            'mf': [4, 5, 6, 7],
            'rf': [8, 9, 10, 11],
            'th': [12, 13, 14, 15]
        }

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


    def finger_control_func(self):
        rate = rospy.Rate(50) # Hz
        # Reset all joints to 0
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.name = ['joint_{}'.format(i) for i in range(16)]
        joint_cmd.position = [0.0] * 16
        
        # Publish joint command message to reseat all joints to 0
        self.joint_cmd_pub.publish(joint_cmd)

        while not rospy.is_shutdown():


            # Construct joint command message
            joint_cmd = JointState()
            joint_cmd.header.stamp = rospy.Time.now()
            joint_cmd.name = ['joint_{}'.format(i) for i in range(16)]
            joint_cmd.position = [0.0] * 16

            # Control each finger separately
            for f_name in ['ff', 'mf', 'rf', 'th']:
                joint_cmds = [0.0, 0.0, 0.0, 0.0]
                tac_id = self.get_tactile_id(f_name)

                # Check if any tactile sensor is being pressed
                if not(np.array(self.tactile_data[tac_id[0]: tac_id[1]]) > 0.0).any():
                    input_val = self.input_1
                else:
                    input_val = self.input_2

                # # Apply control input to finger joints
                # joint_cmds = [0.0, input_val, input_val, input_val]
                # if f_name == 'th':
                #     joint_cmds = [0.0, 0.0, input_val, input_val]
                # for i, joint in enumerate(self.ctrl_id[f_name]):
                #     if joint in [0, 4, 8, 13]:
                #         joint_cmd.position[joint] = 0.0
                    
                #     elif joint == 12:
                #         joint_cmd.position[joint] = 1.3
                    
                #     else:
                #         joint_cmd.position[joint] = self.joint_states.position[joint] + joint_cmds[i]
                '''Apply Control input to finger joints (Power Grasp)'''
                if f_name == 'th':
                    joint_cmds = [1.3, input_val, input_val, input_val]
                    joint_cmds[3] += 0.3
                    
                else:
                    joint_cmds = [0.0, input_val, input_val, input_val]
                    joint_cmds[1] += 0.3
                    joint_cmds[2] += 0.2
                    joint_cmds[3] += 0.01

                for i, joint in enumerate(self.ctrl_id[f_name]):
                    if joint in [0, 4, 8, 13]:
                        joint_cmd.position[joint] = 0.0
                    
                    elif joint == 12:
                        joint_cmd.position[joint] = 1.3
                    
                    else:
                        joint_cmd.position[joint] = self.joint_states.position[joint] + joint_cmds[i]

            # Publish joint command message
            self.joint_cmd_pub.publish(joint_cmd)

            # Sleep to maintain loop rate
            rate.sleep()



if __name__ == '__main__':
    try:
        allegrohand = AllegroHandController()
        allegrohand.finger_control_func()
        print('joint msg:', allegrohand.joint_states.position)
        print('tactile msg:', allegrohand.tactile_data)
        print('object GT:', allegrohand.object_pose)
        
    except rospy.ROSInterruptException:
        pass


