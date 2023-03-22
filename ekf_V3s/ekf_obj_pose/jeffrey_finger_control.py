#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from allegro_tactile_sensor.msg import tactile_msgs
from geometry_msgs.msg import TransformStamped

from std_msgs.msg import Float64
import numpy as np

class AllegroHandController():
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('allegro_hand_controller')

        # Set up publishers for joint commands
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)

        # Set up subscribers for joint states and tactile data
        self.joint_angles_sub = rospy.Subscriber('/allegroHand_0/joint_states', JointState, self.joint_angles_callback)
        self.tactile_sub = rospy.Subscriber('/allegro_tactile', tactile_msgs, self.tactile_callback)

        # Initialize joint states and tactile data
        self.joint_states = JointState()
        self.joint_states.position = np.zeros(16)
        self.tactile_data = np.zeros(635)

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

        # Start the main loop
        self.run()

    def get_tactile_id(self, f_name):
        if f_name == 'ff':
            return (0, 72)
        elif f_name == 'mf':
            return (114, 216)
        elif f_name == 'rf':
            return (288, 360)
        elif f_name == 'th':
            return (432, 504)

    def run(self):
        rate = rospy.Rate(50) # Hz
        while not rospy.is_shutdown():
            # Construct joint command message
            joint_cmd = JointState()
            joint_cmd.header.stamp = rospy.Time.now()
            joint_cmd.name = ['joint_{}'.format(i) for i in range(16)]

            # Control each finger separately
            # for f_part in [('ff', 0, 72), ('mf', 114, 216), ('rf', 288, 360), ('th', 432, 504)]:
            #     f_name = f_part[0]
            #     tac_id = f_part[1:]
            for f_name in ['ff', 'mf', 'rf', 'th']:
                joint_cmds = [0.0, 0.0, 0.0, 0.0]
                tac_id = self.get_tactile_id(f_name)


                # Check if any tactile sensor is being pressed
                if not(np.array(self.tactile_data[tac_id[0]: tac_id[1]]) > 100.0).any():
                    input_val = self.input_1
                else:
                    input_val = self.input_2

                # Apply control input to finger joints
                joint_cmds = [0.0, input_val, input_val, input_val]
                if f_name == 'th':
                    joint_cmds = [0.5, input_val, input_val, input_val]
                for i, joint in enumerate(self.ctrl_id[f_name]):
                    joint_cmd.position[joint] = self.joint_states.position[joint] + joint_cmds[i]

            # Publish joint command message
            self.joint_cmd_pub.publish(joint_cmd)

            # Sleep to maintain loop rate
            rate.sleep()

    def joint_angles_callback(self, joint_msg):
        # 处理接收到的 AllegroHand 关节角数据
        self.joint_states.position = joint_msg.position
        # self.joint_states.velocity = joint_msg.velocity
        print('joint position:\n', self.joint_pos)
        # print('joint velocity:\n', self.joint_vel)
    
    def tactile_callback(self, tac_msg):
        self.index_tip_value = np.reshape(tac_msg.index_tip_Value, [12, 6])
        self.index_mid_value = np.reshape(tac_msg.index_mid_Value, [6, 6])
        self.index_end_value = np.reshape(tac_msg.index_end_Value, [6, 6])

        self.middle_tip_value = np.reshape(tac_msg.middle_tip_Value, [12, 6])
        self.middle_mid_value = np.reshape(tac_msg.middle_mid_Value, [6, 6])
        self.middle_end_value = np.reshape(tac_msg.middle_end_Value, [6, 6])

        self.ring_tip_value = np.reshape(tac_msg.ring_tip_Value, [12, 6])
        self.ring_mid_value = np.reshape(tac_msg.ring_mid_Value, [6, 6])
        self.ring_end_value = np.reshape(tac_msg.ring_end_Value, [6, 6])  
        self.palm_value = np.array(tac_msg.palm_Value)

        print('index_tip_value:\n', self.index_tip_value)
        print('index_mid_value:\n', self.index_mid_value)
        print('index_end_value:\n', self.index_end_value)
        print('palm_value:\n', self.palm_value)

        '''combine the tactile data to taxel_data'''
        self.tactile_data[:72] = self.index_tip_value  # 72
        self.tactile_data[72:108] = self.index_mid_value  # 36
        self.tactile_data[108:144] = self.index_end_value  # 36
        self.tactile_data[144:216] = self.middle_tip_value  # 72
        self.tactile_data[216:252] = self.middle_mid_value  # 36
        self.tactile_data[252:288] = self.middle_end_value  # 36
        self.tactile_data[288:360] = self.ring_tip_value  # 72
        self.tactile_data[360:396] = self.ring_mid_value  # 36
        self.tactile_data[396:432] = self.ring_end_value  # 36
        self.tactile_data[432:504] = self.thumb_tip_value  # 72
        self.tactile_data[504:540] = self.thumb_mid_value  # 36
        self.tactile_data[540:653] = self.palm_value  # 113

if __name__ == '__main__':
    try:
        controller = AllegroHandController()
    except rospy.ROSInterruptException:
        pass
