#!/usr/bin/env python
import rospy
import numpy as np

from sensor_msgs.msg import JointState
from allegro_tactile_sensor.msg import tactile_msgs
from geometry_msgs.msg import TransformStamped
from message_filters import TimeSynchronizer, Subscriber



class AllegroHandController:
    def __init__(self):
        # 初始化订阅者和发布者
        self.joint_angles = JointState()
        self.joint_command_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=1)

        # 初始化同步订阅者
        joint_angles_sub = Subscriber('/allegroHand_0/joint_states', JointState)
        tactile_sub = Subscriber('/allegro_tactile', tactile_msgs)
        vicon_sub = Subscriber('/vicon/fjl/fjl', TransformStamped)
        ts = TimeSynchronizer([joint_angles_sub, tactile_sub, vicon_sub], queue_size=1, allow_headerless=True)
        ts.registerCallback(self.callback)

        # init object pose
        self.object_pose = np.zeros(7)


    def callback(self, joint_msg, tac_msg, vicon_msg):
        # 处理接收到的 AllegroHand 关节角数据
        self.joint_pos = joint_msg.position
        self.joint_vel = joint_msg.velocity
        print('joint position:\n', self.joint_pos)
        print('joint velocity:\n', self.joint_vel)

        # 处理接收到的触觉数据
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

        # 处理接收到的Vicon数据
        self.object_pose[0] = vicon_msg.transform.translation.x
        self.object_pose[1] = vicon_msg.transform.translation.y
        self.object_pose[2] = vicon_msg.transform.translation.z
        self.object_pose[3] = vicon_msg.transform.rotation.x
        self.object_pose[4] = vicon_msg.transform.rotation.y
        self.object_pose[5] = vicon_msg.transform.rotation.z
        self.object_pose[6] = vicon_msg.transform.rotation.w
        print('Object Pose:\n', self.object)

    def joint_command_publish(self):
        # 发布关节角指令
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            joint_command = JointState()
            joint_command.name = self.joint_angles.name
            joint_command.velocity = [0]*16  # 设置速度为0
            joint_command.effort = [0]*16  # 设置力矩为0
            joint_command.position = [0]*16  # 将所有关节的目标位置设置为0

            # joint_command.position[0:16] = [0.5] * 16

            # # 将手指和拇指关节的目标位置设置为0.5，使其弯曲
            joint_command.position[0] = 0.5
            joint_command.position[1] = 0.5
            joint_command.position[2] = 0.5
            joint_command.position[3] = 0.5
            joint_command.position[4] = 0.5
            joint_command.position[5] = 0.5
            joint_command.position[6] = 0.5
            joint_command.position[7] = 0.5
            joint_command.position[8] = 0.5
            joint_command.position[9] = 0.5
            joint_command.position[10] = 0.5
            joint_command.position[11] = 0.5
            joint_command.position[12] = 0.5
            joint_command.position[13] = 0.5
            joint_command.position[14] = 0.5
            joint_command.position[15] = 0.5

            self.joint_command_pub.publish(joint_command)
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('allegrohand_controller', anonymous=True)
        allegrohand_controller = AllegroHandController()
        allegrohand_controller.joint_command_publish()
        # print(allegrohand_controller.joint_pos)
    except rospy.ROSInterruptException:
        pass
