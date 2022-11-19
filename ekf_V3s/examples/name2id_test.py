import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import util_geometry as ug
import numpy as np

from mujoco_py import const

# to evaluate the inverse kinematic of finger. we add marker at the start position
# and end position, finger tip frame linearly move from start_p to end_p

hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
model, sim, viewer = mu_env.init_mujoco()
ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)


# init robot
rob_control = robcontrol
rob_control.robot_init(sim)
mu_env.Camera_set(viewer, model)
sim.model.eq_active[0] = True

for i in range(500):
    sim.step()
    viewer.render()

# move robotic arm to pre-grasping posture
sim.data.mocap_pos[0] = ctrl_wrist_pos
sim.data.mocap_quat[0] = ctrl_wrist_quat
for _ in range(500):
    sim.step()
    viewer.render()


np.set_printoptions(suppress=True)

# #sensor test
# print (model._sensor_id2name[144])
# print (model.sensor_name2id('touch_0_5_11'))
#
# #actuator test
# print (model.actuator_name2id('finger_0_0'))
# print (model.actuator_name2id('finger_1_0'))
# print (model.actuator_name2id('finger_2_0'))
# print (model.actuator_name2id('finger_3_0'))
#
# print ('ur_j0 motor ', model.actuator_name2id('shoulder_pan_T'))
# print ('ur_j1 motor ', model.actuator_name2id('shoulder_lift_T'))
# print ('ur_j2 motor ', model.actuator_name2id('forearm_T'))
# print ('ur_j3 motor ', model.actuator_name2id('wrist_1_T'))
# print ('ur_j4 motor ', model.actuator_name2id('wrist_2_T'))
# print ('ur_j5 motor ', model.actuator_name2id('wrist_3_T'))
#
#
# #joint test
# print('ur_j0 readout ', model.joint_name2id('shoulder_pan_joint'))
# print('ur_j1 readout ', model.joint_name2id('shoulder_lift_joint'))
# print('ur_j2 readout ', model.joint_name2id('elbow_joint'))
# print('ur_j3 readout ', model.joint_name2id('wrist_1_joint'))
# print('ur_j4 readout ', model.joint_name2id('wrist_2_joint'))
# print('ur_j5 readout ', model.joint_name2id('wrist_3_joint'))


print (model.joint_name2id('joint_0.0'))
print (model.joint_name2id('joint_1.0'))
print (model.joint_name2id('joint_2.0'))
print (model.joint_name2id('joint_3.0'))

print('jnt type is ', model.jnt_type[model.joint_name2id('joint_0.0')])
#
# print (model.joint_name2id('joint_4.0'))
# print (model.joint_name2id('joint_5.0'))
# print (model.joint_name2id('joint_6.0'))
# print (model.joint_name2id('joint_7.0'))
#
# print (model.joint_name2id('joint_8.0'))
# print (model.joint_name2id('joint_9.0'))
# print (model.joint_name2id('joint_10.0'))
# print (model.joint_name2id('joint_11.0'))
#
# print (model.joint_name2id('joint_12.0'))
# print (model.joint_name2id('joint_13.0'))
# print (model.joint_name2id('joint_14.0'))
# print (model.joint_name2id('joint_15.0'))


# print(model._joint_id2name[120])
# print(model._joint_id2name[121])
# print(model._joint_id2name[122])
# print(model._joint_id2name[123])








