import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import util_geometry as ug
import numpy as np

from urdf_parser_py.urdf import URDF


hand_param, object_param, alg_param = config_param.pass_arg()

# robot model from urdf
robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')

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

rob_control.hand_pregrasp(sim, viewer)

np.set_printoptions(suppress=True)

angles = rob_control.get_cur_jnt(sim)


pose_link_30_tip_l = ug.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
pos_p_palm, pos_o_palm = ug.posquat2pos_p_o(pose_link_30_tip_l)
print('******************ff pose_p in palm frame', pos_p_palm)
print('******************ff pose_o in palm frame ')
print(pos_o_palm)

pos_p_c, pos_o_c = ug.pose_trans_palm_to_world(sim, pos_p_palm, pos_o_palm)
print('****************** computed ff pose_p in world frame', pos_p_c)
print('****************** computed ff pose_o in world frame ')
print(pos_o_c)

pose_link_30_tip_w = ug.get_relative_posquat(sim, "world", "link_3.0_tip")
pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_link_30_tip_w)
print('****************** real ff pose_p in palm frame', pos_p_world)
print('****************** real ff pose_o in palm frame ')
print(pos_o_world)




# pose_tac_test_tip = ug.get_relative_posquat(sim, "palm_link", "touch_0_3_3")
# pos_p, pos_o = ug.posquat2pos_p_o(pose_tac_test_tip)
# print('******************tac pose_p in mujoco xml', pos_p)
# print('******************tac pose_o in mujoco xml ')
# print(pos_o)





