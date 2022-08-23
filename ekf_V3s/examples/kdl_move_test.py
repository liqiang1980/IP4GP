import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import util_geometry as ug
import numpy as np
from enum import Enum
import time

from pykdl_utils.kdl_kinematics import KDLKinematics

from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import viz
from mujoco_py import const
from tactile_perception import taxel_pose



# to evaluate the inverse kinematic of finger. we add marker at the start position
# and end position, finger tip frame linearly move from start_p to end_p

hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_frozen.xml")
ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)


# init robot
rob_control = robcontrol.ROBCTRL()
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
p_I, o_I, q_start = rob_control.mf_taxel_poseture_I(sim, viewer)
p_II, o_II, q_end = rob_control.mf_taxel_poseture_II(sim, viewer)

print('q_start ', q_start)
p_start = taxel_pose()
p_end = taxel_pose()

p_start.position = p_I
p_start.orientation = o_I

p_end.position = p_II
p_end.orientation = o_II

ik_type = robcontrol.IK_type(1)
#use the instantaneous velocity solution
rob_control.p2p_v_ik(sim, viewer, 3, p_start, p_end, ik_type)
#use the direct position solution
# q_est = rob_control.p2p_p_ik(sim, viewer, p_start, q_end)
# rob_control.moveto_jnt(sim, viewer, 'mf', q_est, 2000)



