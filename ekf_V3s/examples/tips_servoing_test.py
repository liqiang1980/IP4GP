import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import tactile_perception

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



# tactile servoing on tips
hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_frozen.xml")
ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)

tacperception = tactile_perception.cls_tactile_perception()

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

rob_control.pre_thumb(sim, viewer)
rob_control.fingers_contact(sim, viewer, tacperception)

for _ in range(1000):
    sim.step()
    viewer.render()

