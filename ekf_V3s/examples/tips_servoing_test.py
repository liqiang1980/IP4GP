import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import tactile_perception
import mujoco_py

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

for i in range(1000):
    # tacperception.get_hand_tip_center_pose(sim, model, 'world')

    tacperception.get_tip_center_pose(sim, model, 'ff_tip', 'link_3.0_tip')


    # tacperception.get_tip_center_pose(sim, model, 'mf_tip', 'link_7.0_tip')
    # tacperception.get_tip_center_pose(sim, model, 'rf_tip', 'link_11.0_tip')
    # tacperception.get_tip_center_pose(sim, model, 'th_tip', 'link_15.0_tip')

    # rob_control.active_fingers_taxels_render(sim, viewer, tacperception)

    if tacperception.is_finger_contact(sim, 'ff') == True:
        cur_pose, cur_taxel_name, cur_press = tacperception.get_contact_feature(sim, model, 'ff')
        print('pose ', cur_pose, i)
        print('press ', cur_press, i)
        des_pose = ug.posquat2trans(tacperception.fftip_center_taxel_pose)
        des_press = 0.5
        position = des_pose[0:3, 3]
        mat_rot = des_pose[0:3, 0:3]
        viz.geo_visual(viewer, position, mat_rot, 0.001, const.GEOM_BOX, 0, 'r')
        viz.geo_visual(viewer, ug.posquat2trans(cur_pose)[0:3, 3], \
                       ug.posquat2trans(cur_pose)[0:3, 0:3], 0.001, const.GEOM_BOX, 0, "z")
        rob_control.tip_servo_control(sim, model, 'ff', ug.posquat2trans(cur_pose), cur_taxel_name, \
                                      des_pose, des_press - cur_press)

    # if tacperception.is_finger_contact(sim, 'mf') == True:
    #     cur_tac_p = ug.posquat2trans(tacperception.get_contact_taxel_position(sim, model, 'mf', "link_7.0_tip"))
    #     goal_tac_p = ug.posquat2trans(tacperception.mftip_center_taxel_pose)
    #     position = goal_tac_p[0:3, 3]
    #     mat_rot = goal_tac_p[0:3, 0:3]
    #     viz.geo_visual(viewer, position, mat_rot, 0.001, const.GEOM_BOX, 0, 'r')
    #     viz.geo_visual(viewer, cur_tac_p[0:3, 3], cur_tac_p[0:3, 0:3], 0.001, const.GEOM_BOX, 0, "z")
    #     # rob_control.tip_servo_control(sim, model, 'mf', cur_tac_p, goal_tac_p)
    # if tacperception.is_finger_contact(sim, 'rf') == True:
    #     cur_tac_p = ug.posquat2trans(tacperception.get_contact_taxel_position(sim, model, 'rf', "link_11.0_tip"))
    #     goal_tac_p = ug.posquat2trans(tacperception.rftip_center_taxel_pose)
    #     position = goal_tac_p[0:3, 3]
    #     mat_rot = goal_tac_p[0:3, 0:3]
    #     viz.geo_visual(viewer, position, mat_rot, 0.001, const.GEOM_BOX, 0, 'r')
    #     viz.geo_visual(viewer, cur_tac_p[0:3, 3], cur_tac_p[0:3, 0:3], 0.001, const.GEOM_BOX, 0, "z")
    #     # rob_control.tip_servo_control(sim, model, 'rf', cur_tac_p, goal_tac_p)
    # if tacperception.is_finger_contact(sim, 'th') == True:
    #     cur_tac_p = ug.posquat2trans(tacperception.get_contact_taxel_position(sim, model, 'th', "link_15.0_tip"))
    #     goal_tac_p = ug.posquat2trans(tacperception.thtip_center_taxel_pose)
    #     position = goal_tac_p[0:3, 3]
    #     mat_rot = goal_tac_p[0:3, 0:3]
    #     viz.geo_visual(viewer, position, mat_rot, 0.001, const.GEOM_BOX, 0, 'r')
    #     viz.geo_visual(viewer, cur_tac_p[0:3, 3], cur_tac_p[0:3, 0:3], 0.001, const.GEOM_BOX, 0, "z")
    #     # rob_control.tip_servo_control(sim, model, 'th', cur_tac_p, goal_tac_p)


    sim.step()
    viewer.render()

