import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import tactile_perception
import mujoco_py

import util_geometry as ug
import viz
from mujoco_py import const


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
    tacperception.get_tip_center_pose(sim, model, 'ff_tip', 'link_3.0_tip')
    tacperception.get_tip_center_pose(sim, model, 'mf_tip', 'link_7.0_tip')
    tacperception.get_tip_center_pose(sim, model, 'rf_tip', 'link_11.0_tip')
    tacperception.get_tip_center_pose(sim, model, 'th_tip', 'link_15.0_tip')

    # rob_control.active_fingers_taxels_render(sim, viewer, tacperception)
    des_press = 0.01
    if tacperception.is_finger_contact(sim, 'ff') == True:
        ff_cur_pose_tip, ff_cur_taxel_name, ff_cur_press_tip = \
            tacperception.get_contact_feature(sim, model, 'ff')
    else:
        ff_cur_press_tip = 0.0

    ff_des_pose_tip = ug.posquat2trans(tacperception.fftip_center_taxel_pose)
    ff_tran_cur_pose_tip = ug.posquat2trans(ff_cur_pose_tip)
    # position_tip = des_pose_tip[0:3, 3]
    # mat_rot_tip = des_pose_tip[0:3, 0:3]
    # position_world, mat_rot_world = ug.pose_trans_part_to_world(sim, \
    #                                                            'link_3.0_tip', position_tip, mat_rot_tip)

    # cur_position_tip = tran_cur_pose_tip[0:3, 3]
    # cur_rot_tip = tran_cur_pose_tip[0:3, 0:3]
    # cur_position_world, cur_rot_world = ug.pose_trans_part_to_world(sim, \
    #                                                            'link_3.0_tip', cur_position_tip, cur_rot_tip)
    # viz.geo_visual(viewer, position_world, mat_rot_world, 0.001, const.GEOM_BOX, 0, 'r')
    # viz.geo_visual(viewer, cur_position_world, \
    #                    cur_rot_world, 0.001, const.GEOM_BOX, 0, "z")
    # # viz.cor_frame_visual(viewer,cur_position_world,cur_rot_world,0.2,'cf')
    rob_control.tip_servo_control(sim, viewer, model, 'ff', ff_tran_cur_pose_tip, ff_cur_taxel_name, \
                                      ff_des_pose_tip, des_press - ff_cur_press_tip)
    # print('delta_pressure ', des_press - cur_press_tip)
    # posquat_palm_world = ug.get_relative_posquat(sim, "world", "palm_link")
    # T_palm_world = ug.posquat2trans(posquat_palm_world)
    # # visualize coordinate frame of the global, palm
    # viz.cor_frame_visual(viewer, T_palm_world[:3, 3], T_palm_world[:3, :3], 0.3, "Palm")

    # if tacperception.is_finger_contact(sim, 'mf') == True:
    #     mf_cur_pose_tip, mf_cur_taxel_name, mf_cur_press_tip = \
    #         tacperception.get_contact_feature(sim, model, 'mf')
    # else:
    #     mf_cur_press_tip = 0.0
    #
    # mf_des_pose_tip = ug.posquat2trans(tacperception.mftip_center_taxel_pose)
    # mf_tran_cur_pose_tip = ug.posquat2trans(mf_cur_pose_tip)
    # rob_control.tip_servo_control(sim, viewer, model, 'mf', mf_tran_cur_pose_tip, mf_cur_taxel_name, \
    #                                   mf_des_pose_tip, des_press - mf_cur_press_tip)

    # if tacperception.is_finger_contact(sim, 'rf') == True:
    #     rf_cur_pose_tip, rf_cur_taxel_name, rf_cur_press_tip = \
    #         tacperception.get_contact_feature(sim, model, 'rf')
    # else:
    #     rf_cur_press_tip = 0.0
    #
    # rf_des_pose_tip = ug.posquat2trans(tacperception.rftip_center_taxel_pose)
    # rf_tran_cur_pose_tip = ug.posquat2trans(rf_cur_pose_tip)
    # rob_control.tip_servo_control(sim, viewer, model, 'rf', rf_tran_cur_pose_tip, rf_cur_taxel_name, \
    #                                   rf_des_pose_tip, des_press - rf_cur_press_tip)

    if tacperception.is_finger_contact(sim, 'th') == True:
        th_cur_pose_tip, th_cur_taxel_name, th_cur_press_tip = \
            tacperception.get_contact_feature(sim, model, 'th')
    else:
        th_cur_press_tip = 0.0

    th_des_pose_tip = ug.posquat2trans(tacperception.thtip_center_taxel_pose)
    th_tran_cur_pose_tip = ug.posquat2trans(th_cur_pose_tip)
    rob_control.tip_servo_control(sim, viewer, model, 'th', th_tran_cur_pose_tip, th_cur_taxel_name, \
                                      th_des_pose_tip, des_press - th_cur_press_tip)
    sim.step()
    viewer.render()