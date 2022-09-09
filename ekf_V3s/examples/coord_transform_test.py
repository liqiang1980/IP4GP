import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import util_geometry as ug
import numpy as np
import math

# load task-related parameters
hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
model, sim, viewer = mu_env.init_mujoco()
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
for _ in range(50):
    sim.step()
    viewer.render()

# Thumb root movement
rob_control.pre_thumb(sim, viewer)
# other fingers start moving with the different velocity (contact/without contact)
for ii in range(1000):
    print('hand_param[2][1] ', hand_param[2][1])
    if hand_param[1][1] == '1':
        rob_control.index_finger(sim, 0.0055, 0.00001)
    if hand_param[2][1] == '1':
        rob_control.middle_finger(sim, 0.0016, 0.00001)
    if hand_param[3][1] == '1':
        rob_control.ring_finger(sim, 0.02, 0.00001)
    if hand_param[4][1] == '1':
        rob_control.thumb(sim, 0.0003, 0.00001)
    gd_posquat = ug.get_relative_posquat(sim, "palm_link", "cup")
    gd_state = ug.posquat2posrotvec_hacking(gd_posquat)
    rm = ug.getrotvecfromposquat(gd_posquat)
    print('ii ', ii)
    #compute rot_vec
    theta = math.acos((np.trace(rm) - 1.0) / 2.0)
    omega = np.array([0., 0., 0.])
    omega[0] = (rm[2][1] - rm[1][2]) / (2 * math.sin(theta))
    omega[1] = (rm[0][2] - rm[2][0]) / (2 * math.sin(theta))
    omega[2] = (rm[1][0] - rm[0][1]) / (2 * math.sin(theta))
    print('theta ', theta)
    print('omega is ', omega[0], omega[1], omega[2] )

    rob_control.x_gt_palm = np.vstack((rob_control.x_gt_palm, gd_state))
    np.savetxt('../ekf_obj_pose/x_gt_palm.txt', rob_control.x_gt_palm)
    sim.step()
    viewer.render()