import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import ekf
import tactile_perception
import util_geometry as ug
import numpy as np
import viz


# load task-related parameters
hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
model, sim, viewer = mu_env.init_mujoco()
ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)

# instantiate ekf class
grasping_ekf = ekf.EKF()
grasping_ekf.set_contact_flag(False)
grasping_ekf.set_store_flag(alg_param[0])

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

rob_control.hand_zero(sim, viewer)

np.set_printoptions(suppress=True)
angles = rob_control.get_cur_jnt(sim)

pose_tac_test = ug.get_relative_posquat(sim, "world", "touch_0_3_3")
tac_pos_p, tac_pos_o = ug.posquat2pos_p_o(pose_tac_test)
print('******************tac pose_p in mujoco xml', tac_pos_p)
print('******************tac pose_o in mujoco xml ')
print(tac_pos_o)

pose_tip_test = ug.get_relative_posquat(sim, "world", "link_3.0_tip")
tip_pos_p, tip_pos_o = ug.posquat2pos_p_o(pose_tip_test)
print('******************ff tip pose_p in mujoco xml', tip_pos_p)
print('******************ff tip pose_o in mujoco xml ')
print(tip_pos_o)


for _ in range(5000):
    viz.cor_frame_visual(viewer, tac_pos_p, tac_pos_o, 0.1, 'tax_frame')
    viz.cor_frame_visual(viewer, tip_pos_p, tip_pos_o, 0.1, 'tip_frame')
    sim.step()
    viewer.render()



