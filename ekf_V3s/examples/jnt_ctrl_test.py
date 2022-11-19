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

rob_control.hand_pregrasp(sim, viewer)

np.set_printoptions(suppress=True)
angles = rob_control.get_cur_jnt(sim)

print('current jnt ', angles)

ff_q = angles[0:4]
mf_q = angles[4:8]
rf_q = angles[8:12]
th_q = angles[12:16]









