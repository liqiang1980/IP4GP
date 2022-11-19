import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import ekf
import tactile_perception
import util_geometry as ug
import numpy as np

from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF

# load task-related parameters
hand_param, object_param, alg_param = config_param.pass_arg()

# robot model from urdf
robot = URDF.from_xml_file('../../robots/allegro_hand_right_with_tactile.urdf')
kdl_kin0 = KDLKinematics(robot, "palm_link", "link_3.0_tip")
kdl_kin1 = KDLKinematics(robot, "palm_link", "link_7.0_tip")
kdl_kin2 = KDLKinematics(robot, "palm_link", "link_11.0_tip")
kdl_kin3 = KDLKinematics(robot, "palm_link", "link_15.0_tip")

kdl_kin_tac_test = KDLKinematics(robot, "palm_link", "touch_0_3_3")

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

rob_control.hand_zero(sim, viewer)

np.set_printoptions(suppress=True)

angles = rob_control.get_cur_jnt(sim)
ff_q = angles[0:4]
mf_q = angles[4:8]
rf_q = angles[8:12]
th_q = angles[12:16]

pose_link_30_tip = ug.get_relative_posquat(sim, "palm_link", "link_3.0_tip")
pos_p, pos_o = ug.posquat2pos_p_o(pose_link_30_tip)
print('******************ff pose_p in mujoco xml', pos_p)
print('******************ff pose_o in mujoco xml ')
print(pos_o)

kdl_p, kdl_o = kdl_kin0.FK(ff_q)
print('******************ff pose_p in urdf with kdl')
print (kdl_p)
print('******************ff pose_o in urdf with kdl')
print(kdl_o)

pose_link_70_tip = ug.get_relative_posquat(sim, "palm_link", "link_7.0_tip")
pos_p, pos_o = ug.posquat2pos_p_o(pose_link_70_tip)
print('******************mf pose_p in mujoco xml', pos_p)
print('******************mf pose_o in mujoco xml ')
print(pos_o)

kdl_p, kdl_o = kdl_kin1.FK(mf_q)
print('******************mf pose_p in urdf with kdl')
print (kdl_p)
print('******************mf pose_o in urdf with kdl')
print(kdl_o)

pose_link_110_tip = ug.get_relative_posquat(sim, "palm_link", "link_11.0_tip")
pos_p, pos_o = ug.posquat2pos_p_o(pose_link_110_tip)
print('******************rf pose_p in mujoco xml', pos_p)
print('******************rf pose_o in mujoco xml ')
print(pos_o)

kdl_p, kdl_o = kdl_kin2.FK(rf_q)
print('******************rf pose_p in urdf with kdl')
print (kdl_p)
print('******************rf pose_o in urdf with kdl')
print(kdl_o)

pose_link_150_tip = ug.get_relative_posquat(sim, "palm_link", "link_15.0_tip")
pos_p, pos_o = ug.posquat2pos_p_o(pose_link_150_tip)
print('******************th pose_p in mujoco xml', pos_p)
print('******************th pose_o in mujoco xml ')
print(pos_o)


kdl_p, kdl_o = kdl_kin3.FK(th_q)
print('******************th pose_p in urdf with kdl')
print (kdl_p)
print('******************th pose_o in urdf with kdl')
print(kdl_o)

pose_tac_test_tip = ug.get_relative_posquat(sim, "palm_link", "touch_0_3_3")
pos_p, pos_o = ug.posquat2pos_p_o(pose_tac_test_tip)
print('******************tac pose_p in mujoco xml', pos_p)
print('******************tac pose_o in mujoco xml ')
print(pos_o)


kdl_p, kdl_o = kdl_kin_tac_test.FK(ff_q)
print('******************tac pose_p in urdf with kdl')
print (kdl_p)
print('******************tac pose_o in urdf with kdl')
print(kdl_o)




