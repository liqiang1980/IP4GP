from mujoco_py import load_model_from_path, MjSim, MjViewer, const
import numpy as np
import util_geometry as ug
import fcl
import fcl_python


def Camera_set(viewer, model):
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = model.stat.extent * 1.0
    viewer.cam.lookat[2] += .1
    viewer.cam.lookat[0] += .5
    viewer.cam.lookat[1] += .5
    viewer.cam.elevation = -0
    viewer.cam.azimuth = 0

def init_mujoco(filename = "../../robots/UR5_tactile_allegro_hand.xml"):
    xml_path = filename
    model = load_model_from_path(xml_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    return model, sim, viewer

def init_robot_object_mujoco(sim, object_param):
    pose_cup = ug.get_body_posquat(sim, object_param[0])
    trans_cup = ug.posquat2trans(pose_cup)
    # pregrasping related pose ref:cup
    # trans_pregrasp = np.array([[0, 0, 1, 0.1],
    #                            [0, 1, 0, -0.23],
    #                            [-1, 0, 0, 0.05],
    #                            [0, 0, 0, 1]])

    trans_pregrasp = np.array([[0, 0, 1, 0.1],
                               [0, 1, 0, -0.23],
                               [-1, 0, 0, 0.02],
                               [0, 0, 0, 1]])
    # ref: palm
    posequat = ug.get_prepose_posequat(trans_cup, trans_pregrasp)

    ctrl_wrist_pos = posequat[:3]
    ctrl_wrist_quat = posequat[3:]
    return ctrl_wrist_pos, ctrl_wrist_quat

def config_fcl(obj1_name, obj2_name):
    # obj_cup = fcl_python.OBJ("cup_1.obj")
    obj_cup = fcl_python.OBJ(obj1_name)
    verts_cup = obj_cup.get_vertices()
    tris_cup = obj_cup.get_faces()

    # Create mesh geometry
    mesh_cup = fcl.BVHModel()
    mesh_cup.beginModel(len(verts_cup), len(tris_cup))
    mesh_cup.addSubModel(verts_cup, tris_cup)
    mesh_cup.endModel()
    print("len_verts_cup:", len(verts_cup))

    # fcl库加载finger_tip 的 BVH模型
    # obj_fingertip = fcl_python.OBJ("fingertip_part.obj")
    obj_fingertip = fcl_python.OBJ(obj2_name)
    verts_fingertip = obj_fingertip.get_vertices()
    tris_fingertip = obj_fingertip.get_faces()
    print("len_verts_fingertip:", len(verts_fingertip))
    print("len_tris_fingertip:", len(tris_fingertip))

    mesh_fingertip = fcl.BVHModel()
    mesh_fingertip.beginModel(len(verts_fingertip), len(tris_fingertip))
    mesh_fingertip.addSubModel(verts_fingertip, tris_fingertip)
    mesh_fingertip.endModel()








# def move_interperate_point(sim, desire_pos_quat, curr_posquat, viewer=None):
#     # curr_posquat = get_relative_posquat(sim, "palm_link", "link_3.0_tip")
#     delta_k = 5
#     # X = np.arange(0, 1, 1)
#     # Y = [curr_posquat, desire_pos_quat]
#     interpolate_point = []
#     for i in range(1,delta_k+1):
#         interpolate_point.append(curr_posquat + (desire_pos_quat-curr_posquat)/delta_k*i)
#
#     count_execute = 0;
#     for k,inter in enumerate(interpolate_point):
#         done_execute = False
#         # move_ik_kdl_finger_wdls(sim, inter)
#         print("inter:", inter)
#         while(count_execute < 200):
#             done_execute = move_ik_kdl_finger_wdls(sim, inter)
#             count_execute += 1
#             sim.step()
#             viewer.render()
#         count_execute = 0

#对这里进行修正，进行修改即可
# def force_control(sim, force_set, cur_force):
# 	kp, ki, kd = 0.0, 0.3, 0.0
# 	pid = pid.PID(kp, ki, kd)
# 	transfom_factor = 0.000003
# 	setpoint = 10
#
#     transform_base2tip3 = get_relative_posquat(sim, "base_link", "link_3.0 _tip")
#     rot = posquat2trans(transform_base2tip3)[0:3, 0:3]
#     pid_out = pid.calc(cur_force, force_set)
#     ze = np.array([0, 0, pid_out*transfom_factor]).transpose()
#     ze = np.matmul(rot, ze)
#     z = pos-ze
#     transform_base2tip3[:3] = z
#     desire_pos_quat_in_force = np.array();
#     move_ik_kdl_finger_wdls(sim, desire_pos_quat_in_force)


######################     Author: lqg    ##############################


def execute_grasp(sim, viewer=None):
    while not (sum(sim.data.sensordata[-2:] > np.array([0, 0])) == 2):
        sim.data.ctrl[7] = sim.data.qpos[7] - 0.01
        sim.step()
        viewer.render()


