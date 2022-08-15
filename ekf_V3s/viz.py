import util_geometry as ug
import numpy as np
from mujoco_py import const

def cor_frame_visual(viewer, position, mat_rot, length, frame_name):
    x_rot = ug.vec2rot(mat_rot[:, 0])
    y_rot = ug.vec2rot(mat_rot[:, 1])
    z_rot = ug.vec2rot(mat_rot[:, 2])
    viewer.add_marker(pos=position, mat=x_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.001, 0.001, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=y_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.001, 0.001, length]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=z_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.001, 0.001, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))

def geo_visual(viewer, position, mat_rot, length, geo_type, finger_id, c_semantic):
    if geo_type == const.GEOM_ARROW:
        viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label="vec " + str(finger_id) + c_semantic,
                      size=np.array([0.001, 0.001, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    if geo_type == const.GEOM_BOX:
        if c_semantic == 'h':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label="point" + str(finger_id) + c_semantic,
                      size=np.array([length, length, length]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
        if c_semantic == 'z':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label="point" + str(finger_id) + c_semantic,
                      size=np.array([length, length, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))

def active_taxels_visual(sim, model, viewer, taxels):
    for i in taxels:
        active_taxel_name = sim.model._sensor_id2name[i]
        pose_taxel_w = ug.get_relative_posquat(sim, "world", active_taxel_name)
        pos_p_world, pos_o_world = ug.posquat2pos_p_o(pose_taxel_w)
        # print('****************** real ff pose_p in palm frame', pos_p_world)
        # print('****************** real ff pose_o in palm frame ')
        # print(pos_o_world)
        viewer.add_marker(pos=pos_p_world, mat=pos_o_world, type=const.GEOM_BOX,label="",
                      size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))

def touch_visual(sim, model, viewer, a):
    global max_size
    truth = ug.get_relative_posquat(sim, "base_link", "cup")

    save_point_use = np.array([[0, 0, 0, 0, 0, 0, 0]])
    save_point_use = np.append(save_point_use, np.array([truth]), axis=0)
    for i in a:
        for k, l in enumerate(i):
            s_name = model._sensor_id2name[i[k]]
            sensor_pose = ug.get_body_posquat(sim, s_name)
            relative_pose = ug.get_relative_posquat(sim, "base_link", s_name)
            save_point_use = np.append(save_point_use, np.array([relative_pose]), axis=0)

            rot_sensor = ug.as_matrix(np.hstack((sensor_pose[4:], sensor_pose[3])))
            # 用于控制方向，触觉传感器的方向问题
            test_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            viewer.add_marker(pos=sensor_pose[:3], mat=test_rot, type=const.GEOM_ARROW, label="contact",
                              size=np.array([0.001, 0.001, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

    # if save_point_use.shape[0] > max_size:
    #     save_point_output = save_point_use
    #     np.save("output.npy", save_point_output)
    #     where_a = np.where(np.array(sim.data.sensordata) > 0.0)
    #
    # max_size = max(save_point_use.shape[0], max_size)
    viewer.render()
