import util_geometry as ug
import numpy as np
from mujoco_py import const
import tactile_allegro_mujo_const
from scipy.spatial.transform import Rotation


def cor_frame_visual(viewer, position, mat_rot, length, frame_name):
    x_rot = ug.vec2rot(mat_rot[:, 0])
    y_rot = ug.vec2rot(mat_rot[:, 1])
    z_rot = ug.vec2rot(mat_rot[:, 2])
    viewer.add_marker(pos=position, mat=x_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.0005, 0.0005, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=y_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.0005, 0.0005, length]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=z_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.0005, 0.0005, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))


def geo_visual(viewer, position, mat_rot, length, geo_type, finger_id, c_semantic):
    if geo_type == const.GEOM_ARROW:
        if c_semantic == 'h':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label=" ",
                              size=np.array([0.001, 0.001, length]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
        if c_semantic == 'z':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label=" ",
                              size=np.array([0.001, 0.001, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    if geo_type == const.GEOM_BOX:
        if c_semantic == 'r':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label=" ",
                              size=np.array([length, length, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
        if c_semantic == 'h':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label=" ",
                              size=np.array([length, length, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
        if c_semantic == 'z':
            viewer.add_marker(pos=position, mat=mat_rot, type=geo_type, label=" ",
                              size=np.array([length, length, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))


def active_taxels_visual(viewer, taxels_pose, lbl):
    if lbl == 'gt':
        for i in range(len(taxels_pose)):
            viewer.add_marker(pos=taxels_pose[i].position, mat=taxels_pose[i].orientation, type=const.GEOM_SPHERE,
                              label="",
                              size=np.array([0.0005, 0.0005, 0.0005]), rgba=np.array([0.3, 0.3, 0.3, 1.0]))
    if lbl == 'fk':
        for i in range(len(taxels_pose)):
            viewer.add_marker(pos=taxels_pose[i].position, mat=taxels_pose[i].orientation, type=const.GEOM_SPHERE,
                              label="",
                              size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))


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


def vis_frame_in_world(sim, viewer, part_name):
    posquat_part_world = ug.get_relative_posquat(sim, "world", part_name)
    T_part_world = ug.posquat2trans(posquat_part_world)
    cor_frame_visual(viewer, T_part_world[:3, 3], T_part_world[:3, :3], 0.2, part_name)


def vis_state_contact(sim, viewer, tacp, z_t, h_t, robctrl, char, fk):
    """
    All visualization: z_t, h_t and cup marker
    """
    posquat_palm_world = ug.get_relative_posquat(sim, "world", "palm_link")  # wxyz
    T_palm_world = ug.posquat2trans(posquat_palm_world)  # wxyz to xyzw
    # visualize coordinate frame of the global, palm
    # cor_frame_visual(viewer, T_palm_world[:3, 3], T_palm_world[:3, :3], 0.3, "Palm")
    """ z_t visualization """
    # for i, f_part in enumerate(robctrl.f_param):
    #     f_name = f_part[0]
    #     zt_idx = [3 * i, 3 * i + 3]
    #     if tacp.is_contact[f_name]:
    #         pos_zt_palm = z_t[zt_idx[0]: zt_idx[1]]
    #         pos_zt_world = np.ravel(T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_zt_palm.T))
    #         if tactile_allegro_mujo_const.PN_FLAG == 'pn':
    #             rot_zt_palm = ug.vec2rot(z_t[3 * i + 12:3 * i + 15])
    #             rot_zt_world = np.matmul(T_palm_world[:3, :3], rot_zt_palm)
    #             geo_visual(viewer, pos_zt_world, rot_zt_world, 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "z")
    #             geo_visual(viewer, pos_zt_world, rot_zt_world, 0.1, tactile_allegro_mujo_const.GEOM_ARROW, i, "z")
    #         else:
    #             # geo_visual(viewer, pos_zt_world, np.eye(3), 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "z")
    #             viewer.add_marker(pos=pos_zt_world, mat=np.eye(3), type=const.GEOM_BOX, label=f_name+"_z",
    #                               size=np.array([0.001, 0.001, 0.001]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))  # red

            # draw linear vel of contact point (part of twist from ju)
            # from vel generate frame
            # vel_frame = ug.vec2rot(np.matmul(T_palm_world[:3, :3], ju_all[6*i: 6*i+3]))
            # print('vel_frame determinant ', np.linalg.det(vel_frame))
            # viz.geo_visual(viewer, pos_zt_world, vel_frame, 0.1, tactile_allegro_mujo_const.GEOM_ARROW, i, "z_vel")
            #
            # self.ct_p_z_position = np.vstack((self.ct_p_z_position, pos_zt_palm))
            # self.ct_g_z_position = np.vstack((self.ct_g_z_position, pos_zt_world))
            # np.set_printoptions(suppress=True)
            # np.savetxt('ct_g_z_position.txt', self.ct_g_z_position)
            # np.savetxt('ct_p_z_position.txt', self.ct_p_z_position)
    """ h_t visualization """
    # for i, f_part in enumerate(robctrl.f_param):
    #     f_name = f_part[0]
    #     ht_idx = [3 * i, 3 * i + 3]
    #     if tacp.is_contact[f_name]:
    #         pos_ht_palm = h_t[ht_idx[0]: ht_idx[1]]
    #         pos_ht_world = np.ravel(T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], pos_ht_palm.T))
    #         if tactile_allegro_mujo_const.PN_FLAG == 'pn':
    #             rot_ht_palm = ug.vec2rot(h_t[3 * i + 12:3 * i + 15])
    #             rot_ht_world = np.matmul(T_palm_world[:3, :3], rot_ht_palm)
    #             geo_visual(viewer, pos_ht_world, rot_ht_world, 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "h")
    #             geo_visual(viewer, pos_ht_world, rot_ht_world, 0.1, tactile_allegro_mujo_const.GEOM_ARROW, i, "h")
    #         else:
    #             # geo_visual(viewer, pos_ht_world, np.eye(3), 0.001, tactile_allegro_mujo_const.GEOM_BOX, i, "h")
    #             viewer.add_marker(pos=pos_ht_world, mat=np.eye(3), type=const.GEOM_BOX, label=f_name+"_h",
    #                               size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0.34, 0.98, 1., 1.0]))  # gray

    """ FK tip Visualization"""
    # for f_part in robctrl.f_param:
    #     f_name = f_part[0]
    #     T_tip_palm = fk.T_tip_palm[f_name]
    #     T_tip_world = np.matmul(T_palm_world, T_tip_palm)
    #     viewer.add_marker(pos=np.ravel(T_tip_world[:3, 3]), mat=T_tip_world[:3, :3], type=const.GEOM_ARROW,
    #                       label=f_name, size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))  # gray

    """ x_state (cup marker) Visualization """
    # pos_x_world = np.ravel(T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], x_state[:3]))
    pos_x_world = np.ravel(T_palm_world[:3, 3] + np.matmul(T_palm_world[:3, :3], robctrl.pos_cup_palm))
    # rot_x_palm = Rotation.from_rotvec(x_state[3:6]).as_matrix()
    rot_x_palm = robctrl.R_cup_palm
    # v, s = ug.normalize_scale(x_state[3:6])
    v, s = ug.normalize_scale(robctrl.rotvec_cup_palm)
    rot_x_world = np.matmul(T_palm_world[:3, :3], rot_x_palm)
    # cor_frame_visual(viewer, pos_x_world, rot_x_world, 0.2, "est_Obj")

    if (char == "v"):
        if int(robctrl.obj_param[3]) == 3 or int(robctrl.obj_param[3]) == 4:
            """Add marker type in mujoco_py.const: const.GEOM_MESH=7, const.GEOM_ARROW=100 ... """
            """For Cylinder, size=[semi-major axis, semi-minor axis, height], unit=Meter(m)"""
            viewer.add_marker(pos=pos_x_world, mat=rot_x_world, type=const.GEOM_CYLINDER, size=[0.04, 0.04, 0.08],
                              label=' ', rgba=np.array([0.0, 0.0, 1.0, 1.0]), dataid=0)
        else:
            # viewer.add_marker(pos=pos_x_world, mat=rot_x_world, type=7,
            #                   label=' ', rgba=np.array([0.0, 0.0, 1.0, 1.0]), dataid=0)
            viewer.add_marker(pos=pos_x_world, mat=rot_x_world, type=const.GEOM_MESH,
                              label=' ', rgba=np.array([0.0, 0.0, 1.0, 1.0]), dataid=0)

    # viewer.add_marker(pos=pos_x_world, mat=rot_x_world, type=tactile_allegro_mujo_const.GEOM_ARROW,
    #                   label="x_state", size=np.array([0.001, 0.001, 0.1]), rgba=np.array([0.34, 0.98, 1., 1.0]))

    iden_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    position = [0.0, 0.0, 0.0]
    cor_frame_visual(viewer, position, iden_rot, 0.3, 'frame')
