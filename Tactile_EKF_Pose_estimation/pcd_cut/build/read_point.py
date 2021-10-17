import open3d as o3d
import numpy as np
import time, os
from open3d import *
import scipy.io as io


# object_path = './UR5/mesh/visual/bottle.STL'
#
# mesh = o3d.io.read_triangle_mesh(object_path)
# # o3d.io.write_triangle_mesh("bottle.ply", mesh)
# o3d.visualization.draw_geometries([mesh])
#
# pcd = mesh.sample_points_uniformly(number_of_points=20000)
# pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
pcd = o3d.io.read_point_cloud("./fingertip_part.pcd")
# o3d.io.write_point_cloud('bottle.pcd',pcd)
save_point = np.array([[0,0,0]])
save_point_x = np.array([])
save_point_y = np.array([])
save_point_z = np.array([])
pcd.paint_uniform_color([0.5, 0.5, 0.5])
#相应修改
# for i in range(10000):
#     # print(pcd.points[i])
#     # print(len(pcd.points))
#     save_point = np.append(save_point, np.array([pcd.points[i]]), axis = 0)
#     save_point_x = np.append(save_point_x, np.array([pcd.points[i][0]])/100)
#     save_point_y = np.append(save_point_y, np.array([pcd.points[i][1]])/100)
#     save_point_z = np.append(save_point_z, np.array([pcd.points[i][2]])/100)
# print("save_point_x:\n", save_point_x)
# print("save_point_y:\n", save_point_y)
# print("save_point_z:\n", save_point_z)
# io.savemat('data_x.mat',{'x': save_point_x})
# io.savemat('data_y.mat',{'y': save_point_y})
# io.savemat('data_z.mat',{'z': save_point_z})


    # print(pcd.points[15])
np.save("output_point.npy", save_point)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

o3d.visualization.draw_geometries([pcd])


# pcd.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=5))
# o3d.visualization.draw_geometries([pcd])

#另一种计算算法
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=5))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024],
#                                   point_show_normal=True)\

# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
# pcd.colors[15] = [1, 0, 0]
# print("pcd.points[15]:", pcd.points[15])
# [k, idx, _] = pcd_tree.search_radius_vector_3d([23.60738, 144.00172, -6.14537], 19)
#
# np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
# print("idx:", idx)
#
# o3d.visualization.draw_geometries([pcd], zoom=0.5599,
#                                   front=[-0.4958, 0.8229, 0.2773],
#                                   lookat=[2.1126, 1.0163, -1.8543],
#                                   up=[0.1007, -0.2626, 0.9596])
#
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5))
# print("6826normals:", pcd.normals[7010])
# print("7025normals:", pcd.normals[7012])
# print("6827normals:", pcd.normals[8257])
# print("7023normals:", pcd.normals[6834])
# print("6635normals:", pcd.normals[8256])
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024],
#                                   point_show_normal=True)
