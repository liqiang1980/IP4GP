#!/usr/bin/env python3  

import open3d as o3d
import numpy as np
import time, os
from open3d import *
import scipy.io as io

object_path = '../../UR5/mesh/visual/cup_1.STL'

mesh = o3d.io.read_triangle_mesh(object_path)
o3d.visualization.draw_geometries([mesh])

pcd = mesh.sample_points_uniformly(number_of_points=5000)
pcd = mesh.sample_points_poisson_disk(number_of_points=2500, pcl=pcd)

o3d.io.write_point_cloud('../model/cup_1.pcd',pcd)
save_point = np.array([[0,0,0]])
save_point_x = np.array([])
save_point_y = np.array([])
save_point_z = np.array([])
pcd.paint_uniform_color([0.5, 0.5, 0.5])

np.save("output_point.npy", save_point)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

o3d.visualization.draw_geometries([pcd])
