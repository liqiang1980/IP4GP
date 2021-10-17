import open3d as o3d
import numpy as np
import time, os
from open3d import *
import scipy.io as io


pcd = o3d.io.read_point_cloud("./16line_filtered.pcd")
save_point = np.array([[0,0,0]])
save_point_x = np.array([])
save_point_y = np.array([])
save_point_z = np.array([])
pcd.paint_uniform_color([0.5, 0.5, 0.5])
np.save("output_point.npy", save_point)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

o3d.visualization.draw_geometries([pcd])
