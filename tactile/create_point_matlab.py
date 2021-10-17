import open3d as o3d
import numpy as np
import time, os
from open3d import *
import scipy.io as io

#需要修改成对应的名称
pcd = o3d.io.read_point_cloud("./16line_filtered.pcd")

save_point = np.array([[0,0,0]])
save_point_x = np.array([])
save_point_y = np.array([])
save_point_z = np.array([])
pcd.paint_uniform_color([0.5, 0.5, 0.5])
# #相应修改
#575为点云数量
for i in range(575):
    save_point = np.append(save_point, np.array([pcd.points[i]]), axis = 0)
    save_point_x = np.append(save_point_x, np.array([pcd.points[i][0]]))
    save_point_y = np.append(save_point_y, np.array([pcd.points[i][1]]))
    save_point_z = np.append(save_point_z, np.array([pcd.points[i][2]]))

io.savemat('data_x.mat',{'x': save_point_x})
io.savemat('data_y.mat',{'y': save_point_y})
io.savemat('data_z.mat',{'z': save_point_z})

np.save("output_point.npy", save_point)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
o3d.visualization.draw_geometries([pcd])
