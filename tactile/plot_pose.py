import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import numpy as np

file_dir1 = 'save/save_pose_x_t.npy'  #文件的路径
file_dir2 = 'save/save_pose_GD.npy'
file_dir3 = 'save/save_count_time.npy'
# plydata = PlyData.read(file_dir)  # 读取文件
# data = plydata.elements[0].data  # 读取数据
x_t=np.load(file_dir1)
x_t_x = x_t[2:, 0]*1000
x_t_y = x_t[2:, 1]*1000
x_t_z = x_t[2:, 2]*1000
x_t_r = x_t[2:, 3] * 57.296
x_t_p = x_t[2:, 4] * 57.296
x_t_yaw = x_t[2:, 5] * 57.296
#y坐标轴上点的数值
GD=np.load(file_dir2)
GD_x = GD[2:, 0]*1000
GD_y = GD[2:, 1]*1000
GD_z = GD[2:, 2]*1000
GD_r = GD[2:, 3] * 57.296
GD_p = GD[2:, 4] * 57.296
GD_yaw = GD[2:, 5] * 57.296


t = np.load(file_dir3)
t = t[2:]

print(x_t.shape)
print(GD.shape)
print(t.shape)
plt.figure(1)
plt.plot(t,x_t_x-GD_x, label='x_t_x-GD_x', linewidth=5)
plt.legend(loc = 0, prop = {'size':30})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('x_error[mm]', fontsize=35)
plt.title('x_error_show', fontsize=35)

plt.figure(2)
plt.plot(t,x_t_y-GD_y, label='x_t_y-GD_y', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('y_error[mm]', fontsize=35)
plt.title('y_error_show', fontsize=35)

plt.figure(3)
plt.plot(t,x_t_z-GD_z, label='x_t_z-GD_z', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
# plt.plot(t,y)
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('z_error[mm]', fontsize=35)
plt.title('z_error_show', fontsize=35)

plt.figure(4)
plt.plot(t,x_t_r-GD_r, label='x_t_r-GD_r', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Roll_error[°]', fontsize=35)
plt.title('Roll_error_show', fontsize=35)

plt.figure(5)
plt.plot(t,x_t_p-GD_p, label='x_t_p-GD_p', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Pitch_error[°]', fontsize=35)
plt.title('Pitch_error_show', fontsize=35)

plt.figure(6)
plt.plot(t,x_t_yaw-GD_yaw, label='x_t_yaw-GD_yaw', linewidth=5)
# plt.plot(t,y)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Yaw_error[°]', fontsize=35)
plt.title('Yaw_error_show', fontsize=35)

plt.figure(7)
plt.plot(t,x_t_x, label='x_t_x', linewidth=5)
plt.plot(t,GD_x, label='GD_x', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('x_error[mm]', fontsize=35)
plt.title('x_error_show', fontsize=35)

plt.figure(8)
plt.plot(t,x_t_y, label='x_t_y', linewidth=5)
plt.plot(t,GD_y, label='GD_y', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('y_error[mm]', fontsize=35)
plt.title('y_error_show', fontsize=35)

plt.figure(9)
plt.plot(t,x_t_z, label='x_t_z', linewidth=5)
plt.plot(t,GD_z, label='GD_z', linewidth=5)
# plt.plot(t,y)
# plt.legend(handles=[x_t_z,GD_z],labels=['x_t_z','GD_z'],loc='best')
plt.legend(loc = 0, prop = {'size':35})
# plt.legend([x_t_z, GD_z], ['x_t_z', 'GD_z'])
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('z_error[mm]', fontsize=35)
plt.title('z_error_show', fontsize=35)

plt.figure(10)
plt.plot(t,x_t_r, label='x_t_r', linewidth=5)
plt.plot(t,GD_r, label='GD_r', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Roll_error[°]', fontsize=35)
plt.title('Roll_error_show', fontsize=35)

plt.figure(11)
plt.plot(t,x_t_p, label='x_t_p', linewidth=5)
plt.plot(t,GD_p, label='GD_p', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Pitch_error[°]', fontsize=35)
plt.title('Pitch_error_show', fontsize=35)

plt.figure(12)
plt.plot(t,x_t_yaw, label='x_t_yaw', linewidth=5)
plt.plot(t,GD_yaw, label='GD_yaw', linewidth=5)
plt.legend(loc = 0, prop = {'size':35})
# plt.plot(t,y)
plt.tick_params(axis='both',which='major',labelsize=35)
#设置刻度的字号
plt.xlabel('count', fontsize=35)
plt.ylabel('Yaw_error[°]', fontsize=35)
plt.title('Yaw_error_show', fontsize=35)

# plt.annotate('test_rot', xy=(2,5), xytext=(2, 10),
#             arrowprops=dict(facecolor='black', shrink=0.001),
#             )
plt.show()
