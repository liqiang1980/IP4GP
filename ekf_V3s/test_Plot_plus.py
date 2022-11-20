import numpy as np
from matplotlib import pyplot as plt


#############**********#############**********#############**********#############**********#############**********#####
def plot_error(folder):
    # 文件的路径
    file_dir_error_1 = './' + folder + '/save_error_xyz.npy'
    file_dir_error_2 = './' + folder + '/save_error_rpy.npy'
    file_dir_time = './' + folder + '/save_count_time.npy'
    # plydata = PlyData.read(file_dir)  # 读取文件
    # data = plydata.elements[0].data  # 读取数据

    error_xyz = np.load(file_dir_error_1)  # xyz of Error
    print("Shape_error:", error_xyz.shape)
    error_x = error_xyz[2:, 0] * 1000
    error_y = error_xyz[2:, 1] * 1000
    error_z = error_xyz[2:, 2] * 1000
    error_rpy = np.load(file_dir_error_2)  # rpy of Error
    error_row = error_rpy[2:, 0] * 57.296
    error_pitch = error_rpy[2:, 1] * 57.296
    error_yaw = error_rpy[2:, 2] * 57.296

    # axis x
    t = np.load(file_dir_time)
    t = t[2:]

    # Plot error
    plt.figure(13, dpi=240)
    plt.plot(t, error_x, label='x', linewidth=1.5)
    plt.plot(t, error_y, label='y', linewidth=1.5)
    plt.plot(t, error_z, label='z', linewidth=1.5)
    plt.legend(loc=2, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('error[mm]', fontsize=15)
    plt.title('error_xyz', fontsize=20)
    plt.grid(axis="both")
    # plt.xticks(np.linspace(0, 15, 1), np.linspace(0, 15, 1),  fontsize=15)

    plt.figure(14, dpi=240)
    plt.plot(t, error_row, label='row', linewidth=1.5)
    plt.plot(t, error_pitch, label='pitch', linewidth=1.5)
    plt.plot(t, error_yaw, label='yaw', linewidth=1.5)
    plt.legend(loc=2, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    # plt.xlabel('time[s]', fontsize=15)
    plt.ylabel('error[deg]', fontsize=15)
    plt.title('error_rpy', fontsize=20)
    plt.grid(axis="both")
    # plt.xticks(np.linspace(0, 15, 1), np.linspace(0, 15, 1), fontsize=15)

    plt.show()


#############**********#############**********#############**********#############**********#############**********#####
def plot_all():
    # 文件的路径
    file_dir_y_1 = './save_f/save_pose_y_t_xyz.npy'
    file_dir_y_2 = './save_f/save_pose_y_t_rpy.npy'
    file_dir_yi_1 = './save_i/save_pose_y_t_xyz.npy'
    file_dir_yi_2 = './save_i/save_pose_y_t_rpy.npy'
    file_dir_GD_1 = './save_f/save_pose_GD_xyz.npy'
    file_dir_GD_2 = './save_f/save_pose_GD_rpy.npy'
    file_dir_time = './save_i/save_count_time.npy'

    # axis y
    y_t_xyz = np.load(file_dir_y_1)  # xyz of Estimate
    y_t_x = y_t_xyz[2:, 0] * 1000
    y_t_y = y_t_xyz[2:, 1] * 1000
    y_t_z = y_t_xyz[2:, 2] * 1000
    yi_t_xyz = np.load(file_dir_yi_1)  # xyz of Estimate
    yi_t_x = yi_t_xyz[2:, 0] * 1000
    yi_t_y = yi_t_xyz[2:, 1] * 1000
    yi_t_z = yi_t_xyz[2:, 2] * 1000
    y_t_rpy = np.load(file_dir_y_2)  # rpy of Estimate
    y_t_row = y_t_rpy[2:, 0] * 57.296  # 1 (rad) = 180/pi (degree)
    y_t_pitch = y_t_rpy[2:, 1] * 57.296
    y_t_yaw = y_t_rpy[2:, 2] * 57.296
    yi_t_rpy = np.load(file_dir_yi_2)  # rpy of Estimate
    yi_t_row = yi_t_rpy[2:, 0] * 57.296  # 1 (rad) = 180/pi (degree)
    yi_t_pitch = yi_t_rpy[2:, 1] * 57.296
    yi_t_yaw = yi_t_rpy[2:, 2] * 57.296
    GD_xyz = np.load(file_dir_GD_1)  # xyz of Truth
    GD_x = GD_xyz[2:, 0] * 1000
    GD_y = GD_xyz[2:, 1] * 1000
    GD_z = GD_xyz[2:, 2] * 1000
    GD_rpy = np.load(file_dir_GD_2)  # rpy of Truth
    GD_row = GD_rpy[2:, 0] * 57.296
    GD_pitch = GD_rpy[2:, 1] * 57.296
    GD_yaw = GD_rpy[2:, 2] * 57.296

    # axis x
    # t = np.load(file_dir_time)
    # t = t[2:]
    t = np.arange(0, y_t_x.shape[0], 1)

    plt.figure(7)
    plt.plot(t, y_t_x, label='x_fEKF', linewidth=2)
    plt.plot(t, yi_t_x, label='x_iEKF', linewidth=2)
    plt.plot(t, GD_x, label='GD_x', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('x[mm]', fontsize=15)
    plt.title('x_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(8)
    plt.plot(t, y_t_y, label='y_fEKF', linewidth=2)
    plt.plot(t, yi_t_y, label='y_iEKF', linewidth=2)
    plt.plot(t, GD_y, label='GD_y', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('y[mm]', fontsize=15)
    plt.title('y_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(9)
    plt.plot(t, y_t_z, label='z_fEKF', linewidth=2)
    plt.plot(t, yi_t_z, label='z_iEKF', linewidth=2)
    plt.plot(t, GD_z, label='GD_z', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('z[mm]', fontsize=15)
    plt.title('z_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(10)
    plt.plot(t, y_t_row, label='row_fEKF', linewidth=2)
    plt.plot(t, yi_t_row, label='row_iEKF', linewidth=2)
    plt.plot(t, GD_row, label='GD_row', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Roll[°]', fontsize=15)
    plt.title('Roll_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(11)
    plt.plot(t, y_t_pitch, label='pitch_fEKF', linewidth=2)
    plt.plot(t, yi_t_pitch, label='pitch_iEKF', linewidth=2)
    plt.plot(t, GD_pitch, label='GD_pitch', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Pitch[°]', fontsize=15)
    plt.title('Pitch_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(12)
    plt.plot(t, y_t_yaw, label='yaw_fEKF', linewidth=2)
    plt.plot(t, yi_t_yaw, label='yaw_iEKF', linewidth=2)
    plt.plot(t, GD_yaw, label='GD_yaw', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Yaw[°]', fontsize=15)
    plt.title('Yaw_show', fontsize=20)
    plt.grid(axis="both")

    plt.show()


def plot_all1():
    # 文件的路径
    file_dir_y_1 = './save_f/save_pose_y_t_xyz.npy'
    file_dir_y_2 = './save_f/save_pose_y_t_rpy.npy'
    file_dir_yi_1 = './save_i/save_pose_y_t_xyz.npy'
    file_dir_yi_2 = './save_i/save_pose_y_t_rpy.npy'
    file_dir_GD_1 = './save_f/save_pose_GD_xyz.npy'
    file_dir_GD_2 = './save_f/save_pose_GD_rpy.npy'
    file_dir_time = './save_f/save_count_time.npy'

    # axis y
    y_t_xyz = np.load(file_dir_y_1)  # xyz of Estimate
    # print("y_t_xyz:", y_t_xyz)
    y_t_x = y_t_xyz[2:, 0] * 1000
    y_t_y = y_t_xyz[2:, 1] * 1000
    y_t_z = y_t_xyz[2:, 2] * 1000
    yi_t_xyz = np.load(file_dir_yi_1)  # xyz of Estimate
    print("y_t_xyz:", y_t_xyz.shape)
    yi_t_x = yi_t_xyz[2:, 0] * 1000
    yi_t_y = yi_t_xyz[2:, 1] * 1000
    yi_t_z = yi_t_xyz[2:, 2] * 1000
    y_t_rpy = np.load(file_dir_y_2)  # rpy of Estimate
    y_t_row = y_t_rpy[2:, 0] * 57.296  # 1 (rad) = 180/pi (degree)
    y_t_pitch = y_t_rpy[2:, 1] * 57.296
    y_t_yaw = y_t_rpy[2:, 2] * 57.296
    yi_t_rpy = np.load(file_dir_yi_2)  # rpy of Estimate
    yi_t_row = yi_t_rpy[2:, 0] * 57.296  # 1 (rad) = 180/pi (degree)
    yi_t_pitch = yi_t_rpy[2:, 1] * 57.296
    yi_t_yaw = yi_t_rpy[2:, 2] * 57.296
    GD_xyz = np.load(file_dir_GD_1)  # xyz of Truth
    GD_x = GD_xyz[2:, 0] * 1000
    GD_y = GD_xyz[2:, 1] * 1000
    GD_z = GD_xyz[2:, 2] * 1000
    GD_rpy = np.load(file_dir_GD_2)  # rpy of Truth
    GD_row = GD_rpy[2:, 0] * 57.296
    GD_pitch = GD_rpy[2:, 1] * 57.296
    GD_yaw = GD_rpy[2:, 2] * 57.296

    # axis x
    # t = np.load(file_dir_time)
    t = np.arange(0, y_t_x.shape[0], 1)

    plt.figure(7, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_x, label='x_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_x.T, label='x_iEKF', linewidth=1.5)
    plt.plot(t, GD_x, label='GD_x', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('x[mm]', fontsize=15)
    plt.title('X', fontsize=20)
    plt.grid(axis="both")

    plt.figure(8, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_y, label='y_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_y, label='y_iEKF', linewidth=1.5)
    plt.plot(t, GD_y, label='GD_y', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('y[mm]', fontsize=15)
    plt.title('Y', fontsize=20)
    plt.grid(axis="both")

    plt.figure(9, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_z, label='z_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_z, label='z_iEKF', linewidth=1.5)
    plt.plot(t, GD_z, label='GD_z', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('z[mm]', fontsize=15)
    plt.title('Z', fontsize=20)
    plt.grid(axis="both")

    plt.figure(10, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_row, label='row_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_row, label='row_iEKF', linewidth=1.5)
    plt.plot(t, GD_row, label='GD_row', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Row[deg]', fontsize=15)
    plt.title('Row', fontsize=20)
    plt.grid(axis="both")

    plt.figure(11, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_pitch, label='pitch_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_pitch, label='pitch_iEKF', linewidth=1.5)
    plt.plot(t, GD_pitch, label='GD_pitch', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Pitch[deg]', fontsize=15)
    plt.title('Pitch', fontsize=20)
    plt.grid(axis="both")

    plt.figure(12, dpi=240, figsize=(10, 5))
    plt.plot(t, y_t_yaw, label='yaw_fEKF', linewidth=1.5)
    plt.plot(t, yi_t_yaw, label='yaw_iEKF', linewidth=1.5)
    plt.plot(t, GD_yaw, label='GD_yaw', linewidth=1.5)
    # plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Yaw[deg]', fontsize=15)
    plt.title('Yaw', fontsize=20)
    plt.grid(axis="both")

    plt.show()


def plot_all_band(mean1, std1, mean2, std2, mean3, std3, GD1, GD2, GD3, label1, label2, label3):
    t = np.arange(0, mean1.shape[0], 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True, dpi=240)

    ax1.fill_between(t, mean1 + 3 * std1, mean1 - 3 * std1, facecolor='green', alpha=0.3)
    ax1.plot(t, mean1)
    ax1.plot(t, GD1)
    ax1.set_ylabel(label1)

    ax2.fill_between(t, mean2 + 3 * std2, mean2 - 3 * std2, facecolor='green', alpha=0.3)
    ax2.plot(t, mean2)
    ax2.plot(t, GD2)
    ax2.set_ylabel(label2)

    ax3.fill_between(t, mean3 + 3 * std3, mean3 - 3 * std3, facecolor='green', alpha=0.3)
    ax3.plot(t, mean3)
    ax3.plot(t, GD3)
    ax3.set_ylabel(label3)
    ax3.set_xlabel('Times [ms]')

    plt.grid(axis="both")
    plt.show()


def plot_all_BAND(mean1, std1, mean2, std2, mean3, std3,
                  mean4, std4, mean5, std5, mean6, std6,
                  GD1, GD2, GD3, Af1, Af2, Af3, Ai1, Ai2, Ai3,
                  label1, label2, label3):
    t = np.arange(0, mean1.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', sharey='row', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0)

    ax1 = ax[0, 0]
    ax1.fill_between(t, mean1 + 3 * std1, mean1 - 3 * std1, facecolor='green', alpha=0.3)
    ax1.plot(t, Af1, color='red')
    ax1.plot(t, GD1, color='black')
    ax1.set_ylabel(label1, {'size': 17})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=17)

    ax2 = ax[1, 0]
    ax2.fill_between(t, mean2 + 3 * std2, mean2 - 3 * std2, facecolor='green', alpha=0.3)
    ax2.plot(t, Af2, color='red')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 17})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=17)

    ax3 = ax[2, 0]
    ax3.fill_between(t, mean3 + 3 * std3, mean3 - 3 * std3, facecolor='green', alpha=0.3)
    ax3.plot(t, Af3, color='red')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 17})
    ax3.set_xlabel('Times [ms]', {'size': 17})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=17)

    ax4 = ax[0, 1]
    ax4.fill_between(t, mean4 + 3 * std4, mean4 - 3 * std4, facecolor='green', alpha=0.3)
    ax4.plot(t, Ai1, color='red')
    ax4.plot(t, GD1, color='black')
    ax4.grid(axis="both")

    ax5 = ax[1, 1]
    ax5.fill_between(t, mean5 + 3 * std5, mean5 - 3 * std5, facecolor='green', alpha=0.3)
    ax5.plot(t, Ai2, color='red')
    ax5.plot(t, GD2, color='black')
    ax5.grid(axis="both")

    ax6 = ax[2, 1]
    ax6.fill_between(t, mean6 + 3 * std6, mean6 - 3 * std6, facecolor='green', alpha=0.3)
    ax6.plot(t, Ai3, color='red')
    ax6.plot(t, GD3, color='black')
    ax6.set_xlabel('Times [ms]', {'size': 17})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=17)

    plt.show()


def plot_all_BAND2(mean1, std1, mean2, std2, mean3, std3,
                  mean4, mean5, mean6,
                  GD1, GD2, GD3,
                  label1, label2, label3):
    t = np.arange(0, mean1.shape[0], 1)

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex='all', sharey='row', dpi=240)

    ax1 = ax[0]
    ax1.fill_between(t, mean1 + 3 * std1, mean1 - 3 * std1, facecolor='green', alpha=0.3, label='3$\sigma$ uncertainty area')
    ax1.plot(t, mean1, color='red', label='Posterior mean values of pose trajectory')
    ax1.plot(t, mean4, color='blue', linestyle='--', label='Prior mean values of predict state')
    ax1.plot(t, GD1, color='black', label='Ground truth')
    ax1.set_ylabel(label1, {'size': 20})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=20)
    ax1.margins(x=0)
    ax1.legend(bbox_to_anchor=(0.285, 1.005), loc=3, borderaxespad=0, prop={'size': 15})

    ax2 = ax[1]
    ax2.fill_between(t, mean2 + 3 * std2, mean2 - 3 * std2, facecolor='green', alpha=0.3)
    ax2.plot(t, mean2, color='red')
    ax2.plot(t, mean5, color='blue', linestyle='--')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 20})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=20)
    ax2.legend()

    ax3 = ax[2]
    ax3.fill_between(t, mean3 + 3 * std3, mean3 - 3 * std3, facecolor='green', alpha=0.3)
    ax3.plot(t, mean3, color='red')
    ax3.plot(t, mean6, color='blue', linestyle='--')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 20})
    ax3.set_xlabel('Steps', {'size': 20})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=20)

    plt.show()


def plot_all_BAND_once(mean1, mean2, mean3,
                  mean4, mean5, mean6,
                  GD1, GD2, GD3,
                  label1, label2, label3):
    t = np.arange(0, mean1.shape[0], 1)

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex='all', sharey='row', dpi=240)

    ax1 = ax[0]
    ax1.plot(t, mean1, color='red', label='Posterior value of pose trajectory')
    ax1.plot(t, mean4, color='blue', linestyle='--', label='Prior value of predict state')
    ax1.plot(t, GD1, color='black', label='Ground truth')
    ax1.set_ylabel(label1, {'size': 20})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=20)
    ax1.margins(x=0)
    ax1.legend(bbox_to_anchor=(0.285, 1.005), loc=3, borderaxespad=0, prop={'size': 15})

    ax2 = ax[1]
    ax2.plot(t, mean2, color='red')
    ax2.plot(t, mean5, color='blue', linestyle='--')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 20})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=20)
    ax2.legend()

    ax3 = ax[2]
    ax3.plot(t, mean3, color='red')
    ax3.plot(t, mean6, color='blue', linestyle='--')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 20})
    ax3.set_xlabel('Steps', {'size': 20})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=20)

    plt.show()


