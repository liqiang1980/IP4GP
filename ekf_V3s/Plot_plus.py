import numpy as np
from matplotlib import pyplot as plt
import qgFunc as qg

#############**********#############**********#############**********#############**********#############**********#####
def plot_error(folder):
    # 文件的路径
    file_dir_error_1 = '/home/lqg/PycharmProjects/ekf_v2/' + folder + '/save_error_xyz.npy'
    file_dir_error_2 = '/home/lqg/PycharmProjects/ekf_v2/' + folder + '/save_error_rpy.npy'
    file_dir_time = '/home/lqg/PycharmProjects/ekf_v2/' + folder + '/save_count_time.npy'
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
def plot_xt_GD(xt_all, GD_all):
    """
    Plot 6 figs to compare xt and GD
    """
    # axis y
    xt_x = xt_all[:, 0] * 1000
    xt_y = xt_all[:, 1] * 1000
    xt_z = xt_all[:, 2] * 1000
    xt_ez = xt_all[:, 3] * 57.3
    xt_ey = xt_all[:, 4] * 57.3
    xt_ex = xt_all[:, 5] * 57.3
    GD_x = GD_all[:, 0] * 1000
    GD_y = GD_all[:, 1] * 1000
    GD_z = GD_all[:, 2] * 1000
    GD_ez = GD_all[:, 3] * 57.3
    GD_ey = GD_all[:, 4] * 57.3
    GD_ex = GD_all[:, 5] * 57.3

    # axis x
    t = np.arange(0, xt_x.shape[0], 1)

    plt.figure(1)
    plt.plot(t, xt_x, label='x_fEKF', linewidth=2)
    plt.plot(t, GD_x, label='GD_x', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('x[mm]', fontsize=15)
    plt.title('x_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(2)
    plt.plot(t, xt_y, label='y_fEKF', linewidth=2)
    plt.plot(t, GD_y, label='GD_y', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('y[mm]', fontsize=15)
    plt.title('y_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(3)
    plt.plot(t, xt_z, label='z_fEKF', linewidth=2)
    plt.plot(t, GD_z, label='GD_z', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('z[mm]', fontsize=15)
    plt.title('z_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(4)
    plt.plot(t, xt_ez, label='ez_fEKF', linewidth=2)
    plt.plot(t, GD_ez, label='GD_ez', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Euler_z[°]', fontsize=15)
    plt.title('Euler z_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(5)
    plt.plot(t, xt_ey, label='ey_fEKF', linewidth=2)
    plt.plot(t, GD_ey, label='GD_ey', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Euler y[°]', fontsize=15)
    plt.title('Euler y_show', fontsize=20)
    plt.grid(axis="both")

    plt.figure(6)
    plt.plot(t, xt_ex, label='ex_fEKF', linewidth=2)
    plt.plot(t, GD_ex, label='GD_ex', linewidth=2)
    plt.legend(loc=0, prop={'size': 13})
    plt.tick_params(axis='both', which='major', labelsize=15)
    # 设置刻度的字号
    plt.xlabel('count', fontsize=15)
    plt.ylabel('Euler x[°]', fontsize=15)
    plt.title('Euler_x_show', fontsize=20)
    plt.grid(axis="both")

    plt.show()


def plot_4_joint_vel(joint_vel_all, label1, label2, label3, label4):
    print("Plot, shape of joint_vel:", joint_vel_all.shape)
    index0 = joint_vel_all[:, 0] * 57.3
    index1 = joint_vel_all[:, 1] * 57.3
    index2 = joint_vel_all[:, 2] * 57.3
    index3 = joint_vel_all[:, 3] * 57.3
    mid0 = joint_vel_all[:, 4] * 57.3
    mid1 = joint_vel_all[:, 5] * 57.3
    mid2 = joint_vel_all[:, 6] * 57.3
    mid3 = joint_vel_all[:, 7] * 57.3
    ring0 = joint_vel_all[:, 8] * 57.3
    ring1 = joint_vel_all[:, 9] * 57.3
    ring2 = joint_vel_all[:, 10] * 57.3
    ring3 = joint_vel_all[:, 11] * 57.3
    thumb0 = joint_vel_all[:, 12] * 57.3
    thumb1 = joint_vel_all[:, 13] * 57.3
    thumb2 = joint_vel_all[:, 14] * 57.3
    thumb3 = joint_vel_all[:, 15] * 57.3
    # index0 = qg.moving_average(index0, 4)
    # index1 = qg.moving_average(index1, 4)
    # index2 = qg.moving_average(index2, 4)
    # index3 = qg.moving_average(index3, 4)
    t = np.arange(0, joint_vel_all.shape[0], 1)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.05, wspace=0.25)

    ax1 = ax[0, 0]
    ax1.plot(t, index0, color='black')
    ax1.plot(t, index1, color='red')
    ax1.plot(t, index2, color='green')
    ax1.plot(t, index3, color='blue')
    ax1.set_ylabel(label1, {'size': 10})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=10)

    ax1 = ax[0, 1]
    ax1.plot(t, mid0, color='black')
    ax1.plot(t, mid1, color='red')
    ax1.plot(t, mid2, color='green')
    ax1.plot(t, mid3, color='blue')
    ax1.set_ylabel(label2, {'size': 10})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=10)

    ax1 = ax[1, 0]
    ax1.plot(t, ring0, color='black')
    ax1.plot(t, ring1, color='red')
    ax1.plot(t, ring2, color='green')
    ax1.plot(t, ring3, color='blue')
    ax1.set_ylabel(label3, {'size': 10})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=10)

    ax1 = ax[1, 1]
    ax1.plot(t, thumb0, color='black')
    ax1.plot(t, thumb1, color='red')
    ax1.plot(t, thumb2, color='green')
    ax1.plot(t, thumb3, color='blue')
    ax1.set_ylabel(label4, {'size': 10})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=10)

    plt.show()



def plot_xt_GD_6in1(Af, GD, label1, label2, label3, label4, label5, label6):
    print("Plot, shape1, shape2:", Af.shape, GD.shape)
    # ALL_FLAG = False
    ALL_FLAG = True
    k = 19
    if ALL_FLAG:
        Af1 = Af[:, 0] * 1000
        Af2 = Af[:, 1] * 1000
        Af3 = Af[:, 2] * 1000
        Af4 = Af[:, 3] * 57.3
        Af5 = Af[:, 4] * 57.3
        Af6 = Af[:, 5] * 57.3
        GD1 = GD[:, 0] * 1000
        GD2 = GD[:, 1] * 1000
        GD3 = GD[:, 2] * 1000
        GD4 = GD[:, 3] * 57.3
        GD5 = GD[:, 4] * 57.3
        GD6 = GD[:, 5] * 57.3
    else:
        Af1 = Af[:k, 0] * 1000
        Af2 = Af[:k, 1] * 1000
        Af3 = Af[:k, 2] * 1000
        Af4 = Af[:k, 3] * 57.3
        Af5 = Af[:k, 4] * 57.3
        Af6 = Af[:k, 5] * 57.3
        GD1 = GD[:k, 0] * 1000
        GD2 = GD[:k, 1] * 1000
        GD3 = GD[:k, 2] * 1000
        GD4 = GD[:k, 3] * 57.3
        GD5 = GD[:k, 4] * 57.3
        GD6 = GD[:k, 5] * 57.3
    t = np.arange(0, GD1.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, Af1, color='red')
    ax1.plot(t, GD1, color='black')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)

    ax2 = ax[1, 0]
    ax2.plot(t, Af2, color='red')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)

    ax3 = ax[2, 0]
    ax3.plot(t, Af3, color='red')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)

    ax4 = ax[0, 1]
    ax4.plot(t, Af4, color='red')
    ax4.plot(t, GD4, color='black')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")

    ax5 = ax[1, 1]
    ax5.plot(t, Af5, color='red')
    ax5.plot(t, GD5, color='black')
    ax5.set_ylabel(label5, {'size': 13})
    ax5.grid(axis="both")

    ax6 = ax[2, 1]
    ax6.plot(t, Af6, color='red')
    ax6.plot(t, GD6, color='black')
    ax6.set_ylabel(label6, {'size': 13})
    ax6.set_xlabel('Count', {'size': 13})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=13)

    plt.show()


def plot_ut(Af, GD, label1, label2, label3, label4, label5, label6):
    print("Plot, shape1, shape2:", Af.shape, GD.shape)
    Af1 = Af[:, 0] * 57.3
    Af2 = Af[:, 1] * 57.3
    Af3 = Af[:, 2] * 57.3
    Af4 = Af[:, 3] * 57.3
    GD1 = GD[:, 0] * 57.3
    GD2 = GD[:, 1] * 57.3
    GD3 = GD[:, 2] * 57.3
    GD4 = GD[:, 3] * 57.3
    t = np.arange(0, GD1.shape[0], 1)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, Af1, color='red')
    ax1.plot(t, GD1, color='black')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)

    ax2 = ax[1, 1]
    ax2.plot(t, Af2, color='red')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)

    ax3 = ax[1, 0]
    ax3.plot(t, Af3, color='red')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)

    ax4 = ax[0, 1]
    ax4.plot(t, Af4, color='red')
    ax4.plot(t, GD4, color='black')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")


    plt.show()

