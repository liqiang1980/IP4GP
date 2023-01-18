import numpy as np
from matplotlib import pyplot as plt

#############**********#############**********#############**********#############**********#############**********#####
def plot_xt_GD_6in1(Af, GD, label1, label2, label3, label4, label5, label6, mode, cut):
    print("Plot, shape1, shape2:", Af.shape, GD.shape)
    Plot_mode = mode  # Mode0: Plot all; Mode1: cut end part; Mode2: cut start part
    early_end = 19
    later_start = cut

    if Plot_mode==0:
        Af1 = Af[1:, 0] * 1000
        Af2 = Af[1:, 1] * 1000
        Af3 = Af[1:, 2] * 1000
        Af4 = Af[1:, 3] * 57.3
        Af5 = Af[1:, 4] * 57.3
        Af6 = Af[1:, 5] * 57.3
        GD1 = GD[1:, 0] * 1000
        GD2 = GD[1:, 1] * 1000
        GD3 = GD[1:, 2] * 1000
        GD4 = GD[1:, 3] * 57.3
        GD5 = GD[1:, 4] * 57.3
        GD6 = GD[1:, 5] * 57.3
    elif Plot_mode==1:
        Af1 = Af[1:early_end, 0] * 1000
        Af2 = Af[1:early_end, 1] * 1000
        Af3 = Af[1:early_end, 2] * 1000
        Af4 = Af[1:early_end, 3] * 57.3
        Af5 = Af[1:early_end, 4] * 57.3
        Af6 = Af[1:early_end, 5] * 57.3
        GD1 = GD[1:early_end, 0] * 1000
        GD2 = GD[1:early_end, 1] * 1000
        GD3 = GD[1:early_end, 2] * 1000
        GD4 = GD[1:early_end, 3] * 57.3
        GD5 = GD[1:early_end, 4] * 57.3
        GD6 = GD[1:early_end, 5] * 57.3
    elif Plot_mode==2:
        Af1 = Af[later_start:, 0] * 1000
        Af2 = Af[later_start:, 1] * 1000
        Af3 = Af[later_start:, 2] * 1000
        Af4 = Af[later_start:, 3] * 57.3
        Af5 = Af[later_start:, 4] * 57.3
        Af6 = Af[later_start:, 5] * 57.3
        GD1 = GD[later_start:, 0] * 1000
        GD2 = GD[later_start:, 1] * 1000
        GD3 = GD[later_start:, 2] * 1000
        GD4 = GD[later_start:, 3] * 57.3
        GD5 = GD[later_start:, 4] * 57.3
        GD6 = GD[later_start:, 5] * 57.3
    t = np.arange(0, GD1.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, Af1, color='red')
    ax1.plot(t, GD1, color='black')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)
    # ax1.set_ylim(-600, 600)

    ax2 = ax[1, 0]
    ax2.plot(t, Af2, color='red')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)
    # ax2.set_ylim(-600, 600)

    ax3 = ax[2, 0]
    ax3.plot(t, Af3, color='red')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)
    # ax3.set_ylim(-600, 600)

    ax4 = ax[0, 1]
    ax4.plot(t, Af4, color='red')
    ax4.plot(t, GD4, color='black')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")
    # ax4.set_ylim(-50, 50)

    ax5 = ax[1, 1]
    ax5.plot(t, Af5, color='red')
    ax5.plot(t, GD5, color='black')
    ax5.set_ylabel(label5, {'size': 13})
    ax5.grid(axis="both")
    # ax5.set_ylim(-200, 200)

    ax6 = ax[2, 1]
    ax6.plot(t, Af6, color='red')
    ax6.plot(t, GD6, color='black')
    ax6.set_ylabel(label6, {'size': 13})
    ax6.set_xlabel('Count', {'size': 13})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=13)
    # ax6.set_ylim(-200, 200)

    plt.show()

def plot_error_6in1(Af, GD, label1, label2, label3, label4, label5, label6, mode, cut):
    print("Plot error!, shape1, shape2:", Af.shape, GD.shape)
    Plot_mode = mode  # Mode0: Plot all; Mode1: cut end part; Mode2: cut start part
    early_end = 19
    later_start = cut

    if Plot_mode==0:
        Af1 = Af[1:, 0] * 1000
        Af2 = Af[1:, 1] * 1000
        Af3 = Af[1:, 2] * 1000
        Af4 = Af[1:, 3] * 57.3
        Af5 = Af[1:, 4] * 57.3
        Af6 = Af[1:, 5] * 57.3
        GD1 = GD[1:, 0] * 1000
        GD2 = GD[1:, 1] * 1000
        GD3 = GD[1:, 2] * 1000
        GD4 = GD[1:, 3] * 57.3
        GD5 = GD[1:, 4] * 57.3
        GD6 = GD[1:, 5] * 57.3
    elif Plot_mode==1:
        Af1 = Af[1:early_end, 0] * 1000
        Af2 = Af[1:early_end, 1] * 1000
        Af3 = Af[1:early_end, 2] * 1000
        Af4 = Af[1:early_end, 3] * 57.3
        Af5 = Af[1:early_end, 4] * 57.3
        Af6 = Af[1:early_end, 5] * 57.3
        GD1 = GD[1:early_end, 0] * 1000
        GD2 = GD[1:early_end, 1] * 1000
        GD3 = GD[1:early_end, 2] * 1000
        GD4 = GD[1:early_end, 3] * 57.3
        GD5 = GD[1:early_end, 4] * 57.3
        GD6 = GD[1:early_end, 5] * 57.3
    elif Plot_mode==2:
        Af1 = Af[later_start:, 0] * 1000
        Af2 = Af[later_start:, 1] * 1000
        Af3 = Af[later_start:, 2] * 1000
        Af4 = Af[later_start:, 3] * 57.3
        Af5 = Af[later_start:, 4] * 57.3
        Af6 = Af[later_start:, 5] * 57.3
        GD1 = GD[later_start:, 0] * 1000
        GD2 = GD[later_start:, 1] * 1000
        GD3 = GD[later_start:, 2] * 1000
        GD4 = GD[later_start:, 3] * 57.3
        GD5 = GD[later_start:, 4] * 57.3
        GD6 = GD[later_start:, 5] * 57.3

    e1 = Af1 - GD1
    e2 = Af2 - GD2
    e3 = Af3 - GD3
    e4 = Af4 - GD4
    e5 = Af5 - GD5
    e6 = Af6 - GD6
    t = np.arange(0, e1.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, e1, color='blue')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)

    ax2 = ax[1, 0]
    ax2.plot(t, e2, color='blue')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)

    ax3 = ax[2, 0]
    ax3.plot(t, e3, color='blue')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)

    ax4 = ax[0, 1]
    ax4.plot(t, e4, color='blue')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")

    ax5 = ax[1, 1]
    ax5.plot(t, e5, color='blue')
    ax5.set_ylabel(label5, {'size': 13})
    ax5.grid(axis="both")

    ax6 = ax[2, 1]
    ax6.plot(t, e6, color='blue')
    ax6.set_ylabel(label6, {'size': 13})
    ax6.set_xlabel('Count', {'size': 13})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=13)

    plt.show()

def plot_xbar_xsta_GD_6in1(Af1, Af2, GD, label1, label2, label3, label4, label5, label6, mode, cut):
    print("Plot, Af1_shape, Af2_shape, GD_shape:", Af1.shape, Af2.shape, GD.shape)
    Plot_mode = mode  # Mode0: Plot all; Mode1: cut end part; Mode2: cut start part
    early_end = 19
    later_start = cut

    if Plot_mode==0:
        Af11 = Af1[1:, 0] * 1000
        Af12 = Af1[1:, 1] * 1000
        Af13 = Af1[1:, 2] * 1000
        Af14 = Af1[1:, 3] * 57.3
        Af15 = Af1[1:, 4] * 57.3
        Af16 = Af1[1:, 5] * 57.3
        Af21 = Af2[1:, 0] * 1000
        Af22 = Af2[1:, 1] * 1000
        Af23 = Af2[1:, 2] * 1000
        Af24 = Af2[1:, 3] * 57.3
        Af25 = Af2[1:, 4] * 57.3
        Af26 = Af2[1:, 5] * 57.3
        GD1 = GD[1:, 0] * 1000
        GD2 = GD[1:, 1] * 1000
        GD3 = GD[1:, 2] * 1000
        GD4 = GD[1:, 3] * 57.3
        GD5 = GD[1:, 4] * 57.3
        GD6 = GD[1:, 5] * 57.3
    elif Plot_mode==1:
        Af11 = Af1[1:early_end, 0] * 1000
        Af12 = Af1[1:early_end, 1] * 1000
        Af13 = Af1[1:early_end, 2] * 1000
        Af14 = Af1[1:early_end, 3] * 57.3
        Af15 = Af1[1:early_end, 4] * 57.3
        Af16 = Af1[1:early_end, 5] * 57.3
        Af21 = Af2[1:early_end, 0] * 1000
        Af22 = Af2[1:early_end, 1] * 1000
        Af23 = Af2[1:early_end, 2] * 1000
        Af24 = Af2[1:early_end, 3] * 57.3
        Af25 = Af2[1:early_end, 4] * 57.3
        Af26 = Af2[1:early_end, 5] * 57.3
        GD1 = GD[1:early_end, 0] * 1000
        GD2 = GD[1:early_end, 1] * 1000
        GD3 = GD[1:early_end, 2] * 1000
        GD4 = GD[1:early_end, 3] * 57.3
        GD5 = GD[1:early_end, 4] * 57.3
        GD6 = GD[1:early_end, 5] * 57.3
    elif Plot_mode==2:
        Af11 = Af1[later_start:, 0] * 1000
        Af12 = Af1[later_start:, 1] * 1000
        Af13 = Af1[later_start:, 2] * 1000
        Af14 = Af1[later_start:, 3] * 57.3
        Af15 = Af1[later_start:, 4] * 57.3
        Af16 = Af1[later_start:, 5] * 57.3
        Af21 = Af2[later_start:, 0] * 1000
        Af22 = Af2[later_start:, 1] * 1000
        Af23 = Af2[later_start:, 2] * 1000
        Af24 = Af2[later_start:, 3] * 57.3
        Af25 = Af2[later_start:, 4] * 57.3
        Af26 = Af2[later_start:, 5] * 57.3
        GD1 = GD[later_start:, 0] * 1000
        GD2 = GD[later_start:, 1] * 1000
        GD3 = GD[later_start:, 2] * 1000
        GD4 = GD[later_start:, 3] * 57.3
        GD5 = GD[later_start:, 4] * 57.3
        GD6 = GD[later_start:, 5] * 57.3
    t = np.arange(0, GD1.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, Af11, color='red')
    ax1.plot(t, Af21, color='green')
    ax1.plot(t, GD1, color='black')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)
    # ax1.set_ylim(-600, 600)

    ax2 = ax[1, 0]
    ax2.plot(t, Af12, color='red')
    ax2.plot(t, Af22, color='green')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)
    # ax2.set_ylim(-600, 600)

    ax3 = ax[2, 0]
    ax3.plot(t, Af13, color='red')
    ax3.plot(t, Af23, color='green')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)
    # ax3.set_ylim(-600, 600)

    ax4 = ax[0, 1]
    ax4.plot(t, Af14, color='red')
    ax4.plot(t, Af24, color='green')
    ax4.plot(t, GD4, color='black')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")
    # ax4.set_ylim(-50, 50)

    ax5 = ax[1, 1]
    ax5.plot(t, Af15, color='red')
    ax5.plot(t, Af25, color='green')
    ax5.plot(t, GD5, color='black')
    ax5.set_ylabel(label5, {'size': 13})
    ax5.grid(axis="both")
    # ax5.set_ylim(-200, 200)

    ax6 = ax[2, 1]
    ax6.plot(t, Af16, color='red')
    ax6.plot(t, Af26, color='green')
    ax6.plot(t, GD6, color='black')
    ax6.set_ylabel(label6, {'size': 13})
    ax6.set_xlabel('Count', {'size': 13})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=13)
    # ax6.set_ylim(-200, 200)

    plt.show()

def plot_2err_6in1(Af1, Af2, GD, label1, label2, label3, label4, label5, label6, mode, cut):
    print("Plot, Af1_shape, Af2_shape, GD_shape:", Af1.shape, Af2.shape, GD.shape)
    Plot_mode = mode  # Mode0: Plot all; Mode1: cut end part; Mode2: cut start part
    early_end = 19
    later_start = cut

    if Plot_mode==0:
        Af11 = Af1[1:, 0] * 1000
        Af12 = Af1[1:, 1] * 1000
        Af13 = Af1[1:, 2] * 1000
        Af14 = Af1[1:, 3] * 57.3
        Af15 = Af1[1:, 4] * 57.3
        Af16 = Af1[1:, 5] * 57.3
        Af21 = Af2[1:, 0] * 1000
        Af22 = Af2[1:, 1] * 1000
        Af23 = Af2[1:, 2] * 1000
        Af24 = Af2[1:, 3] * 57.3
        Af25 = Af2[1:, 4] * 57.3
        Af26 = Af2[1:, 5] * 57.3
        GD1 = GD[1:, 0] * 1000
        GD2 = GD[1:, 1] * 1000
        GD3 = GD[1:, 2] * 1000
        GD4 = GD[1:, 3] * 57.3
        GD5 = GD[1:, 4] * 57.3
        GD6 = GD[1:, 5] * 57.3
    elif Plot_mode==1:
        Af11 = Af1[1:early_end, 0] * 1000
        Af12 = Af1[1:early_end, 1] * 1000
        Af13 = Af1[1:early_end, 2] * 1000
        Af14 = Af1[1:early_end, 3] * 57.3
        Af15 = Af1[1:early_end, 4] * 57.3
        Af16 = Af1[1:early_end, 5] * 57.3
        Af21 = Af2[1:early_end, 0] * 1000
        Af22 = Af2[1:early_end, 1] * 1000
        Af23 = Af2[1:early_end, 2] * 1000
        Af24 = Af2[1:early_end, 3] * 57.3
        Af25 = Af2[1:early_end, 4] * 57.3
        Af26 = Af2[1:early_end, 5] * 57.3
        GD1 = GD[1:early_end, 0] * 1000
        GD2 = GD[1:early_end, 1] * 1000
        GD3 = GD[1:early_end, 2] * 1000
        GD4 = GD[1:early_end, 3] * 57.3
        GD5 = GD[1:early_end, 4] * 57.3
        GD6 = GD[1:early_end, 5] * 57.3
    elif Plot_mode==2:
        Af11 = Af1[later_start:, 0] * 1000
        Af12 = Af1[later_start:, 1] * 1000
        Af13 = Af1[later_start:, 2] * 1000
        Af14 = Af1[later_start:, 3] * 57.3
        Af15 = Af1[later_start:, 4] * 57.3
        Af16 = Af1[later_start:, 5] * 57.3
        Af21 = Af2[later_start:, 0] * 1000
        Af22 = Af2[later_start:, 1] * 1000
        Af23 = Af2[later_start:, 2] * 1000
        Af24 = Af2[later_start:, 3] * 57.3
        Af25 = Af2[later_start:, 4] * 57.3
        Af26 = Af2[later_start:, 5] * 57.3
        GD1 = GD[later_start:, 0] * 1000
        GD2 = GD[later_start:, 1] * 1000
        GD3 = GD[later_start:, 2] * 1000
        GD4 = GD[later_start:, 3] * 57.3
        GD5 = GD[later_start:, 4] * 57.3
        GD6 = GD[later_start:, 5] * 57.3

    e11 = Af11 - GD1
    e12 = Af12 - GD2
    e13 = Af13 - GD3
    e14 = Af14 - GD4
    e15 = Af15 - GD5
    e16 = Af16 - GD6
    e21 = Af21 - GD1
    e22 = Af22 - GD2
    e23 = Af23 - GD3
    e24 = Af24 - GD4
    e25 = Af25 - GD5
    e26 = Af26 - GD6
    t = np.arange(0, e11.shape[0], 1)

    fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)

    ax1 = ax[0, 0]
    ax1.plot(t, e11, color='blue')
    ax1.plot(t, e21, color='pink')
    ax1.set_ylabel(label1, {'size': 13})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=13)

    ax2 = ax[1, 0]
    ax2.plot(t, e12, color='blue')
    ax2.plot(t, e22, color='pink')
    ax2.set_ylabel(label2, {'size': 13})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=13)

    ax3 = ax[2, 0]
    ax3.plot(t, e13, color='blue')
    ax3.plot(t, e23, color='pink')
    ax3.set_ylabel(label3, {'size': 13})
    ax3.set_xlabel('Count', {'size': 13})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=13)

    ax4 = ax[0, 1]
    ax4.plot(t, e14, color='blue')
    ax4.plot(t, e24, color='pink')
    ax4.set_ylabel(label4, {'size': 13})
    ax4.grid(axis="both")

    ax5 = ax[1, 1]
    ax5.plot(t, e15, color='blue')
    ax5.plot(t, e25, color='pink')
    ax5.set_ylabel(label5, {'size': 13})
    ax5.grid(axis="both")

    ax6 = ax[2, 1]
    ax6.plot(t, e16, color='blue')
    ax6.plot(t, e26, color='pink')
    ax6.set_ylabel(label6, {'size': 13})
    ax6.set_xlabel('Count', {'size': 13})
    ax6.grid(axis="both")
    ax6.tick_params(labelsize=13)

    plt.show()

def plot_all_BAND2(mean1, std1, mean2, std2, mean3, std3,
                  GD1, GD2, GD3,
                  label1, label2, label3):
    t = np.arange(0, mean1.shape[0], 1)

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex='all', sharey='row', dpi=240)

    ax1 = ax[0]
    ax1.fill_between(t, mean1 + 3 * std1, mean1 - 3 * std1, facecolor='green', alpha=0.3, label='3$\sigma$ uncertainty area')
    ax1.plot(t, mean1, color='red', label='Posterior mean values of pose trajectory')
    # ax1.plot(t, mean4, color='blue', linestyle='--', label='Prior mean values of predict state')
    ax1.plot(t, GD1, color='black', label='Ground truth')
    ax1.set_ylabel(label1, {'size': 20})
    ax1.grid(axis="both")
    ax1.tick_params(labelsize=20)
    ax1.margins(x=0)
    # ax1.legend(bbox_to_anchor=(0.285, 1.005), loc=3, borderaxespad=0, prop={'size': 15})

    ax2 = ax[1]
    ax2.fill_between(t, mean2 + 3 * std2, mean2 - 3 * std2, facecolor='green', alpha=0.3)
    ax2.plot(t, mean2, color='red')
    # ax2.plot(t, mean5, color='blue', linestyle='--')
    ax2.plot(t, GD2, color='black')
    ax2.set_ylabel(label2, {'size': 20})
    ax2.grid(axis="both")
    ax2.tick_params(labelsize=20)
    ax2.legend()

    ax3 = ax[2]
    ax3.fill_between(t, mean3 + 3 * std3, mean3 - 3 * std3, facecolor='green', alpha=0.3)
    ax3.plot(t, mean3, color='red')
    # ax3.plot(t, mean6, color='blue', linestyle='--')
    ax3.plot(t, GD3, color='black')
    ax3.set_ylabel(label3, {'size': 20})
    ax3.set_xlabel('Steps', {'size': 20})
    ax3.grid(axis="both")
    ax3.tick_params(labelsize=20)

    plt.show()

