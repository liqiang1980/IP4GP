import numpy as np
import matplotlib.pyplot as plt

x_bar_all = np.loadtxt('x_bar_all.txt')
x_state_all = np.loadtxt('x_state_all.txt')
x_gt_plam = np.loadtxt('x_gt_palm.txt')

x_bar_all1 = x_bar_all[3:, 0]
x_bar_all2 = x_bar_all[3:, 1]
x_bar_all3 = x_bar_all[3:, 2]
x_bar_all4 = x_bar_all[3:, 3]
x_bar_all5 = x_bar_all[3:, 4]
x_bar_all6 = x_bar_all[3:, 5]

x_state_all1 = x_state_all[3:, 0]
x_state_all2 = x_state_all[3:, 1]
x_state_all3 = x_state_all[3:, 2]
x_state_all4 = x_state_all[3:, 3]
x_state_all5 = x_state_all[3:, 4]
x_state_all6 = x_state_all[3:, 5]

x_gt_plam1 = x_gt_plam[3:, 0]
x_gt_plam2 = x_gt_plam[3:, 1]
x_gt_plam3 = x_gt_plam[3:, 2]
x_gt_plam4 = x_gt_plam[3:, 3]
x_gt_plam5 = x_gt_plam[3:, 4]
x_gt_plam6 = x_gt_plam[3:, 5]


t = np.arange(0, x_gt_plam1.shape[0], 1)

fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=100)
fig.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0,0]
ax1.plot(t, x_bar_all1, color='black')
ax1.plot(t, x_state_all1, color='blue')
ax1.plot(t, x_gt_plam1, color='red')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1,0]
ax2.plot(t, x_bar_all2, color='black')
ax2.plot(t, x_state_all2, color='blue')
ax2.plot(t, x_gt_plam2, color='red')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)
#ax2.set_ylim([-0.05, 0.01])

ax3 = ax[2,0]
ax3.plot(t, x_bar_all3, color='black')
ax3.plot(t, x_state_all3, color='blue')
ax3.plot(t, x_gt_plam3, color='red')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)

ax1 = ax[0,1]
ax1.plot(t, x_bar_all4, color='black')
ax1.plot(t, x_state_all4, color='blue')
ax1.plot(t, x_gt_plam4, color='red')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1,1]
ax2.plot(t, x_bar_all5, color='black')
ax2.plot(t, x_state_all5, color='blue')
ax2.plot(t, x_gt_plam5, color='red')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2,1]
ax3.plot(t, x_bar_all6, color='black')
ax3.plot(t, x_state_all6, color='blue')
ax3.plot(t, x_gt_plam6, color='red')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)


plt.show()
