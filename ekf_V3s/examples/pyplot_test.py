import numpy as np
import matplotlib.pyplot as plt

ct_g_z_position1 = np.loadtxt('delta_p_save0.txt')
ct_g_z_position2 = np.loadtxt('delta_p_save1.txt')
ct_g_z_position3 = np.loadtxt('delta_p_save2.txt')


delta_p = np.loadtxt('delta_p_save.txt')
twist = np.loadtxt('twist.txt')
jnt = np.loadtxt('jnt.txt')
jnt_dot = np.loadtxt('jnt_dot.txt')


delta_p1 = delta_p[1:, 0]
delta_p2 = delta_p[1:, 1]
delta_p3 = delta_p[1:, 2]

twist1 = twist[1:, 0]
twist2 = twist[1:, 1]
twist3 = twist[1:, 2]
twist4 = twist[1:, 3]
twist5 = twist[1:, 4]
twist6 = twist[1:, 5]

jnt1 = jnt[1:, 0]
jnt2 = jnt[1:, 1]
jnt3 = jnt[1:, 2]
jnt4 = jnt[1:, 3]

jnt_dot1 = jnt_dot[1:, 0]
jnt_dot2 = jnt_dot[1:, 1]
jnt_dot3 = jnt_dot[1:, 2]
jnt_dot4 = jnt_dot[1:, 3]

t = np.arange(0, delta_p1.shape[0], 1)


fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex='all', dpi=240)
fig.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0]
ax1.plot(t, delta_p1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1]
ax2.plot(t, delta_p2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2]
ax3.plot(t, delta_p3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)



fig2, ax = plt.subplots(4, 1, figsize=(12, 6), sharex='all', dpi=240)
fig2.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0]
ax1.plot(t, jnt1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1]
ax2.plot(t, jnt2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2]
ax3.plot(t, jnt3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)

ax4 = ax[3]
ax4.plot(t, jnt4, color='black')
ax4.set_ylabel('z', {'size': 13})
ax4.set_xlabel('Count', {'size': 13})
ax4.grid(axis="both")
ax4.tick_params(labelsize=13)

fig3, ax = plt.subplots(4, 1, figsize=(12, 6), sharex='all', dpi=240)
fig3.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0]
ax1.plot(t, jnt_dot1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1]
ax2.plot(t, jnt_dot2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2]
ax3.plot(t, jnt_dot3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)

ax4 = ax[3]
ax4.plot(t, jnt_dot4, color='black')
ax4.set_ylabel('z', {'size': 13})
ax4.set_xlabel('Count', {'size': 13})
ax4.grid(axis="both")
ax4.tick_params(labelsize=13)


fig4, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
fig4.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0, 0]
ax1.plot(t, twist1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1, 0]
ax2.plot(t, twist2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2, 0]
ax3.plot(t, twist3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)


ax4 = ax[0, 1]
ax4.plot(t, twist4, color='black')
ax4.set_ylabel('x', {'size': 13})
ax4.grid(axis="both")
ax4.tick_params(labelsize=13)

ax5 = ax[1, 1]
ax5.plot(t, twist5, color='black')
ax5.set_ylabel('y', {'size': 13})
ax5.grid(axis="both")
ax5.tick_params(labelsize=13)

ax6 = ax[2, 1]
ax6.plot(t, twist6, color='black')
ax6.set_ylabel('z', {'size': 13})
ax6.set_xlabel('Count', {'size': 13})
ax6.grid(axis="both")
ax6.tick_params(labelsize=13)


plt.show()
