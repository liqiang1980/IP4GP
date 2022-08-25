import numpy as np
import matplotlib.pyplot as plt

ct_g_z_position = np.loadtxt('ct_g_z_position.txt')
ct_p_z_position = np.loadtxt('ct_p_z_position.txt')
l_ct_position = np.loadtxt('l_ct_position.txt')

ct_g_z_position1 = ct_g_z_position[:, 0]
ct_g_z_position2 = ct_g_z_position[:, 1]
ct_g_z_position3 = ct_g_z_position[:, 2]

ct_p_z_position1 = ct_p_z_position[:, 0]
ct_p_z_position2 = ct_p_z_position[:, 1]
ct_p_z_position3 = ct_p_z_position[:, 2]

l_ct_position1 = l_ct_position[:, 0]
l_ct_position2 = l_ct_position[:, 1]
l_ct_position3 = l_ct_position[:, 2]

t = np.arange(0, ct_p_z_position1.shape[0], 1)

fig, ax = plt.subplots(3, 3, figsize=(12, 6), sharex='all', dpi=240)
fig.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0, 0]
ax1.plot(t, ct_g_z_position1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1, 0]
ax2.plot(t, ct_g_z_position2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2, 0]
ax3.plot(t, ct_g_z_position3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)

ax4 = ax[0, 1]
ax4.plot(t, ct_p_z_position1, color='black')
ax4.set_ylabel('z', {'size': 13})
ax4.set_xlabel('Count', {'size': 13})
ax4.grid(axis="both")
ax4.tick_params(labelsize=13)

ax5 = ax[1, 1]
ax5.plot(t, ct_p_z_position2, color='black')
ax5.set_ylabel('z', {'size': 13})
ax5.set_xlabel('Count', {'size': 13})
ax5.grid(axis="both")
ax5.tick_params(labelsize=13)

ax5 = ax[2, 1]
ax5.plot(t, ct_p_z_position3, color='black')
ax5.set_ylabel('z', {'size': 13})
ax5.set_xlabel('Count', {'size': 13})
ax5.grid(axis="both")
ax5.tick_params(labelsize=13)

ax6 = ax[0, 2]
ax6.plot(t, l_ct_position1, color='black')
ax6.set_ylabel('z', {'size': 13})
ax6.set_xlabel('Count', {'size': 13})
ax6.grid(axis="both")
ax6.tick_params(labelsize=13)

ax7 = ax[1, 2]
ax7.plot(t, l_ct_position2, color='black')
ax7.set_ylabel('z', {'size': 13})
ax7.set_xlabel('Count', {'size': 13})
ax7.grid(axis="both")
ax7.tick_params(labelsize=13)

ax8 = ax[2, 2]
ax8.plot(t, l_ct_position3, color='black')
ax8.set_ylabel('z', {'size': 13})
ax8.set_xlabel('Count', {'size': 13})
ax8.grid(axis="both")
ax8.tick_params(labelsize=13)

plt.show()
