import numpy as np
import matplotlib.pyplot as plt

ju = np.loadtxt('ju_all.txt')

ju1 = ju[:, 0]
ju2 = ju[:, 1]
ju3 = ju[:, 2]

ju4 = ju[:, 3]
ju5 = ju[:, 4]
ju6 = ju[:, 5]


t = np.arange(0, ju1.shape[0], 1)

fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all', dpi=240)
fig.subplots_adjust(hspace=0.1, wspace=0.2)

ax1 = ax[0, 0]
ax1.plot(t, ju1, color='black')
ax1.set_ylabel('x', {'size': 13})
ax1.grid(axis="both")
ax1.tick_params(labelsize=13)

ax2 = ax[1, 0]
ax2.plot(t, ju2, color='black')
ax2.set_ylabel('y', {'size': 13})
ax2.grid(axis="both")
ax2.tick_params(labelsize=13)

ax3 = ax[2, 0]
ax3.plot(t, ju3, color='black')
ax3.set_ylabel('z', {'size': 13})
ax3.set_xlabel('Count', {'size': 13})
ax3.grid(axis="both")
ax3.tick_params(labelsize=13)

ax4 = ax[0, 1]
ax4.plot(t, ju4, color='black')
ax4.set_ylabel('z', {'size': 13})
ax4.set_xlabel('Count', {'size': 13})
ax4.grid(axis="both")
ax4.tick_params(labelsize=13)

ax5 = ax[1, 1]
ax5.plot(t, ju5, color='black')
ax5.set_ylabel('z', {'size': 13})
ax5.set_xlabel('Count', {'size': 13})
ax5.grid(axis="both")
ax5.tick_params(labelsize=13)

ax6 = ax[2, 1]
ax6.plot(t, ju6, color='black')
ax6.set_ylabel('z', {'size': 13})
ax6.set_xlabel('Count', {'size': 13})
ax6.grid(axis="both")
ax6.tick_params(labelsize=13)
plt.show()
