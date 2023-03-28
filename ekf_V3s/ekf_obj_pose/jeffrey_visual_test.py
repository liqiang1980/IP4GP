import numpy as np
import matplotlib.pyplot as plt

# Load data from text files
x_bar_all = np.loadtxt('offline_x_bar_all.txt')
x_state_all = np.loadtxt('offline_x_state_all.txt')
gd_all = np.loadtxt('offline_x_gt_palm.txt')

# Plot the results starting from the second line
plt.figure(figsize=(10, 20))

# Define labels for each subplot
labels = ['x position', 'y position', 'z position', 'x rotation', 'y rotation', 'z rotation', 'w']

# Plot x, y, and z positions
for i in range(1, 7):
    plt.subplot(7, 1, i)
    plt.plot(x_bar_all[1:, i], color='y', alpha=1,label='x_bar')
    plt.plot(x_state_all[1:, i], color='r', alpha=0.5,label='predict')
    plt.plot(gd_all[1:, i], color='b', alpha=0.5,label='GT')
    plt.legend(loc='upper left')
    plt.ylabel(labels[i-1])

plt.tight_layout()
plt.show()
