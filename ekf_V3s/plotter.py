import numpy as np
import Plot_plus as pPlt

x_all = np.loadtxt('x_data.txt')
gd_all = np.loadtxt('gd_data.txt')
pPlt.plot_xt_GD_6in1(x_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')
