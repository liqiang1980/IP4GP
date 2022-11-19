import numpy as np
import Plot_plus as pPlt

x_state_all = np.loadtxt('ekf_obj_pose/x_state_all.txt')
x_bar_all = np.loadtxt('ekf_obj_pose/x_bar_all.txt')
gd_all = np.loadtxt('ekf_obj_pose/x_gt_palm.txt')
pPlt.plot_xt_GD_6in1(gd_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='roll[deg]', label5='pitch[deg]', label6='yaw[deg]', mode=2, cut=3)
pPlt.plot_xt_GD_6in1(x_state_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='roll[deg]', label5='pitch[deg]', label6='yaw[deg]', mode=2, cut=3)
pPlt.plot_error_6in1(x_state_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='roll[deg]', label5='pitch[deg]', label6='yaw[deg]', mode=2, cut=3)
#
# pPlt.plot_xbar_xsta_GD_6in1(x_bar_all, x_state_all, gd_all,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='roll[deg]', label5='pitch[deg]', label6='yaw[deg]', mode=2, cut=3)
# pPlt.plot_2err_6in1(x_state_all, x_bar_all, gd_all,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='roll[deg]', label5='pitch[deg]', label6='yaw[deg]', mode=2, cut=3)
#

