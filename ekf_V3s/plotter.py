import numpy as np
import Plot_plus as pPlt

x_state_all = np.loadtxt('ekf_obj_pose/x_state_all.txt')
x_bar_all = np.loadtxt('ekf_obj_pose/x_bar_all.txt')
gd_all = np.loadtxt('ekf_obj_pose/x_gt_palm.txt')

offset = [20, 20, 20, 10, 10, 10]
offset_ct = 0
pPlt.plot_xt_GD_6in1(gd_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='theta_x[deg]', label5='theta_y[rad]', label6='theta_z[rad]', mode=0, cut=3,
                     offset=offset, offset_ct=offset_ct)
pPlt.plot_xt_GD_6in1(x_state_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='theta_x[rad]', label5='theta_y[rad]', label6='theta_z[rad]', mode=0, cut=3,
                     offset=offset, offset_ct=offset_ct)
pPlt.plot_error_6in1(x_state_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='theta_x[rad]', label5='theta_y[rad]', label6='theta_z[rad]', mode=0, cut=3)

# pPlt.plot_xbar_xsta_GD_6in1(x_bar_all, x_state_all, gd_all,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='theta_x[rad]', label5='theta_y[rad]', label6='theta_z[rad]', mode=2, cut=3)
# pPlt.plot_2err_6in1(x_state_all, x_bar_all, gd_all,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='theta_x[rad]', label5='theta_y[rad]', label6='theta_z[rad]', mode=2, cut=3)

Af1 = x_state_all[1:, 0] * 1000
Af2 = x_state_all[1:, 1] * 1000
Af3 = x_state_all[1:, 2] * 1000
Af4 = x_state_all[1:, 3] * 57.3
Af5 = x_state_all[1:, 4] * 57.3
Af6 = x_state_all[1:, 5] * 57.3
GD1 = gd_all[1:, 0] * 1000
GD2 = gd_all[1:, 1] * 1000
GD3 = gd_all[1:, 2] * 1000
GD4 = gd_all[1:, 3] * 57.3
GD5 = gd_all[1:, 4] * 57.3
GD6 = gd_all[1:, 5] * 57.3
dis1 = np.abs(Af1 - GD1)
dis2 = np.abs(Af2 - GD2)
dis3 = np.abs(Af3 - GD3)
dis4 = np.abs(Af4 - GD4)
dis5 = np.abs(Af5 - GD5)
dis6 = np.abs(Af6 - GD6)

ME1 = np.max(dis1)
ME2 = np.max(dis2)
ME3 = np.max(dis3)
ME4 = np.max(dis4)
ME5 = np.max(dis5)
ME6 = np.max(dis6)

MAE1 = np.mean(dis1)
MAE2 = np.mean(dis2)
MAE3 = np.mean(dis3)
MAE4 = np.mean(dis4)
MAE5 = np.mean(dis5)
MAE6 = np.mean(dis6)

_RMSE1 = 0
_RMSE2 = 0
_RMSE3 = 0
_RMSE4 = 0
_RMSE5 = 0
_RMSE6 = 0
for i in range(dis1.shape[0]):
    _RMSE1 += dis1[i]**2
    _RMSE2 += dis2[i]**2
    _RMSE3 += dis3[i]**2
    _RMSE4 += dis4[i]**2
    _RMSE5 += dis5[i]**2
    _RMSE6 += dis6[i]**2
RMSE1 = (_RMSE1 / dis1.shape[0])**0.5
RMSE2 = (_RMSE2 / dis1.shape[0])**0.5
RMSE3 = (_RMSE3 / dis1.shape[0])**0.5
RMSE4 = (_RMSE4 / dis1.shape[0])**0.5
RMSE5 = (_RMSE5 / dis1.shape[0])**0.5
RMSE6 = (_RMSE6 / dis1.shape[0])**0.5
print("ME:", ME1, ME2, ME3, ME4, ME5, ME6, "\nMAE:", MAE1, MAE2, MAE3, MAE4, MAE5, MAE6, "\nRMSE:", RMSE1, RMSE2, RMSE3, RMSE4, RMSE5, RMSE6)

