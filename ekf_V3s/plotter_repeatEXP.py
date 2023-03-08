import numpy as np
import Plot_plus as pPlt

_FOLDER = ["res_free_cup",  # 0
           "res_frozen_cup",  # 1
           "res_upsidedown_cup",  # 2
           "res_free_cylinder",  # 3
           "res_frozen_cylinder",  # 4
           "res_free_cup_tips",  # 5
           "res_free_cup_4tips",  # 6
           ]
fd = 5  # <<<<< Folder choose
FOLDER = _FOLDER[fd]
x_state_repeatRXP = np.load("ekf_obj_pose/" + FOLDER + "/x_state_repeatEXP.npy")
gd_all = np.loadtxt("ekf_obj_pose/" + FOLDER + "/x_gt_palm.txt")
print("Check x_state_all:", x_state_repeatRXP.shape)
print("Check gd_all:", gd_all.shape)
cut = [[10, 10, 10, 0, 3, 3],  # 0
       [118, 124, 124],  # 1
       [],  # 2
       [],  # 3
       [],  # 4
       [1, 0, 0, 0, 1, 1],  # 5
       []]  # 6
GD1 = gd_all[cut[fd][0]:, 0] * 1000
# GD1 = [gd_all[1, 0] * 1000] * (gd_all.shape[0]-1)
GD2 = gd_all[cut[fd][1]:, 1] * 1000
GD3 = gd_all[cut[fd][2]:, 2] * 1000
# GD4 = gd_all[1:, 3] * 57.3
# GD5 = gd_all[1:, 4] * 57.3
# GD6 = gd_all[1:, 5] * 57.3
GD4 = gd_all[cut[fd][3]:, 3] * 57.3
GD5 = gd_all[cut[fd][4]:, 4] * 57.3
GD6 = gd_all[cut[fd][5]:, 5] * 57.3

x_all = np.array(x_state_repeatRXP[:, cut[fd][0]:, 0] * 1000)
y_all = np.array(x_state_repeatRXP[:, cut[fd][1]:, 1] * 1000)
z_all = np.array(x_state_repeatRXP[:, cut[fd][2]:, 2] * 1000)
# ang1_all = np.array(x_state_repeatRXP[:, 1:, 3] * 57.3)
# ang2_all = np.array(x_state_repeatRXP[:, 1:, 4] * 57.3)
# ang3_all = np.array(x_state_repeatRXP[:, 1:, 5] * 57.3)
ang1_all = np.array(x_state_repeatRXP[:, cut[fd][3]:, 3] * 57.3)
ang2_all = np.array(x_state_repeatRXP[:, cut[fd][4]:, 4] * 57.3)
ang3_all = np.array(x_state_repeatRXP[:, cut[fd][5]:, 5] * 57.3)

x_mean = np.mean(x_all, axis=0)
y_mean = np.mean(y_all, axis=0)
z_mean = np.mean(z_all, axis=0)
ang1_mean = np.mean(ang1_all, axis=0)
ang2_mean = np.mean(ang2_all, axis=0)
ang3_mean = np.mean(ang3_all, axis=0)
x_std = np.std(x_all, axis=0)
y_std = np.std(y_all, axis=0)
z_std = np.std(z_all, axis=0)
ang1_std = np.std(ang1_all, axis=0)
ang2_std = np.std(ang2_all, axis=0)
ang3_std = np.std(ang3_all, axis=0)



pPlt.plot_all_BAND2(mean1=x_mean, mean2=y_mean, mean3=z_mean,
                    std1=x_std, std2=y_std, std3=z_std,
                    GD1=GD1, GD2=GD2, GD3=GD3,
                    label1="x[mm]", label2="y[mm]", label3="z[mm]")
pPlt.plot_all_BAND2(mean1=ang1_mean, mean2=ang2_mean, mean3=ang3_mean,
                    std1=ang1_std, std2=ang2_std, std3=ang3_std,
                    GD1=GD4, GD2=GD5, GD3=GD6,
                    label1="$\phi$[deg]", label2=r"$\theta$[deg]", label3="$\psi$[deg]")
