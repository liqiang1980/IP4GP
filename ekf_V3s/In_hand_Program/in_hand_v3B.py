import numpy as np
import test_Plot_plus as pplt

full_len = 43
full_leni = 39

####################################################
##
##   12 figs , 4 * 12 lines ( there are 12 GD )
####################################################


# <editor-fold desc=">>>>>Load Data.">
file_dir_GD_1 = 'save_i/save_pose_GD_xyz.npy'
file_dir_GD_2 = 'save_i/save_pose_GD_rpy.npy'
GD_xyz = np.load(file_dir_GD_1)  # xyz of Truth
GD_rpy = np.load(file_dir_GD_2)  # rpy of Truth
print("???????????", GD_xyz.shape)
GD_x = GD_xyz[2:, 0] * 1000
GD_y = GD_xyz[2:, 1] * 1000
GD_z = GD_xyz[2:, 2] * 1000
GD_row = GD_rpy[2:, 0] * 57.296
GD_pitch = GD_rpy[2:, 1] * 57.296
GD_yaw = GD_rpy[2:, 2] * 57.296
Af_xyz = np.load("save_date/WithoutInitE/save_f/save_pose_y_t_xyz.npy")
Af_rpy = np.load("save_date/WithoutInitE/save_f/save_pose_y_t_rpy.npy")
Ai_xyz = np.load("save_date/WithoutInitE/save_i/save_pose_y_t_xyz.npy")
Ai_rpy = np.load("save_date/WithoutInitE/save_i/save_pose_y_t_rpy.npy")
print("Check:", Af_xyz.shape, Af_rpy.shape, Ai_xyz.shape, Ai_rpy.shape)
Afx = Af_xyz[6:full_len, 0] * 1000
Afy = Af_xyz[6:full_len, 1] * 1000
Afz = Af_xyz[6:full_len, 2] * 1000
Afrow = Af_rpy[6:full_len, 0] * 57.296
Afpitch = Af_rpy[6:full_len, 1] * 57.296
Afyaw = Af_rpy[6:full_len, 2] * 57.296
Aix = Ai_xyz[2:, 0] * 1000
Aiy = Ai_xyz[2:, 1] * 1000
Aiz = Ai_xyz[2:, 2] * 1000
Airow = Ai_rpy[2:, 0] * 57.296
Aipitch = Ai_rpy[2:, 1] * 57.296
Aiyaw = Ai_rpy[2:, 2] * 57.296
# </editor-fold>

fEKF_all = np.load("save_date/fEKF_incremental.npy")
iEKF_all = np.load("save_date/iEKF_incremental.npy")
print("CHek,all f:", fEKF_all.shape)
print("CHek,all i:", iEKF_all.shape)

fEKFx = fEKF_all[6:full_len, 0] * 1000
fEKFy = fEKF_all[6:full_len, 1] * 1000
fEKFz = fEKF_all[6:full_len, 2] * 1000
fEKFrow = fEKF_all[6:full_len, 3] * 57.296
fEKFpitch = fEKF_all[6:full_len, 4] * 57.296
fEKFyaw = fEKF_all[6:full_len, 5] * 57.296

iEKFx = iEKF_all[2:full_leni, 0] * 1000
iEKFy = iEKF_all[2:full_leni, 1] * 1000
iEKFz = iEKF_all[2:full_leni, 2] * 1000
iEKFrow = iEKF_all[2:full_leni, 3] * 57.296
iEKFpitch = iEKF_all[2:full_leni, 4] * 57.296
iEKFyaw = iEKF_all[2:full_leni, 5] * 57.296

fEKF_all = fEKF_all[full_len:]
iEKF_all = iEKF_all[full_leni:]
while fEKF_all.size != 0:
    fEKFx = np.vstack((fEKFx, fEKF_all[6:full_len, 0] * 1000))
    fEKFy = np.vstack((fEKFy, fEKF_all[6:full_len, 1] * 1000))
    fEKFz = np.vstack((fEKFz, fEKF_all[6:full_len, 2] * 1000))
    fEKFrow = np.vstack((fEKFrow, fEKF_all[6:full_len, 3] * 57.296))
    fEKFpitch = np.vstack((fEKFpitch, fEKF_all[6:full_len, 4] * 57.296))
    fEKFyaw = np.vstack((fEKFyaw, fEKF_all[6:full_len, 5] * 57.296))
    fEKF_all = fEKF_all[full_len:]
while iEKF_all.size != 0:
    print(iEKF_all.shape)
    iEKFx = np.vstack((iEKFx, iEKF_all[2:full_leni, 0] * 1000))
    iEKFy = np.vstack((iEKFy, iEKF_all[2:full_leni, 1] * 1000))
    iEKFz = np.vstack((iEKFz, iEKF_all[2:full_leni, 2] * 1000))
    iEKFrow = np.vstack((iEKFrow, iEKF_all[2:full_leni, 3] * 57.296))
    iEKFpitch = np.vstack((iEKFpitch, iEKF_all[2:full_leni, 4] * 57.296))
    iEKFyaw = np.vstack((iEKFyaw, iEKF_all[2:full_leni, 5] * 57.296))
    iEKF_all = iEKF_all[full_leni:]

print("???", fEKFx.shape, iEKFx.shape)
# <editor-fold desc=">>>>>calculate all mean">
fEKFx_mean = np.mean(fEKFx, 0)
fEKFy_mean = np.mean(fEKFy, 0)
fEKFz_mean = np.mean(fEKFz, 0)
fEKFrow_mean = np.mean(fEKFrow, 0)
fEKFpitch_mean = np.mean(fEKFpitch, 0)
fEKFyaw_mean = np.mean(fEKFyaw, 0)

iEKFx_mean = np.mean(iEKFx, 0)
iEKFy_mean = np.mean(iEKFy, 0)
iEKFz_mean = np.mean(iEKFz, 0)
iEKFrow_mean = np.mean(iEKFrow, 0)
iEKFpitch_mean = np.mean(iEKFpitch, 0)
iEKFyaw_mean = np.mean(iEKFyaw, 0)
# </editor-fold>

# <editor-fold desc=">>>>>calculate all std.">
fEKFx_std = np.std(fEKFx, 0)
fEKFy_std = np.std(fEKFy, 0)
fEKFz_std = np.std(fEKFz, 0)
fEKFrow_std = np.std(fEKFrow, 0)
fEKFpitch_std = np.std(fEKFpitch, 0)
fEKFyaw_std = np.std(fEKFyaw, 0)

iEKFx_std = np.std(iEKFx, 0)
iEKFy_std = np.std(iEKFy, 0)
iEKFz_std = np.std(iEKFz, 0)
iEKFrow_std = np.std(iEKFrow, 0)
iEKFpitch_std = np.std(iEKFpitch, 0)
iEKFyaw_std = np.std(iEKFyaw, 0)
# </editor-fold>
print("CHek f,x:", fEKFx[:, 0], fEKFyaw[:, 0])
print("CHek i,x:", iEKFx[:, 0], iEKFyaw[:, 0])
print("CHek,x mean:", fEKFx_mean)
print("CHek,x std:", fEKFx_std)
print("chEck,x GD:", GD_x)

# pplt.plot_all_BAND2(iEKFx_mean, iEKFx_std, iEKFy_mean, iEKFy_std, iEKFz_mean, iEKFz_std,
#                     fEKFx_mean, fEKFy_mean, fEKFz_mean,
#                     GD_x, GD_y, GD_z,
#                     "x [mm]", "y [mm]", "z [mm]")
#
# pplt.plot_all_BAND2(iEKFrow_mean, iEKFrow_std, iEKFpitch_mean, iEKFpitch_std, iEKFyaw_mean, iEKFyaw_std,
#                    fEKFrow_mean, fEKFpitch_mean, fEKFyaw_mean,
#                     GD_row, GD_pitch, GD_yaw,
#                     "$\phi$ [deg]", r"$\theta$ [deg]", "$\psi$ [deg]")

pplt.plot_all_BAND_once(Aix, Aiy, Aiz, Afx, Afy, Afz,
                        GD_x, GD_y, GD_z,
                        "x [mm]", "y [mm]", "z [mm]")
pplt.plot_all_BAND_once(Airow, Aipitch, Aiyaw, Afrow, Afpitch, Afyaw,
                        GD_row, GD_pitch, GD_yaw,
                        "$\phi$ [deg]", r"$\theta$ [deg]", "$\psi$ [deg]")
