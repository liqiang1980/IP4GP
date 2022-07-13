import numpy as np
import Plot_plus as pPlt

x_all = np.loadtxt('x_data.txt')
gd_all = np.loadtxt('gd_data.txt')
pPlt.plot_xt_GD_6in1(x_all, gd_all,
                     label1='x[mm]', label2='y[mm]', label3='z[mm]',
                     label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')

# u_all = np.loadtxt('u_data.txt')
# u1 = u_all[:, :4]
# u2 = u_all[:, 4:8]
# u3 = u_all[:, 8:12]
# u4 = u_all[:, 12:16]
# pPlt.plot_ut(u1, u1,

#                      label1='u1[deg]', label2='u2[deg]', label3='u3[deg]',
#                      label4='u4[deg]', label5='u5[deg]', label6='u6[deg]')
# pPlt.plot_ut(u2, u2,
#                      label1='u1[deg]', label2='u2[deg]', label3='u3[deg]',
#                      label4='u4[deg]', label5='u5[deg]', label6='u6[deg]')
# pPlt.plot_ut(u3, u3,
#                      label1='u1[deg]', label2='u2[deg]', label3='u3[deg]',
#                      label4='u4[deg]', label5='u5[deg]', label6='u6[deg]')
# pPlt.plot_ut(u4, u4,
#                      label1='u1[deg]', label2='u2[deg]', label3='u3[deg]',
#                      label4='u4[deg]', label5='u5[deg]', label6='u6[deg]')

# ju_all = np.loadtxt('ju_data.txt')
# ju1 = ju_all[:, :6]
# ju2 = ju_all[:, 6:12]
# ju3 = ju_all[:, 12:18]
# ju4 = ju_all[:, 18:24]
# pPlt.plot_xt_GD_6in1(ju1, ju1,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')
# pPlt.plot_xt_GD_6in1(ju2, ju2,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')
# pPlt.plot_xt_GD_6in1(ju3, ju3,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')
# pPlt.plot_xt_GD_6in1(ju4, ju4,
#                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                      label4='W1[deg]', label5='W2[deg]', label6='W3[deg]')
