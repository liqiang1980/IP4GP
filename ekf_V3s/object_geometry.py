import numpy as np


def get_normal_from_formula(coe, point):
    nor = np.array([0., 0., -1.])
    nor[0] = 2. * coe[0] * point[0] + coe[1] * point[1] + coe[3]
    nor[1] = coe[1] * point[0] + 2. * coe[2] * point[1] + coe[4]
    return nor


def surface_cup(cur_x, cur_y, cur_z):
    s = [129.4, 1.984, 139., -0.666, 0.32, -0.207]  # 曲面方程的6个系数
    return (-1.0) * get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s


def surface_bottle(cur_x, cur_y, cur_z):
    # -330.2*x^2-380.2*xy-259.4*y^2+17.66*x-11.82*y-0.183
    s = [-330.2, -380.2, -259.4, 17.66, -11.82, -0.183]  # 曲面方程的6个系数
    return (-1.0) * get_normal_from_formula(s, [cur_x, cur_y, cur_z]), s
