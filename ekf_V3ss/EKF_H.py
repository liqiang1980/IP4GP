import sympy as sy
from sympy import sin, cos, sqrt


def H_calculator(W1, W2, W3, pos_CO_x, pos_CO_y, pos_CO_z):
    H = sy.Matrix([
        [1, 0, 0, pos_CO_x * (
                    W1 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                2 * W1 * W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + pos_CO_y * (
                     -2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W3 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     -2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
         pos_CO_x * (W2 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W2 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_y * (
                     -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_z * (
                     -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 ** 2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), pos_CO_x * (
                     W3 * (-W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W2 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_y * (
                     -2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W3 ** 2 * cos(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     -2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2))],
        [0, 1, 0, pos_CO_x * (-2 * W1 ** 2 * W2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * W3 * cos(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                          1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_z * (
                     -W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)), pos_CO_x * (
                     -2 * W1 * W2 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                         W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_y * (
                     W2 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 * W3 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2)) + pos_CO_z * (
                     -W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2)),
         pos_CO_x * (-2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 ** 2 * cos(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W3 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W3 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W3 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W3 / (
                                             W1 ** 2 + W2 ** 2 + W3 ** 2))) + pos_CO_z * (
                     -W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 * W3 ** 2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (W1 ** 2 + W2 ** 2 + W3 ** 2))],
        [0, 0, 1, pos_CO_x * (-2 * W1 ** 2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                    W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 ** 2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W1 * W2 * cos(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W1 * W2 * sin(
            sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                          1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                          W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 ** 2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W1 * W2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W1 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W1 * W2 ** 2 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W1 / (W1 ** 2 + W2 ** 2 + W3 ** 2))),
         pos_CO_x * (-2 * W1 * W2 * W3 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - W2 ** 2 * cos(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 ** 2 * sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - sin(
             sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_y * (
                     W1 * W2 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 ** 2 * W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 ** 2 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W3 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W2 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 - 2 * W2 / (W1 ** 2 + W2 ** 2 + W3 ** 2))),
         pos_CO_x * (-2 * W1 * W3 ** 2 * (1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W1 * W3 ** 2 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W1 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) + W2 * W3 * sin(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2)) + pos_CO_y * (
                     W1 * W3 * cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W1 * W3 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) - 2 * W2 * W3 ** 2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + W2 * W3 ** 2 * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** (3 / 2) + W2 * (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) / (
                                 W1 ** 2 + W2 ** 2 + W3 ** 2)) + pos_CO_z * (
                     W3 * (-W1 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2) - W2 ** 2 / (W1 ** 2 + W2 ** 2 + W3 ** 2)) * sin(
                 sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)) / sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2) + (
                                 1 - cos(sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2))) * (
                                 2 * W1 ** 2 * W3 / (W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2 + 2 * W2 ** 2 * W3 / (
                                     W1 ** 2 + W2 ** 2 + W3 ** 2) ** 2))]
    ])
    return H


def get_H_symbols():
    W1, W2, W3 = sy.symbols('W1, W2, W3')
    pos_OH_x, pos_OH_y, pos_OH_z = sy.symbols('pos_OH_x, pos_OH_y, pos_OH_z')
    pos_CO_x, pos_CO_y, pos_CO_z = sy.symbols('pos_CO_x, pos_CO_y, pos_CO_z')

    w1 = W1 / sy.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)
    w2 = W2 / sy.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)
    w3 = W3 / sy.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)
    theta = sy.sqrt(W1 ** 2 + W2 ** 2 + W3 ** 2)
    w_mat = sy.Matrix([[0, -w3, w2],
                       [w3, 0, -w1],
                       [-w2, w1, 0]])
    ww_mat = w_mat * w_mat
    print("ww_mat:", ww_mat)

    """ 
    OH: object in hand frame
    CO: contact point in object frame
    CH: contact point in hand frame
    """
    # R_OH object's orientation from Eq.2.14 in "A Mathematical Introduction to
    # Robotic Manipulation"
    R_OH = sy.Matrix(sy.eye(3)) + w_mat * sy.sin(theta) + ww_mat * (1 - sy.cos(theta))
    print("R_OH", R_OH)

    pos_OH = sy.Matrix([[pos_OH_x], [pos_OH_y], [pos_OH_z]])
    pos_CO = sy.Matrix([[pos_CO_x], [pos_CO_y], [pos_CO_z]])

    pos_CH = pos_OH + R_OH * pos_CO
    print("pos:", pos_OH, pos_CO, pos_CH.shape)
    H = sy.Matrix(
        [[sy.diff(pos_CH, pos_OH_x), sy.diff(pos_CH, pos_OH_y), sy.diff(pos_CH, pos_OH_z), sy.diff(pos_CH, W1),
          sy.diff(pos_CH, W2), sy.diff(pos_CH, W3)]])

    print ('H is ', H)

    print("ww_mat:", ww_mat)
    print("H:", H.shape, "\n",
          H[0, :], "\n",
          H[1, :], "\n",
          H[2, :], "\n", )

    H_ch = H.subs(
        {W1: 0.6, W2: 0.3, W3: 0.7, pos_OH_x: 4, pos_OH_y: 7, pos_OH_z: 99, pos_CO_x: 40, pos_CO_y: 70, pos_CO_z: 990})
    print("H_ch", H_ch.shape, "\n",
          H_ch[0, :], "\n",
          H_ch[1, :], "\n",
          H_ch[2, :])

if __name__ == '__main__':
    get_H_symbols()
    H_val = H_calculator(W1=0.6, W2=0.3, W3=0.7, pos_CO_x=40, pos_CO_y=70, pos_CO_z=990)
    print("H_val:", H_val.shape, "\n",
          H_val[0, :], "\n",
          H_val[1, :], "\n",
          H_val[2, :])
