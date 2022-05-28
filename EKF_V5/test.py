import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import qgFunc as qg
from math import sqrt

Anor = np.array([1, 1, 1])
A = np.array([-1, 2, 3])
B = A.sum()
C = A/2
cc1 = qg.normalization(Anor)
cc2 = qg.normalization(cc1)
cc3 = np.linalg.norm(Anor * 2)

poseuler = np.array([8, 3, 2, math.pi/2, 0, 0])
posrpy = np.array([8, 3, 2, 0, 0, math.pi/2])
# print(qg.poseuler2T(poseuler))
# print(qg.posrpy2T(posrpy))


test_r = np.array([0, np.pi/4, np.pi/2])
r5 = R.from_euler('zyx', test_r).as_dcm()  # euler, zyx
r_xyz = R.from_euler('xyz', test_r).as_dcm()  # euler, zyx

# print('   r5:', r5)
# print(R.from_dcm(r5).as_euler('zyx'))
# print("    r_xyz", r_xyz)
# print(R.from_dcm(r_xyz).as_euler('xyz'))
# # as_rotvec()
#
# print(np.abs(A))

U = np.mat([[6, 7, 8],
            [4, 5, 4],
            [9, 8, 8],
            [3, 2, 1]])
U2 = np.ones([4, 3])
u = np.mat([2, 3, 3])

A = np.array(["R", "I", "T", "9", "I"])
B = np.array(["9", "I"])
print(B[:, None])

print(np.mat(B).T)
print(np.where(A == np.mat(B).T))
print(np.where(A == np.mat(B).T)[-1])
a = A
a = np.delete(a, np.where(a == np.mat(B).T)[-1])
print(a)
