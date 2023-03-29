import numpy as np
import math3d
from math import *
import transforms3d as t3d
# from libs.compute_reverse import *
# from transformations import *
from scipy.spatial.transform import Rotation as R

def quat2rotvec(a,b,c,d):
    r = R.from_quat([a, b, c, d])
    return r.as_rotvec()
def rotvec2quat(a,b,c):
    r = R.from_rotvec([a, b, c])
    return r.as_quat()
def quat2eular(a,b,c,d):
    r = R.from_quat([a, b, c, d])
    return r.as_euler('zxy')

def trans_compute(xb, yb, zb, tx, ty, tz):
    trans= np.eye(4)
    rot = R.from_rotvec([xb,yb,zb])
    trans[:3, :3] = rot.as_matrix()
    trans[0, 3] = tx
    trans[1, 3] = ty
    trans[2, 3] = tz
    return trans

def trans2posvec(T):
    r = R.from_matrix(T[:3,:3])
    rot_vec = r.as_rotvec()
    return rot_vec[0],rot_vec[1],rot_vec[2],T[0, 3],T[1, 3],T[2, 3]


def trans_combine(rot, t):
    trans= np.eye(4)

    trans[:3, :3] = rot
    trans[0, 3] = t[0]
    trans[1, 3] = t[1]
    trans[2, 3] = t[2]
    return trans
def trans_from_pos_quat(pose):
    [x, y, z, a, b, c, d] = pose
    trans = np.eye(4)
    r = R.from_quat([a,b,c,d])
    trans[:3, :3] = r.as_matrix()
    trans[0, 3] = x
    trans[1, 3] = y
    trans[2, 3] = z
    return trans
def trans_sep(trans):
    rot = trans[:3,:3]
    t=[trans[0,3],trans[1,3],trans[2,3]]
    return rot,t

def trans_compute_reverse(T):
    rot, t = trans_sep(T)
    r = R.from_matrix(rot)
    r = r.as_rotvec()
    return r[0],r[1],r[2],t[0],t[1],t[2]

def trans_compute_pos_quan(T):
    rot, t = trans_sep(T)
    r = R.from_matrix(rot)
    r = r.as_quat()
    return t[0],t[1],t[2],r[0],r[1],r[2],r[3]


