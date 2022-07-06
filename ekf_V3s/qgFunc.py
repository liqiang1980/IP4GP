import math
from math import sin, cos
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation
import copy
import xml.etree.ElementTree as ET

coe = np.array([129.4, 1.984, 139., -0.666, 0.32, -0.207])  # 6 coefficients of the surface
xml_path = "/home/lqg/PycharmProjects/EKF_v5/allegro_hand_description/UR5_allegro_test.xml"
xml_tree = ET.parse(xml_path)
xml_root = xml_tree.getroot()


def if_match(nodelist, name):
    """
    For XML tree, check if 'name' exists in 'nodelist'
    """
    if nodelist.get('name') == {'name': name}.get('name'):
        return True
    else:
        return False


def get_poseuler_from_xml(nodelist):
    """
    Put pos and euler into params
    """
    pos = nodelist.get('pos')
    euler = nodelist.get('euler')
    print(">>FIND:", nodelist.tag, ':', nodelist.attrib)
    return pos, euler


def kdl_calc_fk(fk, q, pos):
    """
    Calculate FK
    input: fk, a fk chain established by KDL Library.
    input: q, joint positions, established by KDL JntArray
    output:pos
    """
    fk_flag = fk.JntToCart(q, pos)


def posquat2T(posquat):
    """
    Translate posquat to T
    :param posquat: x y z qx qy qz qw
    :return:
    """
    quat = np.array(posquat[3:])
    pos = np.array(posquat[:3])
    # rot = R.from_quat(quat).as_dcm()
    rot = R.from_quat(quat).as_matrix()
    tform = np.eye(4)
    tform[0:3, 0:3] = rot
    tform[0:3, 3] = pos.transpose()
    return tform


def T2posquat(T):
    """
    Translate T to posquat
    :param posquat: x y z qx qy qz qw
    :return:
    """
    pos = T[:3, 3].T
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # as_quat(): x y z w
    posquat = np.empty(7)
    posquat[:3] = pos
    posquat[3:] = quat
    return posquat


def posquat2poseuler(posquat):
    """
    Translate pos + quat to pos + euler
    """
    poseuler = np.zeros(6)
    T0 = posquat2T(posquat)
    poseuler[:3] = posquat[:3]
    poseuler[3:] = Rotation.from_matrix(T0[:3, :3]).as_euler('ZYX')
    return poseuler


def posquat2posrpy(posquat):
    """
    Translate pos + quat to pos + rpy
    """
    posrpy = np.zeros(6)
    posrpy[:3] = posquat[:3]
    posrpy[3:] = Rotation.from_quat(posquat[3:]).as_euler('xyz')
    return posrpy


def posquat2posrotvec(posquat):
    posrotvec = np.zeros(6)
    posrotvec[:3] = posquat[:3]
    posrotvec[3:] = Rotation.from_quat(posquat[3:]).as_rotvec()
    return posrotvec


def posrpy2posrotvec(posrpy):
    posrotvec = np.zeros(6)
    posrotvec[:3] = posrpy[:3]
    posrotvec[3:] = Rotation.from_euler('xyz', posrpy[3:]).as_rotvec()
    return posrotvec

def posrotvec2posrpy(posrotvec):
    posrpy = np.zeros(6)
    posrpy[:3] = posrotvec[:3]
    posrpy[3:] = Rotation.from_rotvec(posrotvec[3:]).as_euler('xyz')
    return posrpy


def cross_product_matrix_from_vector3d(vector3d00):
    """
    cross product matrix (skew symmetric matrix) from 3d-vector
    """
    vector3d = np.reshape(np.array(copy.deepcopy(vector3d00)), 3)
    cpm = np.mat(np.zeros((3, 3)))
    cpm[0, 1] = -vector3d[2]
    cpm[0, 2] = vector3d[1]
    cpm[1, 0] = vector3d[2]
    cpm[1, 2] = -vector3d[0]
    cpm[2, 0] = -vector3d[1]
    cpm[2, 1] = vector3d[0]
    return cpm


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Calculate rotation matrix from 2 given vectors
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if np.linalg.norm(v) == 0:
        if np.dot(a, b) > 0:
            return np.eye(3)
        else:
            nx = [1., 0., 0.]
            if np.cross(a, nx):
                c = np.cross(a, nx)
            else:
                c = np.cross(a, [0., 1., 0.])
            R_rot = Rotation.from_rotvec(180.0 * c, degrees=True).as_matrix()
            return R_rot
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    Q0 = np.ravel(Q0)
    Q1 = np.ravel(Q1)
    # Extract the values from Q0
    w0 = Q0[3]
    x0 = Q0[0]
    y0 = Q0[1]
    z0 = Q0[2]

    # Extract the values from Q1
    w1 = Q1[3]
    x1 = Q1[0]
    y1 = Q1[1]
    z1 = Q1[2]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


# def euler2R(input):
#     """
#     Input euler (zyx), return their R mat in the same frame.
#     """
#     input = np.ravel(input)
#     c_a = math.cos(input[0])
#     c_b = math.cos(input[1])
#     c_g = math.cos(input[2])
#     s_a = math.sin(input[0])
#     s_b = math.sin(input[1])
#     s_g = math.sin(input[2])
#     R = np.mat(np.eye(3))
#     R[0, 0] = c_a * c_b
#     R[0, 1] = c_a * s_b * s_g - s_a * c_g
#     R[0, 2] = c_a * s_b * c_g + s_a * s_g
#     R[1, 0] = s_a * c_b
#     R[1, 1] = s_a * s_b * s_g + c_a * c_g
#     R[1, 2] = s_a * s_b * c_g - c_a * s_g
#     R[2, 0] = -s_b
#     R[2, 1] = c_b * s_g
#     R[2, 2] = c_b * c_g
#     return R

def euler2R(euler):
    """
    Input euler (zyx), return their R mat in the same frame.
    """
    R = Rotation.from_euler('ZYX', euler).as_matrix()
    return R


def rpy2R(rpy):
    """
    Input euler (zyx), return their R mat in the same frame.
    """
    R = Rotation.from_euler('xyz', rpy).as_matrix()
    return R


# def rpy2R(input):
#     """
#     :param input: RPY (xyz)
#     :return: R
#     """
#     input = np.ravel(input)
#     input = np.array([input[2], input[1], input[0]])
#     R = euler2R(input)
#     return R


# def rpy2R(rpy):
#     """
#     :param input: RPY (xyz)
#     :return: R
#     """
#     R = Rotation.from_euler('xyz', rpy).as_dcm()[0]
#     return R


def poseuler2T(input):
    """
    Input: pos and euler (zyx)
    Return: T mat.
    """
    T = np.mat(np.eye(4))
    pos = np.mat(input[:3]).T
    euler = input[3:]
    R = euler2R(euler)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def posrpy2T(input):
    """
    Input: pos and rpy (xyz)
    Return: T mat.
    """
    T = np.mat(np.eye(4))
    pos = np.mat(input[:3]).T
    rpy = input[3:]
    # R = Rotation.from_euler('xyz', rpy).as_dcm()
    R = Rotation.from_euler('xyz', rpy).as_matrix()
    # R = rpy2R(rpy)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def vec2R(vec):
    """
    Input a vector, return its R mat in the same frame.
    """
    # R = Rotation.from_rotvec(vec).as_dcm()
    R = rotation_matrix_from_vectors(np.array([0, 0, 1]), vec)
    return R


def R2vec(R):
    """
    Input R mat, return corresponding vec vector.
    """
    n_unit = Rotation.from_dcm(R).as_rotvec()
    return n_unit


def rpy2vec(rpy):
    """
    Translate rpy to vector
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    vec = np.zeros(3)
    vec[0] = cos(yaw) * cos(pitch)
    vec[1] = sin(yaw) * cos(pitch)
    vec[2] = sin(pitch)
    return vec


def vec2rpy(vec):
    """
    Calculate RPY from xyz and normal
    """
    den = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
    vec = vec / den
    R_o = rotation_matrix_from_vectors(vec1=np.array([0, 0, 1]), vec2=vec)
    rpy = Rotation.from_matrix(R_o).as_euler('xyz', degrees=False)
    return rpy


def get_S(r):
    """
    S is the cross-product matrix
    input: r is any vector 3*1
    """
    r = np.ravel(r)
    S = np.zeros((3, 3))
    S[0, 1] = -r[2]
    S[0, 2] = r[1]
    S[1, 0] = r[2]
    S[1, 2] = -r[0]
    S[2, 0] = -r[1]
    S[2, 1] = r[0]
    return S


def get_G(R, p, ci):
    """
    G is the Grasp Matrix
    input: R is the rotation matrix of object in {W}
    input: p is the position of object in {W}.
    input: ci is the positon of contact point in {W}.
    """
    print("  p & ci   ", p, ci)
    S = get_S(ci - p)  # Get S(c_i - p), palm frame
    G = np.mat(np.eye(6))
    G[:3, :3] = R
    G[3:, 3:] = R
    G[3:, :3] = np.matmul(S, R)
    # print("   S    R    G:\n", S, R, G)
    return G, S


def get_H_O(pos):
    global coe
    """
    input: Taxel Position(x y z) in object frame {O}.
    return: Second order total differential of surface equation. That is, H.
    The H matrix is in th object frame {O}
    The surface equation coefficients is obtained in the simulation program (QG method)
    """
    H = np.mat([[2 * coe[0], coe[1], coe[3] / pos[2]],
                [coe[1], 2 * coe[2], coe[4] / pos[2]],
                [-1 / pos.sum(), -1 / pos.sum(), -1 / pos.sum()]])
    return H


def get_normal_O(pos):
    global coe
    """
    input: Position(x y z) in object frame {O}.
    return: First order total differential of surface equation. That is, normal.
    The normal is in the object frame {O}
    The surface equation coefficients is obtained in the simulation program (QG method)
    """
    pos = np.ravel(pos)
    nor = np.array([0., 0., -1.])
    nor[0] = 2. * coe[0] * pos[0] + coe[1] * pos[1] + coe[3]
    nor[1] = coe[1] * pos[0] + 2. * coe[2] * pos[1] + coe[4]
    nor = np.mat(nor).T
    print("# Normal in {O}:", nor.T)
    return nor


def matmul3mat(mat1, mat2, mat3):
    mat = np.matmul(np.matmul(mat1, mat2), mat3)
    return mat


def get_z_or_h(info, pos0, pos1, pos2, pos3, normal0, normal1, normal2, normal3):
    """
    Splice out z or h
    """
    z = np.vstack((pos0, normal0))
    z = np.vstack((z, pos1))
    z = np.vstack((z, normal1))
    z = np.vstack((z, pos2))
    z = np.vstack((z, normal2))
    z = np.vstack((z, pos3))
    z = np.vstack((z, normal3))
    print("##", info, ".T:", z.T)
    return z


def get_z_or_h2(info, pos0, pos1, pos2, pos3):
    """
    Splice out z or h
    """
    z = np.vstack((pos0, pos1))
    z = np.vstack((z, pos2))
    z = np.vstack((z, pos3))
    print("##", info, ".T:", z.T)
    return z


def get_z_or_h_PN(info, pos0, pos1, pos2, pos3, normal0, normal1, normal2, normal3):
    """
    Splice out z or h
    """
    z = np.vstack((pos0, pos1))
    z = np.vstack((z, pos2))
    z = np.vstack((z, pos3))
    z = np.vstack((z, normal0))
    z = np.vstack((z, normal1))
    z = np.vstack((z, normal2))
    z = np.vstack((z, normal3))
    print("##", info, ".T:", z.T)
    return z


# def getH_object2taxel(R_object_W, posnormal_obj0, posnormal_obj1, posnormal_obj2, posnormal_obj3,
#                       posnormal_taxel0, posnormal_taxel1, posnormal_taxel2, posnormal_taxel3):
#     """
#     Input: x_t, the state of object, 23*1 (3*1 pos + 3*1 rpy + 16 * 1 twist + [1])
#     """
#     pos_obj0 = posnormal_obj0[:3]
#     pos_obj1 = posnormal_obj1[:3]
#     pos_obj2 = posnormal_obj2[:3]
#     pos_obj3 = posnormal_obj3[:3]
#     normal_obj0 = posnormal_obj0[3:]
#     normal_obj1 = posnormal_obj1[3:]
#     normal_obj2 = posnormal_obj2[3:]
#     normal_obj3 = posnormal_obj3[3:]
#     pos_taxel0 = posnormal_taxel0[:3]
#     pos_taxel1 = posnormal_taxel1[:3]
#     pos_taxel2 = posnormal_taxel2[:3]
#     pos_taxel3 = posnormal_taxel3[:3]
#     normal_taxel0 = posnormal_taxel0[3:]
#     normal_taxel1 = posnormal_taxel1[3:]
#     normal_taxel2 = posnormal_taxel2[3:]
#     normal_taxel3 = posnormal_taxel3[3:]
#     delta_pos0 = pos_taxel0 - pos_obj0
#     delta_pos1 = pos_taxel1 - pos_obj1
#     delta_pos2 = pos_taxel2 - pos_obj2
#     delta_pos3 = pos_taxel3 - pos_obj3
#     delta_normal0 = normal_taxel0 - normal_obj0
#     delta_normal1 = normal_taxel1 - normal_obj1
#     delta_normal2 = normal_taxel2 - normal_obj2
#     delta_normal3 = normal_taxel3 - normal_obj3
#     mat0 = np.mat(np.zeros([25, 23]))  # Format mat: 25*23
#     for i in range(8):
#         mat0[i*3: 3+i*3, :3] = np.mat(np.eye(3))
#     mat0[24][22] = 1
#     pos_obj = np.vstack((delta_pos0, delta_pos0))
#     pos_obj = np.vstack((pos_obj, delta_pos1))
#     pos_obj = np.vstack((pos_obj, delta_pos1))
#     pos_obj = np.vstack((pos_obj, delta_pos2))
#     pos_obj = np.vstack((pos_obj, delta_pos2))
#     pos_obj = np.vstack((pos_obj, delta_pos3))
#     pos_obj = np.vstack((pos_obj, delta_pos3))
#     pos_obj = np.vstack((pos_obj, np.mat(1)))
#     print("CHECK pos_obj{W}:", pos_obj)
#     mat_trans = np.hstack((np.mat(np.eye(24)), pos_obj))  # Translation move mat: 24*25
#     mat_R0 = normal2R(delta_normal0)
#     mat_R1 = normal2R(delta_normal1)
#     mat_R2 = normal2R(delta_normal2)
#     mat_R3 = normal2R(delta_normal3)
#     mat_rot = np.mat(np.eye(24))  # Rotation move mat: 24*24
#     mat_h = np.mat()  # normal_O = mat_h * pos_O
#     mat_rot[:3, :3] = mat_R0
#     mat_rot[3:6, 3:6] = np.dot(matmul3mat(R_object_W, , np.linalg.inv(R)), mat_R0)
#     mat_rot[6:9, 6:9] = mat_R1
#     mat_rot[9:12, 9:12] = mat_R1
#     mat_rot[12:15, 12:15] = mat_R2
#     mat_rot[15:18, 15:18] = mat_R2
#     mat_rot[18:21, 18:21] = mat_R3
#     mat_rot[21:24, 21:24] = mat_R3


def moving_average(x, w):
    """
    x: input array
    w: window size
    """
    x = np.hstack((0, x))
    x = np.hstack((x, 0))
    x = np.hstack((x, 0))
    return np.convolve(x, np.ones(w), 'valid') / w


def testH(x_t, pos_object_W, pos_taxel_W0_h, R_O_W):
    global coe
    # xt = np.mat(np.hstack((x_t, 1))).T
    xt = x_t
    print("   @@xt:", xt)
    delta_pos = np.mat(pos_taxel_W0_h - pos_object_W)
    print("   @@delta_pos:", pos_taxel_W0_h, pos_object_W)
    # delta_normal = np.mat(np.ravel(normal_W0_h) - normal_object_W).T

    mat0 = np.mat(np.zeros([4, 23]))
    mat0[:3, :3] = np.mat(np.eye(3))
    mat0[3, 22] = 1
    # print("   @mat0", mat0)
    mat_trans = np.hstack((np.mat(np.eye(3)), delta_pos))
    # mat_rot = normal2R(delta_normal)
    # mat_H_part = matmul3mat(mat_rot, mat_trans, mat0)
    mat_H_part1 = np.matmul(mat_trans, mat0)  # part1 for pos only
    pos_taxel_W = np.matmul(mat_H_part1, xt)

    mat_trans2 = np.mat(np.eye(4))
    mat_trans2[:3, 3] = delta_pos
    mat_R_W_O = np.mat(np.eye(4))
    mat_R_W_O[:3, :3] = np.linalg.inv(R_O_W)
    mat_h = np.mat([[2 * coe[0], coe[1], 0, coe[3]],
                    [coe[1], 2 * coe[2], 0, coe[4]],
                    [0, 0, 0, -1]])
    mat_H_part2 = np.matmul(matmul3mat(R_O_W, mat_h, mat_R_W_O), mat_trans2)  # part2 for normal only
    normal_taxel_W = matmul3mat(mat_H_part2, mat0, xt)
    return pos_taxel_W, normal_taxel_W


def get_bigH(pos_object_W, pos_taxel_W0_h, pos_taxel_W1_h, pos_taxel_W2_h, pos_taxel_W3_h, R_O_W):
    global coe
    delta_pos0 = np.mat(pos_taxel_W0_h - pos_object_W)
    delta_pos1 = np.mat(pos_taxel_W1_h - pos_object_W)
    delta_pos2 = np.mat(pos_taxel_W2_h - pos_object_W)
    delta_pos3 = np.mat(pos_taxel_W3_h - pos_object_W)
    mat_h = np.mat([[2 * coe[0], coe[1], 0, coe[3]],
                    [coe[1], 2 * coe[2], 0, coe[4]],
                    [0, 0, 0, -1]])

    mat0 = np.mat(np.zeros([25, 23]))
    mat0[:3, :3] = np.mat(np.eye(3))
    mat0[3:6, :3] = np.mat(np.eye(3))
    mat0[6:9, :3] = np.mat(np.eye(3))
    mat0[9:12, :3] = np.mat(np.eye(3))
    mat0[12:15, :3] = np.mat(np.eye(3))
    mat0[15:18, :3] = np.mat(np.eye(3))
    mat0[18:21, :3] = np.mat(np.eye(3))
    mat0[21:24, :3] = np.mat(np.eye(3))
    mat0[24, 22] = 1

    mat1 = np.mat(np.eye(25))
    mat1[:3, 24] = delta_pos0
    mat1[3:6, 24] = delta_pos0
    mat1[6:9, 24] = delta_pos1
    mat1[9:12, 24] = delta_pos1
    mat1[12:15, 24] = delta_pos2
    mat1[15:18, 24] = delta_pos2
    mat1[18:21, 24] = delta_pos3
    mat1[21:24, 24] = delta_pos3

    mat2 = np.mat(np.eye(25))
    mat2[3:6, 3:6] = np.linalg.inv(R_O_W)
    mat2[9:12, 9:12] = np.linalg.inv(R_O_W)
    mat2[15:18, 15:18] = np.linalg.inv(R_O_W)
    mat2[21:24, 21:24] = np.linalg.inv(R_O_W)

    mat3 = np.mat(np.zeros([24, 25]))
    mat3[:24, :24] = np.mat(np.eye(24))
    mat3[3:6, 3:6] = mat_h[:3, :3]
    mat3[9:12, 9:12] = mat_h[:3, :3]
    mat3[15:18, 15:18] = mat_h[:3, :3]
    mat3[21:24, 21:24] = mat_h[:3, :3]
    mat3[3:6, 24] = mat_h[:3, 3]
    mat3[9:12, 24] = mat_h[:3, 3]
    mat3[15:18, 24] = mat_h[:3, 3]
    mat3[21:24, 24] = mat_h[:3, 3]

    mat4 = np.mat(np.eye(24))
    mat4[3:6, 3:6] = R_O_W
    mat4[9:12, 9:12] = R_O_W
    mat4[15:18, 15:18] = R_O_W
    mat4[21:24, 21:24] = R_O_W

    mat_H = matmul3mat(mat4, mat3, matmul3mat(mat2, mat1, mat0))  # 24 * 23
    return mat_H


def get_bigH2(R_O_W):
    H = np.mat(np.zeros([12, 35]))
    H[:3, :3] = np.mat(np.eye(3))
    H[3:6, :3] = np.mat(np.eye(3))
    H[6:9, :3] = np.mat(np.eye(3))
    H[9:12, :3] = np.mat(np.eye(3))
    H[:3, 22:25] = R_O_W
    H[3:6, 25:28] = R_O_W
    H[6:9, 28:31] = R_O_W
    H[9:12, 31:34] = R_O_W
    return H


def get_bigH_PN(R_O_W):
    H = np.mat(np.zeros([24, 46]))
    H[:3, :3] = np.mat(np.eye(3))
    H[3:6, :3] = np.mat(np.eye(3))
    H[6:9, :3] = np.mat(np.eye(3))
    H[9:12, :3] = np.mat(np.eye(3))
    H[:3, 22:25] = R_O_W
    H[3:6, 25:28] = R_O_W
    H[6:9, 28:31] = R_O_W
    H[9:12, 31:34] = R_O_W
    H[12:15, 34:37] = R_O_W
    H[15:18, 37:40] = R_O_W
    H[18:21, 40:43] = R_O_W
    H[21:24, 43:46] = R_O_W
    return H


def get_bigF(F_part):
    """
    Input big_G^+ * big_J * delta_t
    Splice out big_F and output it
    """
    big_F = np.mat(np.eye(35))
    big_F[:6, 6:22] = F_part
    return big_F


def get_bigF_PN(F_part):
    """
    Input big_G^+ * big_J * delta_t
    Splice out big_F and output it
    """
    big_F = np.mat(np.eye(46))
    big_F[:6, 6:22] = F_part
    return big_F


def normalization(vec):
    """
    Normalize the vec
    """
    den = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    nor = vec / den
    return nor


# def calculate_cov(mat):
#     average = np.average(mat, axis=0)
#     substract = np.empty(mat.shape)
#     for i in range(mat.shape[0]):
#         substract[i] = mat[i] - average
#     cov = np.matmul(substract.T, substract) / (mat.shape[0] - 1)
#     return cov


def choose_taxel(pos_mean, pos_all):
    """
    Choose the taxel which is most close to the mean_position of taxels
    """
    d = np.abs(pos_all - pos_mean)
    print("       D-value of taxel:\n", pos_all, '\n', d)
    min = np.min(d)
    loc = np.where(d == min)[0][0]
    pos_chosen = pos_all[loc]
    print("       Choose taxel:", loc, pos_chosen)
    return pos_chosen, loc


def get_mean_3D(input_mat):
    mean_vac = np.zeros(3)
    if input_mat.size != 0:
        mean_vac = np.mean(input_mat, axis=0)
    return mean_vac


def get_T_from_poseuler(pos, euler):
    """
    Translate poseuler to T
    """
    R_taxl_in_tip = euler2R(euler)  # Taxel in tip
    T = np.mat(np.eye(4))
    T[:3, :3] = R_taxl_in_tip
    T[:3, 3] = np.mat(pos).T
    return T


def get_T_from_posrpy(pos, rpy):
    """
    Translate poseuler to T
    """
    R_taxl_in_tip = rpy2R(rpy)  # Taxel in tip
    T = np.mat(np.eye(4))
    T[:3, :3] = R_taxl_in_tip
    T[:3, 3] = np.mat(pos).T
    return T


def cal_col_in_J(A):
    """
    Calculate a col of Jacobian matrix
    The joint corresponds to this col must be revolute joint.
    """
    print("  A:  ", A)
    nx = A[0, 0]
    ny = A[1, 0]
    nz = A[2, 0]
    ox = A[0, 1]
    oy = A[1, 1]
    oz = A[2, 1]
    ax = A[0, 2]
    ay = A[1, 2]
    az = A[2, 2]
    px = A[0, 3]
    py = A[1, 3]
    pz = A[2, 3]
    col_J = np.mat(np.zeros([6, 1]))
    col_J[0, 0] = - nx * py + ny * px
    col_J[1, 0] = - ox * py + oy * px
    col_J[2, 0] = - ax * py + ay * px
    col_J[3, 0] = nz
    col_J[4, 0] = oz
    col_J[5, 0] = az
    return col_J


def get_taxel_poseuler(taxel_name):
    """
    Input taxel_name string, return pos and euler.
    """
    pos = np.zeros(3)
    euler = np.zeros(3)
    nodes = xml_root.findall('worldbody/body')
    for child in nodes:
        # print(child.tag, ":", child.attrib)
        if child.get('name') == {'name': 'box_link'}.get('name'):  # Get 'box_link' from all worldbody/body
            # If there is only one <body>, use find() or findall()
            # print("OK", child.tag, ":", child.attrib)
            childnodes = child.findall('body')[0]
            # print(childnodes.tag, ':', childnodes.attrib)
            childnodes1 = childnodes.findall('body')[0]
            # print(childnodes1.tag, ':', childnodes1.attrib)
            childnodes2 = childnodes1.findall('body')[0]
            # print(childnodes2.tag, ':', childnodes2.attrib)
            childnodes3 = childnodes2.findall('body')[0]
            # print(childnodes3.tag, ':', childnodes3.attrib)
            childnodes4 = childnodes3.findall('body')[0]
            # print(childnodes4.tag, ':', childnodes4.attrib)
            childnodes5 = childnodes4.findall('body')[0]
            # print(childnodes5.tag, ':', childnodes5.attrib)
            childnodes6 = childnodes5.findall('body')[0]
            # print(childnodes6.tag, ':', childnodes6.attrib)
            childnodes7 = childnodes6.findall('body')[0]
            # print(childnodes7.tag, ':', childnodes7.attrib)
            childnodes8 = childnodes7.findall('body')[0]
            # print("c8:", childnodes8.tag, ':', childnodes8.attrib)
            # childnodes9 = childnodes8.findall('body')[0]
            for childnodes9 in childnodes8:
                # print("c9:", childnodes9.tag, ':', childnodes9.attrib)
                # childnodes10 = childnodes9.findall('body')[0]
                for childnodes10 in childnodes9.findall('body'):
                    # print("c10", childnodes10.tag, ':', childnodes10.attrib)
                    for childnodes11 in childnodes10.findall('body'):
                        # print("  c11:", childnodes11.tag, ":", childnodes11.attrib)
                        if if_match(nodelist=childnodes11, name=taxel_name):
                            pos, euler = get_poseuler_from_xml(nodelist=childnodes11)
                            break
                        for childnodes12 in childnodes11.findall('body'):
                            # print("    c12", childnodes12.tag, ":", childnodes12.attrib)
                            for childnodes13 in childnodes12.findall('body'):
                                # print("      @13", childnodes13.tag, ":", childnodes13.attrib)
                                for childnodes14 in childnodes13.findall('body'):
                                    # print("        @@14", childnodes14.tag, ":", childnodes14.attrib)
                                    if if_match(nodelist=childnodes14, name=taxel_name):
                                        pos, euler = get_poseuler_from_xml(childnodes14)
                                        break
                                    for childnodes15 in childnodes14.findall('body'):
                                        # print("          @@@15", childnodes15.tag, ":", childnodes15.attrib)
                                        if if_match(nodelist=childnodes15, name=taxel_name):
                                            pos, euler = get_poseuler_from_xml(childnodes15)
                                            break
                                        for childnodes16 in childnodes15.findall('body'):
                                            # print("            $16", childnodes16.tag, ":", childnodes16.attrib)
                                            if if_match(nodelist=childnodes16, name=taxel_name):
                                                pos, euler = get_poseuler_from_xml(childnodes16)
                                                break
            break
    pos = np.fromstring(pos, dtype=float, sep=' ')
    euler = np.fromstring(euler, dtype=float, sep=' ')
    print('TOUCH_poseuler_array:', pos, euler)
    return pos, euler


def id2name_global(taxel_id):
    """
    Translate taxel id to taxel name.
    Not include palm part.
    """
    num_patch = ''
    name = 'empty'
    mode = 1  # mode 0: tip taxels; mode 1: pulp taxels
    if taxel_id < 72:  # 72
        num_patch = '0'
        mode = 0
    elif 72 <= taxel_id < 108:  # 36
        num_patch = '1'
    elif 108 <= taxel_id < 144:  # 36
        num_patch = '2'
    elif 144 <= taxel_id < 216:  # 72
        num_patch = '7'
        mode = 0
    elif 216 <= taxel_id < 252:  # 36
        num_patch = '8'
    elif 252 <= taxel_id < 288:  # 36
        num_patch = '9'
    elif 288 <= taxel_id < 360:  # 72
        num_patch = '11'
        mode = 0
    elif 360 <= taxel_id < 396:  # 36
        num_patch = '12'
    elif 396 <= taxel_id < 432:  # 36
        num_patch = '13'
    elif 432 <= taxel_id < 504:  # 72
        num_patch = '15'
        mode = 0
    elif 504 <= taxel_id < 540:  # 36
        num_patch = '16'
    # elif 540 <= id and id < 653:  # 113
    #     num_patch = '111'
    if mode == 0:
        name = id2name_tip(taxel_id, num_patch)
    elif mode == 1:
        name = id2name_pulp(taxel_id, num_patch)
    return name


def id2name_tip(taxel_id, patch_id):
    """
    Translate tip taxel id to tip taxel name.
    id range: 0:72
    """
    if taxel_id > 71:
        taxel_id = taxel_id % 72
    id_col = str(taxel_id % 6 + 1)
    id_row = str(taxel_id // 6 + 1)
    name = 'touch_' + patch_id + '_' + id_col + '_' + id_row
    return name


def id2name_pulp(taxel_id, patch_id):
    """
    Translate pulp taxel id to pulp taxel name.
    id range: 0:36
    """
    taxel_id = taxel_id % 36
    id_col = str(taxel_id // 6 + 1)
    id_row = str(taxel_id % 6 + 1)
    name = 'touch_' + patch_id + '_' + id_col + '_' + id_row
    return name


def vec2rot(vec):
    rot = np.zeros([3, 3])
    rot_x = np.zeros(3)
    x = vec[0]
    y = vec[1]
    rot_x[0] = y / (x**2 + y**2)**0.5
    rot_x[1] = - x / (x**2 + y**2)**0.5
    rot_y = np.cross(vec, rot_x)
    rot[:3, 0] = rot_x
    rot[:3, 1] = rot_y
    rot[:3, 2] = vec.T
    print("    vec==rot:", rot)
    return rot
