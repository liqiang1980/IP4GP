import numpy as np
# import tactile_perception as tacp
from scipy.spatial.transform import Rotation
from mujoco_py import load_model_from_path, MjSim, MjViewer, const
import PyKDL as kdl
import qgFunc as qg
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

# ['ff', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [0, 72], 'touch_0_3_6']
# ['ffd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [72, 108], 'touch_1_3_3']
# ['ffq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [108, 144], 'touch_2_3_3']
# ['mf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [144, 216], 'touch_7_3_6']
# ['mfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [216, 252], 'touch_8_3_3']
# ['mfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [252, 288], 'touch_9_3_3']
# ['rf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [288, 360], 'touch_11_3_6']
# ['rfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [360, 396], 'touch_12_3_3']
# ['rfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [396, 432], 'touch_13_3_3']
# ['th', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [432, 504], 'touch_15_3_6']
# ['thd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [504, 540], 'touch_16_3_3']
# ['palm', 1, {'j1': '-1', 'j2': '-1', 'j3': '-1', 'j4': '-1'}, [540, 635], 'touch_111_6_6']

filename = "../robots/UR5_tactile_allegro_hand.xml"
xml_path = filename
model = load_model_from_path(xml_path)
# sim = MjSim(model)
# viewer = MjViewer(sim)

# id_min = 0
# id_max = 72
# taxels_id = np.where(sim.data.sensordata[id_min: id_max] > 0.0)
# taxels_id = 38
# taxels_id = 86
# taxels_id = 122
# taxels_id = 182
# taxels_id = 230
# taxels_id = 266
# taxels_id = 326
# taxels_id = 374
# taxels_id = 410
# taxels_id = 470
# taxels_id = 518
# taxels_id = 604

# taxels_id = 62
# taxels_id = 144 + 57
taxels_id = 504 + 14
tac_name = model._sensor_id2name[taxels_id]

print(taxels_id, ": ", tac_name)

# quat1 = [0.01858882, 0.69386842, 0.02273803, 0.71950264]
# quat2 = [-0.03241207,  0.69586523, -0.02923444,  0.71684475]
# rotvec1 = Rotation.from_quat(quat1).as_rotvec()
# rotvec2 = Rotation.from_quat(quat2).as_rotvec()
# R1 = Rotation.from_quat(quat1).as_matrix()
# R2 = Rotation.from_quat(quat2).as_matrix()
# print(rotvec1)
# print(rotvec2)
# print(R1)
# print(R2)
# print("=======")
# rotvec_1 = [0.04109731, 1.53404728, 0.05027064]
# rotvec_2 = [-0.0717318, 1.54003334, -0.06469933]
# rotvec_22 = [0.0717318, 1.54003334, 0.06469933]
# print(" new 22:", Rotation.from_rotvec(rotvec_22).as_quat(), "\n", Rotation.from_rotvec(rotvec_22).as_matrix())

is_contact = {"ff": False, "ffd": False, "ffq": False,
              "mf": False, "mfd": False, "mfq": False,
              "rf": False, "rfd": False, "rfq": False,
              "th": False, "thd": True, "palm": False}
# print(any(list(is_contact.values())))

A = [np.array([[1, 2, 3],
              [1, 2, 3]])]
# AA = np.append(A, np.array([[[3, 2, 3],
#                              [3, 2, 3]]]), axis=0)
# AAA = np.append(AA, np.array([[[4, 3, 1],
#                                [4, 3, 1]]]), axis=0)
print(A)
# print(AA)
# print(AAA)

A.append(np.array([[9, 9, 9],
                   [9, 9, 9]]))
print(A)