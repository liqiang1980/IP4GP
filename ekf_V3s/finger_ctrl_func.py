import tactile_allegro_mujo_const as tacCONST
import numpy as np


def robot_init(sim):
    UR_ctrl = {tacCONST.UR_CTRL_1: 0.8,
               tacCONST.UR_CTRL_2: -0.78,
               tacCONST.UR_CTRL_3: 1.13,
               tacCONST.UR_CTRL_4: -1.,
               tacCONST.UR_CTRL_5: 0,
               tacCONST.UR_CTRL_6: -0.3
               }
    for ur_ctrl_id in UR_ctrl:
        sim.data.ctrl[ur_ctrl_id] = UR_ctrl[ur_ctrl_id]


def ctrl_finger(sim, input1, input2, f_part):
    """
    One finger control
    """
    f_name = f_part[0]
    tac_id = f_part[3]  # tac_id = [min, max]
    # tac_id = {"ff": [tacCONST.FF_TAXEL_NUM_MIN, tacCONST.FF_TAXEL_NUM_MAX],
    #           "mf": [tacCONST.MF_TAXEL_NUM_MIN, tacCONST.MF_TAXEL_NUM_MAX],
    #           "rf": [tacCONST.RF_TAXEL_NUM_MIN, tacCONST.RF_TAXEL_NUM_MAX],
    #           "th": [tacCONST.TH_TAXEL_NUM_MIN, tacCONST.TH_TAXEL_NUM_MAX]
    #           }
    ctrl_id = {"ff": [tacCONST.FF_CTRL_2, tacCONST.FF_CTRL_3, tacCONST.FF_CTRL_4],
               "mf": [tacCONST.MF_CTRL_2, tacCONST.MF_CTRL_3, tacCONST.MF_CTRL_4],
               "rf": [tacCONST.RF_CTRL_2, tacCONST.RF_CTRL_3, tacCONST.RF_CTRL_4],
               "th": [tacCONST.TH_CTRL_2, tacCONST.TH_CTRL_3, tacCONST.TH_CTRL_4]
               }
    _input = 0
    if not (np.array(sim.data.sensordata[tac_id[0]: tac_id[1]]) > 0.0).any():
        _input = input1
    else:
        _input = input2
    for cid in ctrl_id[f_name]:
        sim.data.ctrl[cid] += _input


# def ctrl_finger_pre(sim, input1, f_part, stop):
#     """
#     One finger control
#     """
#     f_name = f_part[0]
#     # tac_id = f_part[3]  # tac_id = [min, max]
#     ctrl_id = {"ff": [tacCONST.FF_CTRL_2, tacCONST.FF_CTRL_3, tacCONST.FF_CTRL_4],
#                "mf": [tacCONST.MF_CTRL_2, tacCONST.MF_CTRL_3, tacCONST.MF_CTRL_4],
#                "rf": [tacCONST.RF_CTRL_2, tacCONST.RF_CTRL_3, tacCONST.RF_CTRL_4],
#                "th": [tacCONST.TH_CTRL_2, tacCONST.TH_CTRL_3, tacCONST.TH_CTRL_4]
#                }
#     _input = input1
#     for cid in ctrl_id[f_name]:
#         sim.data.ctrl[cid] += _input
#         if stop:
#             sim.data.ctrl[cid] = 0


def pre_thumb(sim, viewer):
    for _ in range(1000):
        sim.data.ctrl[tacCONST.TH_CTRL_1] += 0.05
        sim.step()
        viewer.render()

def index_finger(sim, input1, input2):
    # print("|||shape||||ctrl: ", len(sim.data.ctrl))

    if not (np.array(sim.data.sensordata[tacCONST.FF_TAXEL_NUM_MIN: \
            tacCONST.FF_TAXEL_NUM_MAX]) > 0.0).any():
        sim.data.ctrl[tacCONST.FF_CTRL_2] = \
            sim.data.ctrl[tacCONST.FF_CTRL_2] + input1
        sim.data.ctrl[tacCONST.FF_CTRL_3] = \
            sim.data.ctrl[tacCONST.FF_CTRL_3] + input1
        sim.data.ctrl[tacCONST.FF_CTRL_4] = \
            sim.data.ctrl[tacCONST.FF_CTRL_4] + input1
    else:
        sim.data.ctrl[tacCONST.FF_CTRL_2] = \
            sim.data.ctrl[tacCONST.FF_CTRL_2] + input2
        sim.data.ctrl[tacCONST.FF_CTRL_3] = \
            sim.data.ctrl[tacCONST.FF_CTRL_3] + input2
        sim.data.ctrl[tacCONST.FF_CTRL_4] = \
            sim.data.ctrl[tacCONST.FF_CTRL_4] + input2

# def middle_finger(sim, input1, input2):
#     if not (np.array(sim.data.sensordata[tacCONST.MF_TAXEL_NUM_MIN:
#     tacCONST.MF_TAXEL_NUM_MAX]) > 0.0).any():  # 中指
#         sim.data.ctrl[tacCONST.MF_CTRL_2] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_2] + input1
#         sim.data.ctrl[tacCONST.MF_CTRL_3] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_3] + input1
#         sim.data.ctrl[tacCONST.MF_CTRL_4] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_4] + input1
#     else:
#
#         sim.data.ctrl[tacCONST.MF_CTRL_2] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_2] + input2
#         sim.data.ctrl[tacCONST.MF_CTRL_3] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_3] + input2
#         sim.data.ctrl[tacCONST.MF_CTRL_4] = \
#             sim.data.ctrl[tacCONST.MF_CTRL_4] + input2
#
#
# def ring_finger(sim, input1, input2):
#     if not (np.array(sim.data.sensordata[tacCONST.RF_TAXEL_NUM_MIN: \
#             tacCONST.RF_TAXEL_NUM_MAX]) > 0.0).any():  # 小拇指
#         sim.data.ctrl[tacCONST.RF_CTRL_2] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_2] + input1
#         sim.data.ctrl[tacCONST.RF_CTRL_3] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_3] + input1
#         sim.data.ctrl[tacCONST.RF_CTRL_4] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_4] + input1
#     else:
#         sim.data.ctrl[tacCONST.RF_CTRL_2] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_2] + input2
#         sim.data.ctrl[tacCONST.RF_CTRL_3] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_3] + input2
#         sim.data.ctrl[tacCONST.RF_CTRL_4] = \
#             sim.data.ctrl[tacCONST.RF_CTRL_4] + input2
#
#
# def thumb(sim, input1, input2):
#     if not (np.array(sim.data.sensordata[tacCONST.TH_TAXEL_NUM_MIN: \
#             tacCONST.TH_TAXEL_NUM_MAX]) > 0.0).any():  # da拇指
#         sim.data.ctrl[tacCONST.TH_CTRL_2] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_2] + input1
#         sim.data.ctrl[tacCONST.TH_CTRL_3] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_3] + input1
#         sim.data.ctrl[tacCONST.TH_CTRL_4] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_4] + input1
#     else:
#         sim.data.ctrl[tacCONST.TH_CTRL_2] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_2] + input2
#         sim.data.ctrl[tacCONST.TH_CTRL_3] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_3] + input2
#         sim.data.ctrl[tacCONST.TH_CTRL_4] = \
#             sim.data.ctrl[tacCONST.TH_CTRL_4] + input2
