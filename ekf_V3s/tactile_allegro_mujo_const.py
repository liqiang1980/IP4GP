"""
IN 'UR5_tactile_allegro_hand.xml' :
len(sim.data.ctrl) = 22
    Defined in the end of xml: <actuator> --> 6*<motor>,16*<position>

len(sim.data.sensordata) = 653
    Defined in the end of xml: <sensor> --> 653*<touch>

len(sim.data.qpos) = 682 (653+22+7, cup is a freejoint)
    <freejoint> use 7 qpos (Attention: )
    <joint> use 1 qpos


IN 'UR5_tactile_allegro_hand_obj_frozen.xml' :
len(sim.data.ctrl) = 22
    Defined in the end of xml: <actuator> --> 6*<motor>,16*<position>

len(sim.data.sensordata) = 653
    Defined in the end of xml: <sensor> --> 653*<touch>

len(sim.data.qpos) = 675 (653+22)
"""

UR_CTRL_1 = 0
UR_CTRL_2 = 1
UR_CTRL_3 = 2
UR_CTRL_4 = 3
UR_CTRL_5 = 4
UR_CTRL_6 = 5

FF_CTRL_1 = 6
FF_CTRL_2 = 7
FF_CTRL_3 = 8
FF_CTRL_4 = 9


MF_CTRL_1 = 10
MF_CTRL_2 = 11
MF_CTRL_3 = 12
MF_CTRL_4 = 13

RF_CTRL_1 = 14
RF_CTRL_2 = 15
RF_CTRL_3 = 16
RF_CTRL_4 = 17

TH_CTRL_1 = 18
TH_CTRL_2 = 19
TH_CTRL_3 = 20
TH_CTRL_4 = 21

# if object (as a joint) is free type, then OFF_SET = 0
# if object (as a joint) is hinge type, then OFF_SET = 6
# if object (as a joint) is ball type, then OFF_SET = 3
# if object (as a joint) is no type, then OFF_SET = 7


OFF_SET = 0
FF_MEA_1 = 126-OFF_SET
FF_MEA_2 = 127-OFF_SET
FF_MEA_3 = 164-OFF_SET
FF_MEA_4 = 201-OFF_SET

MF_MEA_1 = 274-OFF_SET
MF_MEA_2 = 275-OFF_SET
MF_MEA_3 = 312-OFF_SET
MF_MEA_4 = 349-OFF_SET

RF_MEA_1 = 422-OFF_SET
RF_MEA_2 = 423-OFF_SET
RF_MEA_3 = 460-OFF_SET
RF_MEA_4 = 497-OFF_SET

TH_MEA_1 = 570-OFF_SET
TH_MEA_2 = 571-OFF_SET
TH_MEA_3 = 572-OFF_SET
TH_MEA_4 = 573-OFF_SET


FF_TAXEL_NUM_MIN = 0
FF_TAXEL_NUM_MAX = 72
MF_TAXEL_NUM_MIN = 144
MF_TAXEL_NUM_MAX = 216
RF_TAXEL_NUM_MIN = 288
RF_TAXEL_NUM_MAX = 360
TH_TAXEL_NUM_MIN = 432
TH_TAXEL_NUM_MAX = 504

GEOM_ARROW = 100
GEOM_BOX = 6
FULL_FINGER_JNTS_NUM = 16

PN_FLAG = 'p'  # Observation controller: assign 'p' (position) or 'pn' (position and normal)
GT_FLAG = '1G'  # G Matrix controller: assign '1G' (splice a big G, then pinv) or '4G' (inv 4 GT, then splice)
posteriori_FLAG = True
# posteriori_FLAG = False
initE_FLAG = True
betterJ_FLAG = True
solver_ik_type_wdls = False
