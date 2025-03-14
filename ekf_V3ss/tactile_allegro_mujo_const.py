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

# OFF_SET = 7
OFF_SET = 0
FF_MEA_1 = 126 - OFF_SET
FF_MEA_2 = 127 - OFF_SET
FF_MEA_3 = 164 - OFF_SET
FF_MEA_4 = 201 - OFF_SET

MF_MEA_1 = 274 - OFF_SET
MF_MEA_2 = 275 - OFF_SET
MF_MEA_3 = 312 - OFF_SET
MF_MEA_4 = 349 - OFF_SET

RF_MEA_1 = 422 - OFF_SET
RF_MEA_2 = 423 - OFF_SET
RF_MEA_3 = 460 - OFF_SET
RF_MEA_4 = 497 - OFF_SET

TH_MEA_1 = 570 - OFF_SET
TH_MEA_2 = 571 - OFF_SET
TH_MEA_3 = 572 - OFF_SET
TH_MEA_4 = 573 - OFF_SET

FF_TAXEL_NUM_MIN = 0
FF_TAXEL_NUM_MAX = 72
FFd_TAXEL_NUM_MIN = 72
FFd_TAXEL_NUM_MAX = 108
FFq_TAXEL_NUM_MIN = 108
FFq_TAXEL_NUM_MAX = 144
MF_TAXEL_NUM_MIN = 144
MF_TAXEL_NUM_MAX = 216
MFd_TAXEL_NUM_MIN = 216
MFd_TAXEL_NUM_MAX = 252
MFq_TAXEL_NUM_MIN = 252
MFq_TAXEL_NUM_MAX = 288
RF_TAXEL_NUM_MIN = 288
RF_TAXEL_NUM_MAX = 360
RFd_TAXEL_NUM_MIN = 360
RFd_TAXEL_NUM_MAX = 396
RFq_TAXEL_NUM_MIN = 396
RFq_TAXEL_NUM_MAX = 432
TH_TAXEL_NUM_MIN = 432
TH_TAXEL_NUM_MAX = 504
THd_TAXEL_NUM_MIN = 504
THd_TAXEL_NUM_MAX = 540
PALM_TAXEL_NUM_MIN = 540
PALM_TAXEL_NUM_MAX = 63

GEOM_ARROW = 100
GEOM_BOX = 6
FULL_FINGER_JNTS_NUM = 16
TAC_TOTAL_NUM = 635

PN_FLAG = 'p'  # Observation controller: assign 'p' (position) or 'pn' (position and normal)
GT_FLAG = '1G'  # G Matrix controller: assign '1G' (splice a big G, then pinv) or '4G' (inv 4 GT, then splice)
posteriori_FLAG = True
# posteriori_FLAG = False
initE_FLAG = True
# initE_FLAG = False
# betterJ_FLAG = True
solver_ik_type_wdls = False

# repeatEXP = 25
repeatEXP = 5
#######################################################################################################################
txt_dir = [
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/One_fin/1/",  # 0
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/One_fin/2/",  # 1
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/Two_fin/1/",  # 2
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/Two_fin/2/",  # 3
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/Four_fin/1/",  # 4
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/Four_fin/2/",  # 5
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/largeshaker/Four_fin/3/",  # 6
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/One_fin/1/",  # 7
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/One_fin/2/",  # 8
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/One_fin/3/",  # 9
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Two_fin/1/",  # 10
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Two_fin/2/",  # 11
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Two_fin/3/",  # 12
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Four_fin/1/",  # 13
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Four_fin/2/",  # 14
    "/home/lqg/catkin_ws/src/EKF_txt_publisher/data/bottle/Four_fin/3/"  # 15
]
#######################################################################################################################
