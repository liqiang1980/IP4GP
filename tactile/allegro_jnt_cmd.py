import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import func as f
import tactile_allegro_mujo_const
import sys

def Camera_set():
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = model.stat.extent * 1.0
    viewer.cam.lookat[2] += .1
    viewer.cam.lookat[0] += .5
    viewer.cam.lookat[1] += .5
    viewer.cam.elevation = -0
    viewer.cam.azimuth = 0

def robot_init():
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_1] = 0.8
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_2] = -0.78
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_3] = 1.13
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_4] = -1.
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_5] = 0
    sim.data.ctrl[tactile_allegro_mujo_const.UR_CTRL_6] = -0.3

def hand_init():
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = -0.00502071
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = 0.2
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = 0.68513787
    sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = 0.85640426

#****************************上面是函数定义************************************#

finger_name = sys.argv[1]
jnt1_vel = sys.argv[2]
jnt2_vel = sys.argv[3]
jnt3_vel = sys.argv[4]
jnt4_vel = sys.argv[5]

xml_path = "../UR5/hand_only_scene.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

robot_init()
hand_init()
Camera_set()

for i in range(50):
    for _ in range(50):
        sim.step()
    viewer.render()

trans_cup = np.array([[ 9.99999989e-01, -7.91254649e-05, -1.23549804e-04, -5.03597532e-05],
 [ 7.91486867e-05,  9.99999979e-01,  1.87961042e-04,  8.00910847e-01],
 [ 1.23534929e-04, -1.87970818e-04,  9.99999975e-01,  8.64981304e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

trans_pregrasp = np.array([[0, 0, 1, 0.08],
                         [0, 1, 0, -0.22],
                         [-1, 0, 0, 0.01],
                         [0, 0, 0, 1]])

posequat = f.get_prepose_posequat(trans_cup, trans_pregrasp)
ctrl_wrist_pos = posequat[:3]
ctrl_wrist_quat = posequat[3:]

sim.model.eq_active[0] = True
for i in range(4):
    sim.data.mocap_pos[0] = ctrl_wrist_pos
    sim.data.mocap_quat[0] = ctrl_wrist_quat

    for _ in range(50):
        sim.step()
    viewer.render()

print("action start")

while True:
    sensor_data = sim.data.sensordata
    print(finger_name, ' incremental vel are ', jnt1_vel, jnt2_vel, jnt3_vel, jnt4_vel)
    # after the testing it seems that the joint limitation does not work
    q_pos = np.array([sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_1], \
                      sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_2], \
                      sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_3], \
                      sim.data.qpos[tactile_allegro_mujo_const.TH_MEA_4]])
    print(q_pos[0], q_pos[1], q_pos[2], q_pos[3])

    if finger_name == 'TH':
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_1] + float(jnt1_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_2] + float(jnt2_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_3] + float(jnt3_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.TH_CTRL_4] + float(jnt4_vel)
    elif finger_name == 'FF':
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_1] + float(jnt1_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_2] + float(jnt2_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_3] + float(jnt3_vel)
        sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] = \
            sim.data.ctrl[tactile_allegro_mujo_const.FF_CTRL_4] + float(jnt4_vel)
    elif finger_name == 'MF':
            print('mf')
    elif finger_name == 'RF':
            print('rf')
    else:
            print('wrong finger name in cmd line')


    contact = sim.data.contact

    sim.step()
    viewer.render()
