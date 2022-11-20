import threading

import config_param
import robot_control as robcontrol
import mujoco_environment as mu_env
import ekf
import tactile_perception
import tactile_heatmap
import numpy as np
from threading import Thread
import sys, termios, tty, os, time
import matplotlib.pyplot as plt
import test_Plot_plus as pplt


class MainLoop(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # for ii in range(2000):
        for ii in range(400):
            if hand_param[1][1] == '1':
                rob_control.index_finger(sim, 0.005, 0.000001)
            if hand_param[2][1] == '1':
                rob_control.middle_finger(sim, 0.005, 0.000001)
            if hand_param[3][1] == '1':
                rob_control.ring_finger(sim, 0.005, 0.000001)
            if hand_param[4][1] == '1':
                rob_control.thumb(sim, 0.002, 0.000001)
            """EKF process"""
            rob_control.interaction(sim, model, viewer, hand_param, object_param, alg_param, grasping_ekf, tacperception, char)

            """Update tacdata for heapmap plot"""
            tacperception.update_tacdata(sim=sim)

            sim.step()
            viewer.render()
            del viewer._markers[:]
        tacperception.tac_track_data()
        tactile_heatmap.plot_once(tacperception)


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


button_delay = 0.2


def function02(arg, name):
    global char
    while True:
        char = getch()

        if (char == "p"):
            print("Stop!")
            exit(0)


# load task-related parameters
hand_param, object_param, alg_param = config_param.pass_arg()

# init mujoco environment
if int(object_param[3]) == 1:
    model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_frozen.xml")
else:
    model, sim, viewer = mu_env.init_mujoco()

ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)
mu_env.config_fcl("cup_1.obj", "fingertip_part.obj")

# instantiate ekf class
grasping_ekf = ekf.EKF()
grasping_ekf.set_contact_flag(False)
grasping_ekf.set_store_flag(alg_param[0])

tacperception = tactile_perception.cls_tactile_perception()

# init robot
rob_control = robcontrol.ROBCTRL()
rob_control.robot_init(sim)
mu_env.Camera_set(viewer, model)
sim.model.eq_active[0] = True

for i in range(500):
    sim.step()
    viewer.render()

# move robotic arm to pre-grasping posture
sim.data.mocap_pos[0] = ctrl_wrist_pos
sim.data.mocap_quat[0] = ctrl_wrist_quat
for _ in range(50):
    sim.step()
    viewer.render()

    # start interaction
    # number of triggered fingers
tacperception.fin_num = 0
# The fingers which are triggered are Marked them with "1"
tacperception.fin_tri = np.zeros(4)

# Thumb root movement
rob_control.pre_thumb(sim, viewer)
char = "v"
# char = "i"

#######################################################################
ekfer = MainLoop()
ekfer.start()
# tactile_heatmap.TacPlotClass_dip(tacperception)
# plt.show()

