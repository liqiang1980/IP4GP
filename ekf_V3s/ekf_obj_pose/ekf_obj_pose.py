import threading

import finger_ctrl_func as fct
import config_param
import robot_control as robctrl
import mujoco_environment as mu_env
import ekf
import tactile_perception
import tactile_plotter
import numpy as np
from threading import Thread
import sys, termios, tty, os, time
import matplotlib.pyplot as plt


class HeatmapLoop(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        ani_Heatmap = tactile_plotter.HeatmapAnimate_Dip(tacperception)
        plt.show()


class LineChartLoop(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        tactile_plotter.LineChartAnimate_Obj(rob_control, 100, 1,
                                             label1='x[mm]', label2='y[mm]', label3='z[mm]',
                                             label4='theta_x[deg]', label5='theta_y[deg]',
                                             label6='theta_z[deg]')
        plt.show()


class MainLoop(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ctrl_val = {"ff": [0.005, 0.000001],
                    "mf": [0.005, 0.000001],
                    "rf": [0.005, 0.000001],
                    "th": [0.002, 0.000001]
                    }
        for ii in range(2000):
            for f_part in hand_param[1:]:
                f_name = f_part[0]
                used_flag = f_part[1]
                if f_name in ctrl_val and used_flag == "1":
                    fct.ctrl_finger(sim=sim, input1=ctrl_val[f_name][0], input2=ctrl_val[f_name][1], f_name=f_name)
            # if hand_param[1][1] == '1':
            #     fct.ctrl_finger(sim, 0.005, 0.000001, hand_param[1][0])
            # if hand_param[2][1] == '1':
            #     # fct.middle_finger(sim, 0.005, 0.000001)
            #     fct.ctrl_finger(sim, 0.005, 0.000001, hand_param[2][0])
            # if hand_param[3][1] == '1':
            #     # fct.ring_finger(sim, 0.005, 0.000001)
            #     fct.ctrl_finger(sim, 0.005, 0.000001, hand_param[3][0])
            # if hand_param[4][1] == '1':
            #     # fct.thumb(sim, 0.002, 0.000001)
            #     fct.ctrl_finger(sim, 0.002, 0.000001, hand_param[4][0])
            """EKF process"""
            rob_control.interaction(sim, model, viewer, hand_param, object_param, alg_param, grasping_ekf,
                                    tacperception, char)

            """Update tacdata for heapmap plot"""
            tacperception.update_tacdata(sim=sim)

            sim.step()
            viewer.render()
            del viewer._markers[:]


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
elif int(object_param[3]) == 2:
    model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_upsidedown.xml")
elif int(object_param[3]) == 3:
    model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_cylinder.xml")
elif int(object_param[3]) == 4:
    model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_cylinder_frozen.xml")
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
rob_control = robctrl.ROBCTRL()
fct.robot_init(sim)
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
tacperception.fin_tri = np.zeros(len(hand_param)-1)

# Thumb root movement
fct.pre_thumb(sim, viewer)
char = "v"
# char = "i"

#######################################################################
ekfer = MainLoop()
ekfer.start()
# ani_Heatmap = tactile_plotter.HeatmapAnimate_Dip(tacperception)
# ani_LineChart = tactile_plotter.LineChartAnimate_Obj(rob_control, 100, 1,
#                                                      label1='x[mm]', label2='y[mm]', label3='z[mm]',
#                                                      label4='theta_x[deg]', label5='theta_y[deg]',
#                                                      label6='theta_z[deg]')
# plt.show()

# tactile_plotter.AllAnimate(tacperception=tacperception, robo=rob_control, x_len=100, y_len=1, label1='x[mm]',
#                            label2='y[mm]', label3='z[mm]',
#                            label4='theta_x[deg]', label5='theta_y[deg]',
#                            label6='theta_z[deg]')
# plt.show()
