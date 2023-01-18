import threading

import finger_ctrl_func as fct
import config_param
import robot_control
import mujoco_environment as mu_env
import ekf
import tactile_perception
import tactile_plotter
import numpy as np
from threading import Thread
import sys, termios, tty, os, time
import matplotlib.pyplot as plt
import forward_kinematics
import tactile_allegro_mujo_const as tacConst


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
        tactile_plotter.LineChartAnimate_Obj(robo=robctrl, x_len=100, y_len=1,
                                             label1='x[mm]', label2='y[mm]', label3='z[mm]',
                                             label4='theta_x[deg]', label5='theta_y[deg]',
                                             label6='theta_z[deg]')
        plt.show()


class MainLoop(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ctrl_val = [{"ff": [0.0044, 0.000001, False],
                          "mf": [0.0065, 0.000001, False],
                          "rf": [0.0067, 0.000001, False],
                          "th": [0.002, 0.000001, False]
                          },  # case 0: cup free
                         {"ff": [0.0044, 0.000001, False],
                          "mf": [0.0065, 0.000001, False],
                          "rf": [0.0067, 0.000001, False],
                          "th": [0.002, 0.000001, False]
                          },  # case 1: cup frozen
                         {"ff": [0.0044, 0.000001, False],
                          "mf": [0.0065, 0.000001, False],
                          "rf": [0.0067, 0.000001, False],
                          "th": [0.002, 0.000001, False]
                          },  # case 2: cup upsidedown free
                         {"ff": [0.0041, 0.000001, False],
                          "mf": [0.0057, 0.000001, False],
                          "rf": [0.0055, 0.000001, False],
                          "th": [0.002, 0.000001, False]
                          },  # case 3: cylinder free
                         {"ff": [0.0041, 0.000001, False],
                          "mf": [0.0057, 0.000001, False],
                          "rf": [0.0055, 0.000001, False],
                          "th": [0.002, 0.000001, False]
                          },  # case 4: cylinder frozen
                         {"ff": [0.0041, 0.000001, False],
                          "mf": [0.0, 0.0, False],
                          "rf": [0.0, 0.0, False],
                          "th": [0.002, 0.000001, False],
                          },  # case 5: cup free, tips operation
                         {"ff": [0.0041, 0.00004, False],
                          "mf": [0.0, 0.0, False],
                          "rf": [0.0, 0.0, False],
                          "th": [-0.002, -0.00035, False],
                          },  # case 6: cup free, tips operation
                         {"ff": [-0.002, -0.00035, False],
                          "mf": [0.0, 0.0, False],
                          "rf": [0.0, 0.0, False],
                          "th": [0.0041, 0.00004, False],
                          },  # case 7: cup free, tips operation
                         ]

    def run(self):
        f_param = hand_param[1:]
        first_contact_flag = False

        # if not first_contact_flag:
        #     while True:
        #         # print("Pre set fingers")
        #         for f_part in f_param:
        #             f_name = f_part[0]
        #             tac_id = f_part[3]  # [min_id, max_id]
        #             if f_name in ctrl_val and not ctrl_val[f_name][2] and not (np.array(sim.data.sensordata[tac_id[0]: tac_id[1]]) > 0.0).any():
        #                 print(f_name, " step.")
        #                 fct.ctrl_finger_pre(sim=sim, f_part=f_part, input1=ctrl_val[f_name][0], stop=False)
        #             elif f_name in ctrl_val and not ctrl_val[f_name][2]:
        #                 print(f_name, "Contact!")
        #                 ctrl_val[f_name][2] = True
        #                 fct.ctrl_finger_pre(sim=sim, f_part=f_part, input1=0, stop=True)
        #         sim.step()
        #         viewer.render()
        #         del viewer._markers[:]
        #         if all([x[2] for x in list(ctrl_val.values())]):
        #             first_contact_flag = True
        #             break
        # print("Preset OK！")
        ctrl_val = self.ctrl_val[object_param[3]]

        fin_tran = 0
        _f_param_th = f_param[-1]
        # for ii in range(1500):
        for ii in range(700):
            if object_param[3] == 5 and (
                    np.array(sim.data.sensordata[0: tacConst.FF_TAXEL_NUM_MAX]) > 0.0).any() and not fin_tran:
                # print("?????????????", np.array(sim.data.sensordata[0: tacConst.FF_TAXEL_NUM_MAX]))
                ctrl_val = self.ctrl_val[object_param[3] + 1]
                fin_tran = ii + 150
            if fin_tran != 0 and ii == fin_tran:
                ctrl_val = self.ctrl_val[object_param[3] + 2]
            # print("uuuuuuuuuu:", ctrl_val)
            for f_part in f_param:
                f_name = f_part[0]
                tac_id = f_part[3]  # tac_id = [min, max]
                if f_name in ctrl_val:
                    fct.ctrl_finger(sim=sim, f_part=f_part, input1=ctrl_val[f_name][0], input2=ctrl_val[f_name][1])
            """EKF process"""
            if not first_contact_flag and (np.array(sim.data.sensordata[0: 636]) > 0.0).any():
                first_contact_flag = True
            if first_contact_flag:  # EKF Start
                print(robctrl.cnt_test, "EKF round:")
                robctrl.interaction(sim=sim, model=model, viewer=viewer,
                                    object_param=object_param,
                                    alg_param=alg_param,
                                    ekf_grasping=grasping_ekf,
                                    tacp=tacperception,
                                    fk=fk,
                                    char=char)
            """Update tacdata for heapmap plot"""
            # tacperception.update_tacdata(sim=sim)

            sim.step()
            viewer.render()
            del viewer._markers[:]

        """ Save Data for multiple exps """
        _FOLDER = ["res_free_cup", "res_frozen_cup", "res_upsidedown_cup", "res_free_cylinder", "res_frozen_cylinder",
                   "res_free_cup_tips"]
        FOLDER = _FOLDER[object_param[3]]
        CNT = [np.loadtxt(FOLDER + "/CNT.txt")]
        print("CNT:", CNT)
        if CNT[0]:  # >>>>> Not first saving
            data_cur = np.loadtxt("x_state_all.txt")
            data_all = list(np.load(FOLDER + "/x_state_repeatEXP.npy", allow_pickle=True))
            print("data_cur shape:", len(data_cur), len(data_cur[0]))
            print("data_all shape:", len(data_all), len(data_all[0]), len(data_all[0][0]))
            data_all.append(data_cur)
            print("after append, data_all shape:", len(data_all), len(data_all[0]), len(data_all[0][0]))
            np.save(FOLDER + "/x_state_repeatEXP.npy", data_all)
        else:  # >>>>> First saving
            data_cur = [np.loadtxt("x_state_all.txt")]
            print(data_cur, len(data_cur), len(data_cur[0]), len(data_cur[0][0]))
            np.save(FOLDER + "/x_state_repeatEXP.npy", data_cur)
        np.savetxt(FOLDER + "/CNT.txt", [CNT[0] + 1])
        # save gd values
        gd_cur = np.loadtxt("x_gt_palm.txt")
        print(" gd save shape:", FOLDER + "x_gt_palm.txt   ", gd_cur.shape)
        np.savetxt(FOLDER + "/x_gt_palm.txt", gd_cur)


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# button_delay = 0.2


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
xml_path = "../../robots/UR5_tactile_allegro_hand.xml"
if int(object_param[3]) == 1:
    xml_path = "../../robots/UR5_tactile_allegro_hand_obj_frozen.xml"
    # model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_frozen.xml")
elif int(object_param[3]) == 2:
    xml_path = "../../robots/UR5_tactile_allegro_hand_obj_upsidedown.xml"
    # model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_obj_upsidedown.xml")
elif int(object_param[3]) == 3:
    xml_path = "../../robots/UR5_tactile_allegro_hand_cylinder.xml"
    # model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_cylinder.xml")
elif int(object_param[3]) == 4:
    xml_path = "../../robots/UR5_tactile_allegro_hand_cylinder_frozen.xml"
    # model, sim, viewer = mu_env.init_mujoco("../../robots/UR5_tactile_allegro_hand_cylinder_frozen.xml")
# else:
#     model, sim, viewer = mu_env.init_mujoco()
model, sim, viewer = mu_env.init_mujoco(filename=xml_path)

ctrl_wrist_pos, ctrl_wrist_quat = \
    mu_env.init_robot_object_mujoco(sim, object_param)
mu_env.config_fcl("cup_1.obj", "fingertip_part.obj")

""" Instantiate FK class """
fk = forward_kinematics.ForwardKinematics(hand_param=hand_param)
""" Instantiate ekf class """
grasping_ekf = ekf.EKF()
grasping_ekf.set_contact_flag(False)
grasping_ekf.set_store_flag(alg_param[0])

tacperception = tactile_perception.cls_tactile_perception(xml_path=xml_path, fk=fk)

# init robot
robctrl = robot_control.ROBCTRL(obj_param=object_param, hand_param=hand_param, model=model, xml_path=xml_path, fk=fk)
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
tacperception.fin_tri = np.zeros(len(hand_param) - 1)

# Thumb root movement
fct.pre_thumb(sim, viewer)
# char = "v"
char = "i"

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
