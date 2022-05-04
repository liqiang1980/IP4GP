import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import func as f

# sim.data.ctrl[0] = 0
# sim.data.ctrl[1] = -1.18
# sim.data.ctrl[2] = 1.06
# sim.data.ctrl[3] = -0.35
# sim.data.ctrl[9] = -0.1

# xml_path = "/home/bidan/project/adaptivegrasp/UR5/UR5gripper.xml"
xml_path = "../../UR5/UR5_tactile_allegro_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
# sim.model.eq_active[7] = False

sim.step()
print(f.get_body_posquat(sim, "gripperfinger_1_link_3"))


viewer = MjViewer(sim)

radius = 0.03
scale = 0.003
pos_bottle = f.get_body_posquat(sim, "bottle")
ctrl_wrist = pos_bottle[:3] + [0.04, -0.6, 0.125]

# sim.model.eq_active[7] = True
for i in range(100):
    sim.data.mocap_pos[0] = ctrl_wrist
    for _ in range(50):
        sim.step()
    viewer.render()

print('ctrl', ctrl_wrist,'wrist_3_link', f.get_body_posquat(sim, "wrist_3_link"))
print('ctrl', ctrl_wrist, 'mocap ', sim.data.mocap_pos[0])

adaptive = 1
#ctrl_wrist = pos_bottle[:3] + [0.04-0.02, -0.314, 0.125] #non-adaptive, side
ctrl_wrist = pos_bottle[:3] + [0.04-0.03, -0.32, 0.155] #adaptive, side
# ctrl_wrist = pos_bottle[:3] + [0.04-0.03, -0.34, 0.155] #adaptive fail, side
# ctrl_wrist = pos_bottle[:3] + [0.04-0.003, -0.312, 0.155] #non-adaptive, middle
goal = pos_bottle[:3] + [0.04 - 0.02, -0.308, 0.155]


for i in range(100):
    sim.data.mocap_pos[0] = ctrl_wrist
    for _ in range(50):
        sim.step()
    viewer.render()


# sim.data.mocap_pos[0] = f.get_body_posquat(sim, "wrist_3_link")[:3]
# sim.data.mocap_quat[0] = f.get_body_posquat(sim, "wrist_3_link")[3:]
print('goal', goal)
print("wrist posquat", f.get_body_posquat(sim, "wrist_3_link"))
print("bottle posquat", f.get_body_posquat(sim, "bottle"))

numzeros = [0, 0, 0]
alltouch = 0
graspjnt = [0, 0, 0]
while True:
    # print('time:', sim.data.time)
    # sim.data.ctrl[0] = 0
    # sim.data.ctrl[1] = -1.18
    # sim.data.ctrl[2] = 1.06
    # sim.data.ctrl[3] = -0.35
    # sim.data.ctrl[9] = -0.1
    sim.data.mocap_pos[0] += np.array([0, 0, 0.0])
    sim.data.mocap_quat[0] += np.array([0., 0., 0., 0.])

    tactile_force = sim.data.sensordata[7::3]
    contact_threshold = 5e-5
    binary_tactile = np.where(np.abs(tactile_force) > contact_threshold, 1, 0)
    touch = [0, 0, 0]

    # for idx, item in enumerate(tactile_force):
    #     if np.abs(item) > contact_threshold:
    #         finger_idx = int(np.floor(idx / 16))
    #         touch[finger_idx] = 1

    for i in range(3):
        numzeros[i] += 1

    for idx in range(16):
        if np.abs(tactile_force[idx]) > contact_threshold:
            touch[0] = 1
            numzeros[0] = 0
            break

    for idx in range(16,32):
        if np.abs(tactile_force[idx]) > contact_threshold:
            touch[1] = 1
            numzeros[1] = 0
            break

    for idx in range(32,48):
        if np.abs(tactile_force[idx]) > contact_threshold:
            touch[2] = 1
            numzeros[2] = 0
            break

    if alltouch >= 5:
        alltouch = 1000
    else:
        alltouch = (alltouch + 1) * np.prod(touch)
        # alltouch += np.prod(touch)

    if alltouch < 1e3:
        for i in range(3):
            if adaptive == 0:
                sim.data.ctrl[i + 6] = sim.data.ctrl[i + 6] + 0.005
            else:
                if touch[i] == 0:
                    sim.data.ctrl[i+6] = sim.data.ctrl[i+6] + 0.0005 * numzeros[i]
                else:
                    sim.data.ctrl[i+6] = sim.data.ctrl[i+6] - 0.00001

                    pos_wrist = f.get_body_posquat(sim, "wrist_3_link")
                    # pos_bottle = f.get_body_posquat(sim, "bottle")
                    # ctrl_wrist = pos_wrist[:3] + (goal - pos_wrist[:3])/10
                    ctrl_wrist = sim.data.mocap_pos[0] + (goal - sim.data.mocap_pos[0])/5
                    sim.data.mocap_pos[0] = ctrl_wrist
                    print('ctrl_wrist', np.round(ctrl_wrist[:3], 4),'pos_wrist', np.round(pos_wrist[:3], 4),
                          'goal', np.round(goal[:3], 4), '+', np.round(goal-pos_wrist[:3],4))

            graspjnt = sim.data.ctrl[6:9]
            graspee = f.get_body_posquat(sim, "wrist_3_link")
            # ctrl_wrist = graspee[:3] + [0, 0, 0.1]
    else:
        # sim.data.ctrl[1] = -1.5
        sim.data.ctrl[6:9] = graspjnt + [0.0025, 0.0025, 0.005]
        sim.data.mocap_pos[0] += [0, 0, 0.001]

    # for i in range(3):
    #     sim.data.ctrl[i+6] = sim.data.ctrl[i+6] + 0.001

    # sim.data.ctrl[8] = sim.data.time/100

    # skip frame
    for _ in range(50):
        sim.step()

    viewer.render()

    # print('qpos ', np.round(sim.data.qpos, 2))
    # print('binary_tactile', binary_tactile)

    print('touch ', touch, 'alltouch', alltouch)
    # print(f.get_body_posquat(sim, "bottle"))
    # print('sim.data.ctrl', sim.data.ctrl[6:9] )
    # print('numzeros', numzeros)
    # print('alltouch', alltouch)



