#!/usr/bin/env python
# demonstration of markers (visual-only geoms)
import util_geometry as ug
from mujoco_py import const
from scipy.spatial.transform import Rotation

import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

def cor_frame_visual(viewer, position, mat_rot, length, frame_name):
    x_rot = ug.vec2rot(mat_rot[:, 0])
    y_rot = ug.vec2rot(mat_rot[:, 1])
    z_rot = ug.vec2rot(mat_rot[:, 2])
    viewer.add_marker(pos=position, mat=x_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.005, 0.005, length]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=y_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.005, 0.005, length]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    viewer.add_marker(pos=position, mat=z_rot, type=const.GEOM_ARROW, label=frame_name,
                      size=np.array([0.005, 0.005, length]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <asset>
        <!--Meshes-->
        <mesh name="cup"      file="/home/qiang/IP4GP/robots/mesh/visual/cup_1.STL"   scale="0.001 0.001 0.001"/>
        <mesh name="bottle"      file="/home/qiang/IP4GP/robots/mesh/visual/bottlerough.STL"   scale="0.001 0.001 0.001"/>
        <texture name="gripper_tex" type="2d"       builtin="flat" height="32" width="32" rgb1="0.45 0.45 0.45" rgb2="0 0 0"/>
        <material name="gripper_mat"    texture="gripper_tex"   shininess="0.9" specular="0.75" reflectance="0.4" />
    </asset>
    <worldbody>
        <!--<body name="box" pos="0 0 0.2">
            <geom size="0.15 0.15 0.15" type="box"/>
            <joint axis="1 0 0" name="box:x" type="slide"/>
            <joint axis="0 1 0" name="box:y" type="slide"/>
        </body>-->
        <body name="floor" pos="0 0 0">
            <geom size="1.0 1.0 0.001" rgba="0 1 0 1" type="box"/>
        </body>
        <body name="cup" pos=" 0.0 0.8 0.86"  >
            <site name="cup" pos="0 0 0 " euler="0 0 0"/>
            <geom type="mesh" material="gripper_mat" mesh="cup" euler="0 0 0" friction="1 0.1 0.5"/>
            <inertial pos="0 0 0" mass="0.3" diaginertia="0.0035 0.0035 0.0035" />
        </body>
    </worldbody>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
while True:
    iden_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    position = [0.0, 0.0, 0.0]

    g1 = ug.get_relative_posquat(sim, "world", "cup")
    p, o = ug.posquat2pos_p_o(g1)

    viewer.add_marker(pos=p, mat=o, type=7,
                      label='cup', rgba=np.array([1.0, 0.0, 0.0, 1.0]), dataid=0)

    # viewer.add_marker(pos=np.array([0, 0, 0.]), type=7,
    #                   label='bottle', rgba=np.array([1.0, 0.0, 1.0, 1.0]), dataid=1)

    viewer.add_marker(pos=np.array([0, 0, 0.3]), mat=iden_rot, type=6,
                      label='cube', size=np.array([0.01, 0.01, 0.1]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))

    # cor_frame_visual(viewer, position, iden_rot, 0.3, 'frame')



    # g2 = ug.get_body_posquat(sim, "cup")
    #
    # pos = sim.data.get_body_xpos("cup")
    # print('position ', pos)
    # quat = sim.data.get_body_xquat("cup")
    # quat = np.hstack((quat[1:], quat[0]))  # Change to x y z w
    # cup_rot = Rotation.from_quat(quat).as_matrix()
    # print('orientation ', cup_rot)



    T_obj_world = ug.posquat2trans(g1)
    pos_obj_world = T_obj_world[:3, 3].T
    rot_obj_world = T_obj_world[:3, :3]
    cor_frame_visual(viewer, pos_obj_world, rot_obj_world, 0.2, "Obj")

    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
