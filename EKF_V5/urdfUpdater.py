import os

import numpy as np
from mujoco_py import load_model_from_path
from urdf_parser_py.urdf import URDF
import pathlib
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import xml.etree.ElementTree as ET
import qgFunc as qg

file_dir_part = './allegro_hand_description/urdf_part/tactile_part.urdf'
xml_path = "./UR5/UR5_allegro_test.xml"
model = load_model_from_path(xml_path)
xml_tree = ET.parse(xml_path)
xml_root = xml_tree.getroot()


def urdf_updater(urdf_file):
    """
    Write taxels named in 'touch_x_x_x' format into Allegro_hand urdf model.
    """
    joint_num = np.array([3, 7, 11, 15])
    patch_num = np.array([0, 7, 11, 15])
    tip_name = np.array(['index_tip_tactile', 'mid_tip_tactile', 'ring_tip_tactile', 'thumb_tip_tactile'])
    for i in range(4):  # Write patches on 4 tips : tip_3, tip_7, tip_11, tip_15
        urdf_file.write('<link name="link_' + str(joint_num[i]) + '.0_tip_tactile">\n'
                        + '</link>\n'
                        + '<joint name="joint_' + str(joint_num[i]) + '.0_tip_tactile" type="fixed">\n'
                        + '  <parent link="' + tip_name[i] + '"/>\n'
                        + '  <child link="link_' + str(joint_num[i]) + '.0_tip_tactile"/>\n'
                        + '  <origin rpy="0 0 0" xyz="0.009 0 0.016"/>\n'
                        + '</joint>\n')
    for i in range(4):
        for j in range(1, 7):
            for k in range(1, 13):
                taxel_name = 'touch_' + str(patch_num[i]) + '_' + str(j) + '_' + str(k)
                pos, euler = qg.get_taxel_poseuler(taxel_name=taxel_name, xml_root=xml_root)
                str_tmp = '<link name=\"' + taxel_name + '\">\n' \
                          + '  <visual>\n' + '    <geometry>\n' \
                          + '      <cylinder length=\"0.0007\" radius=\"0.0011\"/>\n' \
                          + '    </geometry>\n' + \
                          '    <origin rpy=\"0 1.57 0\" xyz=\"0 0 0\"/>\n' \
                          + '    <material name=\"blue\">\n' \
                          + '      <color rgba=\"0 0 1 1\"/>\n' \
                          + '    </material>\n' \
                          + '  </visual>\n' \
                          + '</link>\n' \
                          + '<joint name=\"link_' + str(
                    joint_num[i]) + '.0_tip_tactile_to_' + taxel_name + '\" type=\"fixed\">\n' \
                          + '  <origin euler=\"' + euler + '\" xyz=\"' + pos + '\"/>\n' \
                          + '  <parent link=\"link_' + str(joint_num[i]) + '.0_tip_tactile\"/>\n' \
                          + '  <child link=\"' + taxel_name + '\"/>\n' \
                          + '</joint>\n'
                print(str_tmp)
                urdf_file.write(str_tmp)


def read_file_as_str(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")
    all_the_text = open(file_path).read()
    return all_the_text


def makeURDFcomplete():
    part_hand = read_file_as_str('./allegro_hand_description/urdf_part/allegro_hand_right_part.urdf')
    part_tacxel = read_file_as_str('./allegro_hand_description/urdf_part/tactile_part.urdf')

    target = open('./allegro_hand_description/allegro_hand_tactile_v1.4.urdf', mode='w')
    target.write(str(part_hand))
    target.write(str(part_tacxel))

    target.write("</robot>")
    print("URDF OK!")
    target.close()


if __name__ == '__main__':
    urdf_part = open(file_dir_part, mode='w')
    urdf_updater(urdf_part)
    urdf_part.close()

    makeURDFcomplete()
