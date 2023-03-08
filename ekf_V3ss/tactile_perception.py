import numpy as np
import tactile_allegro_mujo_const
import tactile_allegro_mujo_const as tacCONST
import util_geometry as ug
import object_geometry as og
import qgFunc as qg
import time
from scipy.spatial.transform import Rotation
from copy import deepcopy


class taxel_pose:
    def __init__(self):
        self.position = np.array([0, 0, 0])
        self.orientation = np.identity(3)


class cls_tactile_perception:
    def __init__(self, xml_path, fk):
        self.xml_path = xml_path
        self.fk = fk
        self.c_point_name = []
        self.tuple_fin_ref_pose = ()
        """Record which f_part is contact in current interaction round"""
        self.is_contact = {"ff": False, "ffd": False, "ffq": False,
                           "mf": False, "mfd": False, "mfq": False,
                           "rf": False, "rfd": False, "rfq": False,
                           "th": False, "thd": False, "palm": False}
        self.is_contact_recent = {"ff": False, "ffd": False, "ffq": False,
                                  "mf": False, "mfd": False, "mfq": False,
                                  "rf": False, "rfd": False, "rfq": False,
                                  "th": False, "thd": False, "palm": False}
        """Record which f_part is contact in the first interaction round"""
        self.is_first_contact = {"ff": False, "ffd": False, "ffq": False,
                                 "mf": False, "mfd": False, "mfq": False,
                                 "rf": False, "rfd": False, "rfq": False,
                                 "th": False, "thd": False, "palm": False}
        """Record last tac and cur tac: tac_name , position & rotvec """
        self.last_tac = {"ff": ["touch_0_3_6", np.zeros(6)], "ffd": ["touch_1_3_3", np.zeros(6)],
                         "ffq": ["touch_2_3_3", np.zeros(6)],
                         "mf": ["touch_7_3_6", np.zeros(6)], "mfd": ["touch_8_3_3", np.zeros(6)],
                         "mfq": ["touch_9_3_3", np.zeros(6)],
                         "rf": ["touch_11_3_6", np.zeros(6)], "rfd": ["touch_12_3_3", np.zeros(6)],
                         "rfq": ["touch_13_3_3", np.zeros(6)],
                         "th": ["touch_15_3_6", np.zeros(6)], "thd": ["touch_16_3_3", np.zeros(6)],
                         "palm": ["touch_111_6_6", np.zeros(6)]}
        self.cur_tac = {"ff": ["touch_0_3_6", np.zeros(6)], "ffd": ["touch_1_3_3", np.zeros(6)],
                        "ffq": ["touch_2_3_3", np.zeros(6)],
                        "mf": ["touch_7_3_6", np.zeros(6)], "mfd": ["touch_8_3_3", np.zeros(6)],
                        "mfq": ["touch_9_3_3", np.zeros(6)],
                        "rf": ["touch_11_3_6", np.zeros(6)], "rfd": ["touch_12_3_3", np.zeros(6)],
                        "rfq": ["touch_13_3_3", np.zeros(6)],
                        "th": ["touch_15_3_6", np.zeros(6)], "thd": ["touch_16_3_3", np.zeros(6)],
                        "palm": ["touch_111_6_6", np.zeros(6)]}
        """
        The 4 array correspond to tip,mdp,mqp,palm part. 
        The 4 id correspond to ff,mf,rf,th respectively. There is only 1 id in palm part. 
        id=-1 means no contact.
        """
        self.mapping = {"ff": 0, "mf": 1, "rf": 2, "th": 3}
        self.tacdata_ztid = [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], -1]
        self.tacdata_htid = [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], -1]

    def get_hand_tip_center_pose(self, sim, model, ref_frame):
        self.get_tip_center_pose(sim, model, 'ff_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'mf_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'rf_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'th_tip', ref_frame)

    def is_fingers_contact(self, basicData, f_param, fk):
        for f_part in f_param:
            self.is_finger_contact(basicData=basicData, f_part=f_part, fk=fk)
        """ If no any contact, use contacts in last round """
        # if not any(list(self.is_contact.values())):
        #     self.is_contact = self.is_contact_recent

    def is_finger_contact(self, basicData, f_part, fk):
        """
        Detect if this finger part contacts.
        Yes:
            Update cur_tac_name by contact area center tac method.
            Update cur_tac_pos by fk.
            Update cur_tac_rotvec by default jnt
        No：
        """
        f_name = f_part[0]
        min_id = f_part[3][0]
        max_id = f_part[3][1]
        print("  ", f_name)
        # if (np.array(basicData.taxel_data[min_id: max_id]) > 0.0).any():
        if (np.array(basicData.taxel_data[min_id: max_id]) > 0.0).any():
            """ Update contact_flags """
            self.is_contact[f_name] = True
            """ Update contact_tac name & position & rotvec """
            tac_name, tac_id = self.get_contact_taxel_name(basicData=basicData, f_part=f_part)
            pv_tac_palm = self.Cur_tac_renew(tac_name=tac_name, f_part=f_part, fk=fk)
            self.cur_tac[f_name][0] = tac_name
            self.cur_tac[f_name][1] = pv_tac_palm
            print(">>", f_name, " contact: ", tac_name, "\n")
            return True
        else:  # No contact
            print("      No contact")
            self.is_contact[f_name] = False
            """ If not contact, use last tac to update cur tac info """
            tac_name = self.last_tac[f_name][0]
            pv_tac_palm = self.Cur_tac_renew(tac_name=tac_name, f_part=f_part, fk=fk)
            self.cur_tac[f_name][0] = tac_name
            self.cur_tac[f_name][1] = pv_tac_palm
            return False

    def Last_tac_renew(self, f_param):
        """
        Update last_tac by cur_tac.
        This func always used in the tail of one EKF round.
        """
        for f_part in f_param:
            f_name = f_part[0]
            self.last_tac[f_name][0] = deepcopy(self.cur_tac[f_name][0])
            self.last_tac[f_name][1] = deepcopy(self.cur_tac[f_name][1])
        self.is_contact_recent = deepcopy(self.is_contact)

    def Cur_tac_renew(self, tac_name, f_part, fk):
        """
        Update cur_tac by name.
        Get pos, rpy, Trans of [cur_tac in tip frame] from xml file.
        Get Trans of [tip in palm frame] from fk calculation in the head of cur EKF round.
        """
        f_name = f_part[0]
        pos_tac_palm, rotvec_tac_palm, T_tac_palm = fk.get_relative_posrot(tac_name=tac_name, f_name=f_name, xml_path=self.xml_path)
        self.cur_tac[f_name][0] = tac_name
        self.cur_tac[f_name][1][:3] = pos_tac_palm
        self.cur_tac[f_name][1][3:] = rotvec_tac_palm
        return self.cur_tac[f_name][1]  # return posrotvec

    def get_contact_taxel_name_pressure(self, sim, model, finger_name):
        taxels_id = self.get_contact_taxel_id(sim, finger_name)
        if finger_name == 'ff':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
        if finger_name == 'mf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
        if finger_name == 'rf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
        if finger_name == 'th':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN

        actived_tmp_position = np.zeros((3, len(c_points)))
        active_taxel_presssure = []
        taxel_position = np.zeros((3, 72))
        active_taxel_name = []
        dev_taxel_value = []
        if len(c_points) > 1:
            for i in range(len(c_points)):
                active_taxel_presssure.append(sim.data.sensordata[c_points[i]])
                active_taxel_name.append(model._sensor_id2name[c_points[i]])
                actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
            avg_position = actived_tmp_position.mean(1)
        else:
            avg_position = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[c_points[0]])[:3]
        c_avr_pressure = sum(active_taxel_presssure) / len(active_taxel_presssure)
        if len(c_points) > 1:
            for i in range(len(c_points)):
                taxel_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
                dev_taxel_value.append(np.linalg.norm(taxel_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            c_point_name = model._sensor_id2name[c_points[dev_taxel_value.index(min_value)]]
        else:
            c_point_name = model._sensor_id2name[c_points[0]]

        return c_point_name, c_avr_pressure

    def get_contact_feature(self, sim, model, finger_name):
        taxels_id = self.get_contact_taxel_id(sim, finger_name)

        if finger_name == 'ff':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
        if finger_name == 'mf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
        if finger_name == 'rf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
        if finger_name == 'th':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN

        actived_tmp_position = np.zeros((3, len(c_points)))
        active_taxel_presssure = []
        taxel_position = np.zeros((3, 72))
        active_taxel_name = []
        dev_taxel_value = []
        if len(c_points) > 1:
            for i in range(len(c_points)):
                active_taxel_presssure.append(sim.data.sensordata[c_points[i]])
                active_taxel_name.append(model._sensor_id2name[c_points[i]])
                actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
            avg_position = actived_tmp_position.mean(1)
            c_avr_pressure = sum(active_taxel_presssure) / len(active_taxel_presssure)
        else:
            avg_position = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[c_points[0]])[:3]
            c_avr_pressure = 0.0
        if len(c_points) > 1:
            for i in range(len(c_points)):
                taxel_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
                dev_taxel_value.append(np.linalg.norm(taxel_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            c_point_name = model._sensor_id2name[c_points[dev_taxel_value.index(min_value)]]
        else:
            c_point_name = model._sensor_id2name[c_points[0]]

        if finger_name == 'ff':
            c_point_pose = ug.get_relative_posquat(sim, "link_3.0_tip", c_point_name)
        if finger_name == 'mf':
            c_point_pose = ug.get_relative_posquat(sim, "link_7.0_tip", c_point_name)
        if finger_name == 'rf':
            c_point_pose = ug.get_relative_posquat(sim, "link_11.0_tip", c_point_name)
        if finger_name == 'th':
            c_point_pose = ug.get_relative_posquat(sim, "link_15.0_tip", c_point_name)

        return c_point_pose, c_point_name, c_avr_pressure

    def get_normal(self, sim, model, c_points, trans_cup2palm):
        pos_contact_avg_cupX = np.empty([1, 0])
        pos_contact_avg_cupY = np.empty([1, 0])
        pos_contact_avg_cupZ = np.empty([1, 0])
        for i in c_points:
            c_point_name_zz = model._sensor_id2name[i]
            # todo why here the cartesian mean is used, not in the contact position computation (pos_contact0)
            posquat_contact_cup_zz = ug.get_relative_posquat(sim, "cup", c_point_name_zz)
            pos_contact_avg_cupX = np.append(pos_contact_avg_cupX, posquat_contact_cup_zz[0])
            pos_contact_avg_cupY = np.append(pos_contact_avg_cupY, posquat_contact_cup_zz[1])
            pos_contact_avg_cupZ = np.append(pos_contact_avg_cupZ, posquat_contact_cup_zz[2])

        # get mean position:
        pos_contact_avg_cup = np.array(
            [pos_contact_avg_cupX.mean(), pos_contact_avg_cupY.mean(), pos_contact_avg_cupZ.mean()])
        #############################  Get normal of contact point on the cup   ########################################
        nor_contact_in_cup, res = og.surface_cup(pos_contact_avg_cup[0], pos_contact_avg_cup[1],
                                                 pos_contact_avg_cup[2])
        nor_in_p = np.matmul(trans_cup2palm, np.hstack((nor_contact_in_cup, 1)).T).T[:3]
        # Normalization:
        den = (nor_in_p[0] ** 2 + nor_in_p[1] ** 2 + nor_in_p[2] ** 2) ** 0.5
        nn_nor_contact_in_p = np.array(
            [nor_in_p[0] / den, nor_in_p[1] / den, nor_in_p[2] / den])
        print("normal after normalization:", nn_nor_contact_in_p)

        return nn_nor_contact_in_p, res

    def get_tip_center_pose(self, sim, model, fingertipname, ref_frame):
        taxels_position = np.zeros((3, 72))
        taxels_name = []
        dev_taxel_value = []

        if fingertipname == 'ff_tip':
            # attention avg_position is varied because the taxels_position is changing.
            for i in range(72):
                taxels_name.append(model._sensor_id2name[i])
                taxels_position[:, i] = ug.get_relative_posquat(sim, "link_3.0_tip", taxels_name[i])[:3]
            avg_position = taxels_position.mean(1)

            for i in range(72):
                dev_taxel_value.append(np.linalg.norm(taxels_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            center_taxel_name = model._sensor_id2name[dev_taxel_value.index(min_value)]
            self.fftip_center_taxel_pose = ug.get_relative_posquat(sim, ref_frame, center_taxel_name)

        if fingertipname == 'mf_tip':
            for i in range(72):
                taxels_name.append(model._sensor_id2name[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN + i])
                taxels_position[:, i] = ug.get_relative_posquat(sim, "link_7.0_tip", taxels_name[i])[:3]
            avg_position = taxels_position.mean(1)

            for i in range(72):
                dev_taxel_value.append(np.linalg.norm(taxels_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            center_taxel_name = model._sensor_id2name[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN \
                                                      + dev_taxel_value.index(min_value)]
            self.mftip_center_taxel_pose = ug.get_relative_posquat(sim, ref_frame, center_taxel_name)

        if fingertipname == 'rf_tip':
            for i in range(72):
                taxels_name.append(model._sensor_id2name[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN + i])
                taxels_position[:, i] = ug.get_relative_posquat(sim, "link_11.0_tip", taxels_name[i])[:3]
            avg_position = taxels_position.mean(1)

            for i in range(72):
                dev_taxel_value.append(np.linalg.norm(taxels_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            center_taxel_name = model._sensor_id2name[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN + \
                                                      dev_taxel_value.index(min_value)]
            self.rftip_center_taxel_pose = ug.get_relative_posquat(sim, ref_frame, center_taxel_name)

        if fingertipname == 'th_tip':
            for i in range(72):
                taxels_name.append(model._sensor_id2name[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN + i])
                taxels_position[:, i] = ug.get_relative_posquat(sim, "link_15.0_tip", taxels_name[i])[:3]
            avg_position = taxels_position.mean(1)

            for i in range(72):
                dev_taxel_value.append(np.linalg.norm(taxels_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            center_taxel_name = model._sensor_id2name[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN + \
                                                      dev_taxel_value.index(min_value)]

            self.thtip_center_taxel_pose = ug.get_relative_posquat(sim, ref_frame, center_taxel_name)

        return ug.get_relative_posquat(sim, ref_frame, center_taxel_name)

    def get_contact_taxel_id_withoffset(self, sim, fingername):
        taxels_id = self.get_contact_taxel_id(sim, fingername)

        if fingername == 'ff':
            c_points = taxels_id[0]

        if fingername == 'mf':
            c_points = taxels_id[0] + 144

        if fingername == 'rf':
            c_points = taxels_id[0] + 288

        if fingername == 'th':
            c_points = taxels_id[0] + 432
        return c_points

    def get_contact_taxel_id(self, basicData, f_part):
        id_min = f_part[3][0]
        id_max = f_part[3][1]
        return np.where(np.array(basicData.taxel_data[id_min: id_max]) > 0.0)

    def get_contact_taxel_position(self, sim, model, fingername, ref_frame, z_h_flag):
        """
        Get the position of the contact taxel in the reference frame.
        Always use the contact point closest to the center position of contact area.
        """
        # get the name
        c_point_name = self.get_contact_taxel_name(sim, model, fingername, z_h_flag)
        # get the position
        pos_contact = ug.get_relative_posquat(sim, ref_frame, c_point_name)
        tmp_list = []
        tmp_list.append(fingername)
        tmp_list.append(ref_frame)
        tmp_list.append(c_point_name)
        tmp_list.append(pos_contact)
        self.tuple_fin_ref_pose = tuple(tmp_list)
        return pos_contact

    def get_contact_taxel_name(self, basicData, f_part):
        """
        Always use the contact point closest to the center position of contact area.
        """
        f_name = f_part[0]
        tac_id = f_part[3]  # tac_id = [min, max]
        default_tac_name = f_part[4][0]
        default_tac_id = f_part[4][1]
        taxels_id = self.get_contact_taxel_id(basicData=basicData, f_part=f_part)
        if taxels_id is None:  # No contact
            print("～～～～It is weird to get Here～～～～")  # No-contact cases were already avoided before
            return default_tac_name, default_tac_id
        c_points = taxels_id[0] + tac_id[0]
        print("taxels_id:", taxels_id)

        # tac_pos_all = []
        # for tid in range(tac_id[0], tac_id[1], 1):
        #     tac_name = basicData.model._sensor_id2name[tid]
        #     # print("tid:", tid)
        #     # pos_tac_palm, rotvec_tac_palm, T_tac_palm = self.fk.get_relative_posrot(tac_name=tac_name,
        #     #                                                                         f_name=f_name,
        #     #                                                                         xml_path=self.xml_path)
        #     pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name, xml_path=self.xml_path)
        #     # tac_pos_all.append(pos_tac_palm)
        #     tac_pos_all.append(pos_tac_tip)
        active_tac_tip_name = []
        active_tac_tip_pos = np.zeros((len(c_points), 3))
        dev_taxel_value = []
        """ Get average position of contact points """
        if len(c_points) > 1:
            # print(f_name + " pose compute taxels: ", end='')
            for i, c_point in enumerate(c_points):
                # print(basicData.model._sensor_id2name[c_point] + ' ', end='')
                tac_name = basicData.model._sensor_id2name[c_point]
                # pos_tac_palm, rotvec_tac_palm, T_tac_palm = self.fk.get_relative_posrot(tac_name=tac_name,
                #                                                                         f_name=f_name,
                #                                                                         xml_path=self.xml_path)
                pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name, xml_path=self.xml_path)
                active_tac_tip_name.append(tac_name)
                active_tac_tip_pos[i] = pos_tac_tip
            avg_pos_tac_tip = active_tac_tip_pos.mean(0)
            # print(active_tac_pos, avg_position)
            # print('')
        else:
            # print("c_pints[0]:", c_points[0])
            tac_name0 = basicData.model._sensor_id2name[c_points[0]]
            # pos_tac_palm, rotvec_tac_palm, T_tac_palm = self.fk.get_relative_posrot(tac_name=tac_name0,
            #                                                                         f_name=f_name,
            #                                                                         xml_path=self.xml_path)
            pos_tac_tip, rpy_tac_tip = qg.get_taxel_poseuler(taxel_name=tac_name0, xml_path=self.xml_path)
            avg_pos_tac_tip = pos_tac_tip

        """ Get the name of contact point closest to avg_position """
        if len(c_points) > 1:
            # for i in range(len(c_points)):
            #     tac_pos_all[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
            #     # The norm() defaults to 2-norm, that is, distance
            #     dev_taxel_value.append(np.linalg.norm(tac_pos_all[:, i] - avg_position))
            for i, _tac_tip_pos in enumerate(basicData.tac_tip_pos[f_name]):
                # print("_tac_pos, avg_position:", _tac_pos, avg_position, np.array(_tac_pos) - avg_position)
                dev_taxel_value.append(np.linalg.norm(np.array(_tac_tip_pos) - avg_pos_tac_tip))
            min_value = min(dev_taxel_value)
            id_chosen = dev_taxel_value.index(min_value) + tac_id[0]
            c_point_name = basicData.model._sensor_id2name[id_chosen]
            # print(dev_taxel_value, "\nmin:", min_value, "  id_choose:", id_chosen, " c_point_name:", c_point_name)
        else:
            id_chosen = -1
            c_point_name = basicData.model._sensor_id2name[c_points[0]]
        return c_point_name, id_chosen

    def get_contact_taxel_position_from_name(self, sim, model, fingername, ref_frame, c_point_name):
        # get the position
        pos_contact = ug.get_relative_posquat(sim, ref_frame, c_point_name)
        tmp_list = []
        tmp_list.append(fingername)
        tmp_list.append(ref_frame)
        tmp_list.append(c_point_name)
        tmp_list.append(pos_contact)
        self.tuple_fin_ref_pose = tuple(tmp_list)
        return pos_contact

    def get_contact_taxel_nv(self, sim, model, fingername, ref_frame):
        """
        get normal vector direction
        """
        c_point_name = self.get_contact_taxel_name(sim, model, fingername)
        # get the position
        pos_contact = ug.get_relative_posquat(sim, ref_frame, c_point_name)  # wxyz
        T_contact = ug.posquat2trans(pos_contact)
        # nv = T_contact[:3, 2]  # Get R of contact point
        # attention here, normal direciton is x axis.
        nv = T_contact[:3, 0]  # Get R of contact point
        return nv
