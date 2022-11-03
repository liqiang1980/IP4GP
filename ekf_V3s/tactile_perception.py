import numpy as np
import tactile_allegro_mujo_const
import util_geometry as ug
import object_geometry as og
import time


class taxel_pose:
    def __init__(self):
        self.position = np.array([0, 0, 0])
        self.orientation = np.identity(3)


class cls_tactile_perception:
    def __init__(self):
        self.c_point_name = []
        # contact finger number
        self.fin_num = 0
        # identify which fingers are contacted
        self.fin_tri = np.zeros(4)
        # tuple variable to tell (contact_finger_name, reference_frame, contacted_pose)
        self.tuple_fin_ref_pose = ()
        self.is_ff_contact = False
        self.is_mf_contact = False
        self.is_rf_contact = False
        self.is_th_contact = False
        self.fftip_center_taxel_pose = [0., 0., 0., 0., 0., 0., 0.]
        self.mftip_center_taxel_pose = [0., 0., 0., 0., 0., 0., 0.]
        self.rftip_center_taxel_pose = [0., 0., 0., 0., 0., 0., 0.]
        self.thtip_center_taxel_pose = [0., 0., 0., 0., 0., 0., 0.]
        self.tacdata_ff = np.zeros([12, 6])
        self.tacdata_mf = np.zeros([12, 6])
        self.tacdata_rf = np.zeros([12, 6])
        self.tacdata_th = np.zeros([12, 6])
        """
        The 4 array correspond to tip,mdp,mqp,palm part. 
        The 4 id correspond to ff,mf,rf,th respectively. There is only 1 id in palm part. 
        id=-1 means no contact.
        """
        self.mapping = {"ff": 0, "mf": 1, "rf": 2, "th": 3}
        self.tacdata_ztid = [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], -1]
        self.tacdata_htid = [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], -1]

    def update_tacdata(self, sim):
        """
        Updata tac data: 4 tips, 12*6 for each
        tmp_Af: ff, mf, rf, th: 1*72
        """
        tmp_Af = []
        mapping_const = {0: [tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN, tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX],
                         1: [tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN, tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX],
                         2: [tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN, tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX],
                         3: [tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN, tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX],
                         }
        """ Get tacdata from sim """
        for i in range(4):
            tmp_Af.append(np.array(sim.data.sensordata[mapping_const[i][0]: mapping_const[i][1]]))
            tmp_Af[i] = np.int64(tmp_Af[i] > 0) * 2  # Binarize tac data
            """ Put z_t into tacdata """
            if self.tacdata_ztid[0][i] != -1:  # Is contact
                tmp_Af[i][self.tacdata_ztid[0][i] - mapping_const[i][0]] = -4
            """ Put h_t into tacdata """
            if self.tacdata_htid[0][i] != -1:  # Is contact
                tmp_Af[i][self.tacdata_htid[0][i] - mapping_const[i][0]] = -2
            # print("tac_z:\n", self.tacdata_ztid, "\ntac_h:\n", self.tacdata_htid)
        """ Ready to plot tacdata """
        for i in range(11, -1, -1):
            self.tacdata_ff[i] = tmp_Af[0][i * 6: i * 6 + 6][::-1]
            self.tacdata_mf[i] = tmp_Af[1][i * 6: i * 6 + 6][::-1]
            self.tacdata_rf[i] = tmp_Af[2][i * 6: i * 6 + 6][::-1]
            self.tacdata_th[i] = tmp_Af[3][i * 6: i * 6 + 6][::-1]

    def get_hand_tip_center_pose(self, sim, model, ref_frame):
        self.get_tip_center_pose(sim, model, 'ff_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'mf_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'rf_tip', ref_frame)
        self.get_tip_center_pose(sim, model, 'th_tip', ref_frame)

    def is_finger_contact(self, sim, finger_name):
        if finger_name == 'ff':
            if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
                    tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any() == True:
                self.is_ff_contact = True
                return True
            else:
                self.is_ff_contact = False
                return False
        if finger_name == 'mf':
            if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
                    tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any() == True:
                self.is_mf_contact = True
                return True
            else:
                self.is_mf_contact = False
                return False
        if finger_name == 'rf':
            if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
                    tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any() == True:
                self.is_rf_contact = True
                return True
            else:
                self.is_rf_contact = False
                return False
        if finger_name == 'th':
            if (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
                    tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any() == True:
                self.is_th_contact = True
                return True
            else:
                self.is_th_contact = False
                return False

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

    def get_contact_taxel_id(self, sim, finger_name):
        # print("|||shape||||sensordata: ", len(sim.data.sensordata))

        if finger_name == 'ff':
            return np.where(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
                                                tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX] > 0.0)
        if finger_name == 'mf':
            return np.where(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
                                                tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX] > 0.0)
        if finger_name == 'rf':
            return np.where(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
                                                tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX] > 0.0)
        if finger_name == 'th':
            return np.where(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
                                                tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX] > 0.0)

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

    def get_contact_taxel_name(self, sim, model, fingername, z_h_flag):
        """
        Always use the contact point closest to the center position of contact area.
        """
        taxels_id = self.get_contact_taxel_id(sim, fingername)

        if fingername == 'ff':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
            if len(c_points) == 0:  # No contact
                return 'link_3.0_tip'
        if fingername == 'mf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
            if len(c_points) == 0:  # No contact
                return 'link_7.0_tip'
        if fingername == 'rf':
            c_points = taxels_id[0] + tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
            if len(c_points) == 0:  # No contact
                return 'link_11.0_tip'
        if fingername == 'th':  # No contact
            c_points = taxels_id[0] + tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN
            if len(c_points) == 0:
                return 'link_15.0_tip'

        actived_tmp_position = np.zeros((3, len(c_points)))
        taxel_position = np.zeros((3, 72))
        active_taxel_name = []
        dev_taxel_value = []
        """ Get average position of contact points """
        if len(c_points) > 1:
            print(fingername + " pose compute taxels: ", end='')
            for i in range(len(c_points)):
                active_taxel_name.append(model._sensor_id2name[c_points[i]])
                print(model._sensor_id2name[c_points[i]] + ' ', end='')
                actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
            avg_position = actived_tmp_position.mean(1)
            print('')
        else:
            avg_position = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[c_points[0]])[:3]

        """ Get the name of contact point closest to avg_position """

        if len(c_points) > 1:
            for i in range(len(c_points)):
                taxel_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
                # The norm() defaults to 2-norm, that is, distance
                dev_taxel_value.append(np.linalg.norm(taxel_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            id_chosen = c_points[dev_taxel_value.index(min_value)]
            c_point_name = model._sensor_id2name[id_chosen]
        else:
            id_chosen = -1
            c_point_name = model._sensor_id2name[c_points[0]]
        """ Update tacdata_z or tacdata_h """
        if z_h_flag == "z":
            self.tacdata_ztid[0][self.mapping[fingername]] = id_chosen
        elif z_h_flag == "h":
            self.tacdata_htid[0][self.mapping[fingername]] = id_chosen

        return c_point_name

    # get the median of all contact_nums, and translate to contact_name
    # def get_c_point_name(self, model, c_points):
    #     if len(c_points) % 2 == 0:  # even number of contact points
    #         c_points = np.hstack((-1, c_points))  # change to odd
    #         #todo why the median of taxels are the contact point?
    #         c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
    #         print(np.median(c_points))
    #     else:
    #         c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
    #         print(np.median(c_points))
    #
    #     return c_point_name
    #
    # # get contact_name using the average contact position and min distance to the taxel
    # def get_avg_c_point(self, sim, model, c_points, fingername):
    #     actived_tmp_position = np.zeros((3, len(c_points)))
    #     taxel_position = np.zeros((3, 72))
    #     active_taxel_name = []
    #     dev_taxel_value = []
    #     for i in range(len(c_points)):
    #         active_taxel_name.append(model._sensor_id2name[c_points[i]])
    #         print(active_taxel_name[i])
    #         actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
    #     avg_pressure = actived_tmp_position.mean(1)
    #
    #     if fingername == 'ff':
    #         min_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
    #         max_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX
    #     if fingername == 'mf':
    #         min_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
    #         max_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX
    #     if fingername == 'rf':
    #         min_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
    #         max_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX
    #     if fingername == 'th':
    #         min_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN
    #         max_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX
    #
    #     for i in range(min_taxel_id, max_taxel_id):
    #         # taxel_array_name.append()
    #         print(model._sensor_id2name[i])
    #         taxel_position[:, i - min_taxel_id] = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[i])[:3]
    #         dev_taxel_value.append(np.linalg.norm(taxel_position[:, i - min_taxel_id] - avg_pressure))
    #
    #     min_value = min(dev_taxel_value)
    #     print(dev_taxel_value.index(min_value))
    #     c_point_name = model._sensor_id2name[dev_taxel_value.index(min_value) + min_taxel_id]
    #     return c_point_name

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

    # get normal vector direction
    def get_contact_taxel_nv(self, sim, model, fingername, ref_frame):
        c_point_name = self.get_contact_taxel_name(sim, model, fingername, "z")
        # get the position
        pos_contact = ug.get_relative_posquat(sim, ref_frame, c_point_name)

        T_contact = ug.posquat2trans(pos_contact)
        # nv = T_contact[:3, 2]  # Get R of contact point
        # attention here, normal direciton is x axis.
        nv = T_contact[:3, 0]  # Get R of contact point
        return nv
