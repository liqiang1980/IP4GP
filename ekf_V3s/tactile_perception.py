import numpy as np
import tactile_allegro_mujo_const
import util_geometry as ug
import object_geometry as og
import time

class cls_tactile_perception:
    def __init__(self):
        c_point_name = []
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

    def get_contact_taxel_id(self, sim, finger_name):
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

    def get_normal(self, sim, model, c_points, trans_cup2palm):
        pos_contact_avg_cupX = np.empty([1, 0])
        pos_contact_avg_cupY = np.empty([1, 0])
        pos_contact_avg_cupZ = np.empty([1, 0])
        for i in c_points:
            c_point_name_zz = model._sensor_id2name[i]
            #todo why here the cartesian mean is used, not in the contact position computation (pos_contact0)
            posquat_contact_cup_zz = ug.get_relative_posquat(sim, "cup", c_point_name_zz)
            pos_contact_avg_cupX = np.append(pos_contact_avg_cupX, posquat_contact_cup_zz[0])
            pos_contact_avg_cupY = np.append(pos_contact_avg_cupY, posquat_contact_cup_zz[1])
            pos_contact_avg_cupZ = np.append(pos_contact_avg_cupZ, posquat_contact_cup_zz[2])

        # get mean position:
        pos_contact_avg_cup = np.array([pos_contact_avg_cupX.mean(), pos_contact_avg_cupY.mean(), pos_contact_avg_cupZ.mean()])
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

    def get_contact_taxel_name(self, sim, model, fingername):
        taxels_id = self.get_contact_taxel_id(sim, fingername)

        if fingername == 'ff':
            c_points = taxels_id[0]
            print(">>c_points:", c_points)
            if len(c_points) == 0:
                return 'link_3.0_tip_tactile'
        if fingername == 'mf':
            c_points = taxels_id[0] + 144
            print(">>c_points:", c_points)
            if len(c_points) == 0:
                return 'link_7.0_tip_tactile'
        if fingername == 'rf':
            c_points = taxels_id[0] + 288
            print(">>c_points:", c_points)
            if len(c_points) == 0:
                return 'link_11.0_tip_tactile'
        if fingername == 'th':
            c_points = taxels_id[0] + 432
            print(">>c_points:", c_points)
            if len(c_points) == 0:
                return 'link_15.0_tip_tactile'

        actived_tmp_position = np.zeros((3, len(c_points)))
        taxel_position = np.zeros((3, 72))
        active_taxel_name = []
        dev_taxel_value = []
        if len(c_points) > 1:
            for i in range(len(c_points)):
                active_taxel_name.append(model._sensor_id2name[c_points[i]])
                # print(active_taxel_name[i])
                actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
            avg_position = actived_tmp_position.mean(1)
        else:
            avg_position = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[c_points[0]])[:3]

        if len(c_points) > 1:
            for i in range(len(c_points)):
                taxel_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
                dev_taxel_value.append(np.linalg.norm(taxel_position[:, i] - avg_position))
            min_value = min(dev_taxel_value)
            c_point_name = model._sensor_id2name[dev_taxel_value.index(min_value)]
        else:
            c_point_name = model._sensor_id2name[c_points[0]]

        # compare the whole fingertip taxel cost more time
        # if fingername == 'ff':
        #     min_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
        #     max_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX
        # if fingername == 'mf':
        #     min_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
        #     max_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX
        # if fingername == 'rf':
        #     min_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
        #     max_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX
        # if fingername == 'th':
        #     min_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN
        #     max_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX
        #
        # for i in range(min_taxel_id, max_taxel_id):
        #     # taxel_array_name.append()
        #     # print(model._sensor_id2name[i])
        #     taxel_position[:, i - min_taxel_id] = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[i])[:3]
        #     dev_taxel_value.append(np.linalg.norm(taxel_position[:, i - min_taxel_id] - avg_position))
        #
        # min_value = min(dev_taxel_value)
        # c_point_name = model._sensor_id2name[dev_taxel_value.index(min_value) + min_taxel_id]
        return c_point_name

    # get the median of all contact_nums, and translate to contact_name
    def get_c_point_name(self, model, c_points):
        if len(c_points) % 2 == 0:  # even number of contact points
            c_points = np.hstack((-1, c_points))  # change to odd
            #todo why the median of taxels are the contact point?
            c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
            print(np.median(c_points))
        else:
            c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
            print(np.median(c_points))

        return c_point_name

    # get contact_name using the average contact position and min distance to the taxel
    def get_avg_c_point(self, sim, model, c_points, fingername):
        actived_tmp_position = np.zeros((3, len(c_points)))
        taxel_position = np.zeros((3, 72))
        active_taxel_name = []
        dev_taxel_value = []
        for i in range(len(c_points)):
            active_taxel_name.append(model._sensor_id2name[c_points[i]])
            print(active_taxel_name[i])
            actived_tmp_position[:, i] = ug.get_relative_posquat(sim, "palm_link", active_taxel_name[i])[:3]
        avg_pressure = actived_tmp_position.mean(1)

        if fingername == 'ff':
            min_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN
            max_taxel_id = tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX
        if fingername == 'mf':
            min_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN
            max_taxel_id = tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX
        if fingername == 'rf':
            min_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN
            max_taxel_id = tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX
        if fingername == 'th':
            min_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN
            max_taxel_id = tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX

        for i in range(min_taxel_id, max_taxel_id):
            # taxel_array_name.append()
            print(model._sensor_id2name[i])
            taxel_position[:, i - min_taxel_id] = ug.get_relative_posquat(sim, "palm_link", model._sensor_id2name[i])[:3]
            dev_taxel_value.append(np.linalg.norm(taxel_position[:, i - min_taxel_id] - avg_pressure))

        min_value = min(dev_taxel_value)
        print(dev_taxel_value.index(min_value))
        c_point_name = model._sensor_id2name[dev_taxel_value.index(min_value) + min_taxel_id]
        return c_point_name

    # get the position of the contact taxel in the object frame
    def get_contact_taxel_position(self, sim, model, fingername, ref_frame):
        # start  = time.time()
        c_point_name = self.get_contact_taxel_name(sim, model, fingername)
        # end1 = time.time()
        # print('1 time diff ',end1-start)
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
        c_point_name = self.get_contact_taxel_name(sim, model, fingername)
        # get the position
        pos_contact = ug.get_relative_posquat(sim, ref_frame, c_point_name)

        T_contact = ug.posquat2trans(pos_contact)
        # nv = T_contact[:3, 2]  # Get R of contact point
        nv = T_contact[:3, 0]  # Get R of contact point
        return nv