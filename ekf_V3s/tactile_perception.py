import numpy as np
import tactile_allegro_mujo_const
import util_geometry as ug
import object_geometry as og


def is_finger_contact(sim, finger_name):
    if finger_name == 'ff':
        return (np.array(sim.data.sensordata[tactile_allegro_mujo_const.FF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.FF_TAXEL_NUM_MAX]) > 0.0).any()
    if finger_name == 'mf':
        return (np.array(sim.data.sensordata[tactile_allegro_mujo_const.MF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.MF_TAXEL_NUM_MAX]) > 0.0).any()
    if finger_name == 'rf':
        return (np.array(sim.data.sensordata[tactile_allegro_mujo_const.RF_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.RF_TAXEL_NUM_MAX]) > 0.0).any()
    if finger_name == 'th':
        return (np.array(sim.data.sensordata[tactile_allegro_mujo_const.TH_TAXEL_NUM_MIN: \
            tactile_allegro_mujo_const.TH_TAXEL_NUM_MAX]) > 0.0).any()

def get_contact_taxel_id(sim, finger_name):
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
def get_normal(sim, model, c_points, trans_cup2palm):
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

def get_c_point_name(model, c_points):  # get the median of all contact_nums, and translate to contact_name
    if len(c_points) % 2 == 0:  # even number of contact points
        c_points = np.hstack((-1, c_points))  # change to odd
        #todo why the median of taxels are the contact point?
        c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
        print(np.median(c_points))
    else:
        c_point_name = model._sensor_id2name[int(np.median(c_points))]  # use median to get the contact_point name
        print(np.median(c_points))

    return c_point_name


