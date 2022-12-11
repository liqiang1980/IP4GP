# used for configure parameters
import sys
from xml.dom.minidom import parseString


def read_xml(xml_file):
    with open(xml_file, 'r') as f:
        data = f.read()
    return parseString(data)


def pass_arg():
    if (len(sys.argv) < 2):
        print("Error: Missing parameter.")
    else:
        dom = read_xml(sys.argv[1])
        hand_name = dom.getElementsByTagName('name')[0].firstChild.data
        hand_param = []
        hand_param.append(hand_name)

        # parse fingers' parameters
        # the parameters will be organized in list type in the following way
        # allegro
        # ['ff', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [0, 72], ['touch_0_3_6', 38], 'link_3.0_tip']
        # ['ffd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [72, 108], ['touch_1_3_3', 86], 'link_2.0_tip']
        # ['ffq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [108, 144], ['touch_2_3_3', 122], 'link_1.0_tip']
        # ['mf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [144, 216], ['touch_7_3_6', 182], 'link_7.0_tip']
        # ['mfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [216, 252], ['touch_8_3_3', 230], 'link_6.0_tip']
        # ['mfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [252, 288], ['touch_9_3_3', 266], 'link_5.0_tip']
        # ['rf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [288, 360], ['touch_11_3_6', 326], 'link_11.0_tip']
        # ['rfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [360, 396], ['touch_12_3_3', 374], 'link_10.0_tip']
        # ['rfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [396, 432], ['touch_13_3_3', 410], 'link_9.0_tip']
        # ['th', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [432, 504], ['touch_15_3_6', 470], 'link_15.0_tip']
        # ['thd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [504, 540], ['touch_16_3_3', 518], 'link_14.0_tip']
        # ['palm', 1, {'j1': '-1', 'j2': '-1', 'j3': '-1', 'j4': '-1'}, [540, 635], ['touch_111_1_6', 604], 'palm_link']
        fingers = dom.getElementsByTagName('finger')
        for finger in fingers:
            finger_name = finger.getAttribute("name")
            is_used = int(finger.getElementsByTagName("used")[0].firstChild.data)
            if not is_used:
                continue
            # print(is_used.firstChild.data)
            js = finger.getElementsByTagName('init_posture')
            for jnt in js:
                j_init_dic = {
                    "j1": jnt.getElementsByTagName("j1")[0].firstChild.data,
                    "j2": jnt.getElementsByTagName("j2")[0].firstChild.data,
                    "j3": jnt.getElementsByTagName("j3")[0].firstChild.data,
                    "j4": jnt.getElementsByTagName("j4")[0].firstChild.data
                }
            tacs = finger.getElementsByTagName("taxel_id")
            for tac in tacs:
                tac_id_list = [int(tac.getElementsByTagName("min")[0].firstChild.data),
                               int(tac.getElementsByTagName("max")[0].firstChild.data)]
            default_tac_name = finger.getElementsByTagName("default_tac")[0].getElementsByTagName("name")[0].firstChild.data
            default_tac_id = int(finger.getElementsByTagName("default_tac")[0].getElementsByTagName("id")[0].firstChild.data)
            jnt = finger.getElementsByTagName("jnt")[0].firstChild.data
            """ One finger part param """
            finger_param = [finger_name, is_used, j_init_dic, tac_id_list, [default_tac_name, default_tac_id], jnt]
            hand_param.append(finger_param)

        for item in hand_param:
            print(item)

        # parse object info
        object_param = []
        object_name = dom.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        object_position_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_position')[0].firstChild.data
        object_orientation_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_orientation')[0].firstChild.data
        object_static = dom.getElementsByTagName('object')[0].getElementsByTagName('static')[0].firstChild.data
        object_param.append(object_name)
        object_param.append(object_position_noise)
        object_param.append(object_orientation_noise)
        object_param.append(int(object_static))
        print('\n', object_param)

        # algorithm parameters
        alg_param = []
        is_data_stored = dom.getElementsByTagName('alg')[0].getElementsByTagName('store_data')[0].firstChild.data
        alg_param.append(is_data_stored)
        print('\n', alg_param)
        return hand_param, object_param, alg_param
