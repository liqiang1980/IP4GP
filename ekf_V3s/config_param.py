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
        # ['ff', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [0, 72], 'touch_0_3_6']
        # ['ffd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [72, 108], 'touch_1_3_3']
        # ['ffq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [108, 144], 'touch_2_3_3']
        # ['mf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [144, 216], 'touch_7_3_6']
        # ['mfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [216, 252], 'touch_8_3_3']
        # ['mfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [252, 288], 'touch_9_3_3']
        # ['rf', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [288, 360], 'touch_11_3_6']
        # ['rfd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [360, 396], 'touch_12_3_3']
        # ['rfq', 1, {'j1': '-1', 'j2': '-1', 'j3': '7', 'j4': '8'}, [396, 432], 'touch_13_3_3']
        # ['th', 1, {'j1': '5', 'j2': '6', 'j3': '7', 'j4': '8'}, [432, 504], 'touch_15_3_6']
        # ['thd', 1, {'j1': '-1', 'j2': '6', 'j3': '7', 'j4': '8'}, [504, 540], 'touch_16_3_3']
        # ['palm', 1, {'j1': '-1', 'j2': '-1', 'j3': '-1', 'j4': '-1'}, [540, 635], 'touch_111_6_6']
        fingers = dom.getElementsByTagName('finger')
        for finger in fingers:
            finger_name = finger.getAttribute("name")
            is_used = int(finger.getElementsByTagName("used")[0].firstChild.data)
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
            default_tac = finger.getElementsByTagName("default_tac")[0].firstChild.data
            finger_param = [finger_name, is_used, j_init_dic, tac_id_list, default_tac]
            hand_param.append(finger_param)

        for item in hand_param:
            print(item)
        # print(hand_param)
        # print(hand_param[1][1])
        # print(hand_param[2][1])
        # print(hand_param[3][1])
        # print(hand_param[4][1])
        # access to data in configure file
        # hand_param[0]: name of the hand
        # hand_param[1]: parameter of "ff" finger
        # hand_param[1][0]: name of "ff" finger
        # hand_param[1][1]: is "ff" finger used for grasping
        # hand_param[1][2]["j1"]: init of j1 of "ff" finger

        # parse object info
        object_param = []
        object_name = dom.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        object_position_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_position')[
            0].firstChild.data
        object_orientation_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_orientation')[
            0].firstChild.data
        object_static = dom.getElementsByTagName('object')[0].getElementsByTagName('static')[0].firstChild.data
        object_param.append(object_name)
        object_param.append(object_position_noise)
        object_param.append(object_orientation_noise)
        object_param.append(object_static)
        print('\n', object_param)

        # algorithm parameters
        alg_param = []
        is_data_stored = dom.getElementsByTagName('alg')[0].getElementsByTagName('store_data')[0].firstChild.data
        alg_param.append(is_data_stored)
        print('\n', alg_param)
        return hand_param, object_param, alg_param
