#used for configure parameters
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

                #parse fingers' parameters
                #the parameters will be organized in list type in the following way
                #['allegro', ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['th', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}], ['ff', 1, {'j1': 5, 'j2': 6, 'j3': 7, 'j4': 8}]]
                fingers = dom.getElementsByTagName('finger')
                for finger in fingers:
                        finger_name = finger.getAttribute("name")
                        is_used = finger.getElementsByTagName("used")[0]
                        print(is_used.firstChild.data)
                        js = finger.getElementsByTagName('init_posture')
                        for jnt in js:
                                j_init_dic = {
                                    "j1":jnt.getElementsByTagName("j1")[0].firstChild.data,
                                    "j2":jnt.getElementsByTagName("j2")[0].firstChild.data,
                                    "j3":jnt.getElementsByTagName("j3")[0].firstChild.data,
                                    "j4":jnt.getElementsByTagName("j4")[0].firstChild.data
                                }
                        finger_param = [finger_name, is_used.firstChild.data, j_init_dic]
                        hand_param.append(finger_param)
                print(hand_param)
                print(hand_param[1][1])
                print(hand_param[2][1])
                print(hand_param[3][1])
                print(hand_param[4][1])
                #access to data in configure file
                #hand_param[0]: name of the hand
                #hand_param[1]: parameter of "ff" finger
                #hand_param[1][0]: name of "ff" finger
                #hand_param[1][1]: is "ff" finger used for grasping
                #hand_param[1][2]["j1"]: init of j1 of "ff" finger

                #parse object info
                object_param = []
                object_name = dom.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
                object_position_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_position')[0].firstChild.data
                object_orientation_noise = dom.getElementsByTagName('object')[0].getElementsByTagName('noise_orientation')[0].firstChild.data
                object_param.append(object_name)
                object_param.append(object_position_noise)
                object_param.append(object_orientation_noise)
                print(object_param)

                # algorithm parameters
                alg_param = []
                is_data_stored = dom.getElementsByTagName('alg')[0].getElementsByTagName('store_data')[0].firstChild.data
                alg_param.append(is_data_stored)
                print(alg_param)
                return hand_param, object_param, alg_param
