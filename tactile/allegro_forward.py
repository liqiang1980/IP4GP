from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
# robot = URDF.from_xml_file('/home/zongtaowang/workspace/sim_tactile/test_urdf.urdf')
robot = URDF.from_xml_file('../UR5/allegro_hand_tactile_right.urdf')
tree = kdl_tree_from_urdf_model(robot)

print(tree.getNrOfSegments())
chain1 = tree.getChain("palm_link", "link_3.0_tip")
# chain2 = tree.getChain("palm_link", "link_15.0_tip")
kdl_kin = KDLKinematics(robot, "palm_link", "link_3.0_tip")
q = [0, 0, 0,0]
pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 matrix)
print (pose )

q_ik = kdl_kin.inverse(pose) # inverse kinematics
print (q_ik)

if q_ik is not None:
    pose_sol = kdl_kin.forward(q_ik) # should equal pose
    print (pose_sol)

J = kdl_kin.jacobian(q)
print ('J:', J)
# chain = tree.getChain("link1", "link4")
print (chain1.getNrOfJoints())
# print (chain2.getNrOfJoints())
