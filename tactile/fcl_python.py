import numpy as np
import fcl
import pcl

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.face_only = []
        self.uses1 = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = [*map(float, values[1:4])]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
                # print(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0])-1) #在fcl中load的时候减去了1
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
                self.face_only.append((face))
                self.uses1.append((norms))

        # print(len(self.vertices))
        # print(self.vertices[0])
        # print(self.vertices[9999])

        # print("use1:", len(self.uses1))
        # print("use1:", self.uses1[0])
        # print("use1:", self.uses1[9999])

    def get_vertices(self):
        return self.vertices
        # print(len(self.faces))
        # print(self.uses)

    def get_faces(self):
        return self.face_only


def print_collision_result(o1_name, o2_name, result):
    print( 'Collision between {} and {}:'.format(o1_name, o2_name))
    print( '-'*30)
    print( 'Collision?: {}'.format(result.is_collision))
    print( 'Number of contacts: {}'.format(len(result.contacts)))
    print( '')

def print_continuous_collision_result(o1_name, o2_name, result):
    print( 'Continuous collision between {} and {}:'.format(o1_name, o2_name))
    print( '-'*30)
    print( 'Collision?: {}'.format(result.is_collide))
    print( 'Time of collision: {}'.format(result.time_of_contact))
    print( '')

def print_distance_result(o1_name, o2_name, result):
    print( 'Distance between {} and {}:'.format(o1_name, o2_name))
    print( '-'*30)
    print( 'Distance: {}'.format(result.min_distance))
    print( 'Closest Points:')
    print( result.nearest_points[0])
    print( result.nearest_points[1])
    print( '')

file_dir1 = 'pos_save/cup_trans.npy'  #文件的路径
file_dir2 = 'pos_save/link_3_tip_trans.npy'
# Create simple geometries
#单位是 mm
box = fcl.Box(1.0, 2.0, 3.0)
sphere = fcl.Sphere(4.0)
cone = fcl.Cone(5.0, 6.0)
cyl = fcl.Cylinder(2.0, 2.0)

#创建cup obj模型的vert和tris参数
obj_cup = OBJ( "cup_1.obj")
verts_cup = obj_cup.get_vertices()
tris_cup = obj_cup.get_faces()
# Create mesh geometry

mesh_cup = fcl.BVHModel()
mesh_cup.beginModel(len(verts_cup), len(tris_cup))
mesh_cup.addSubModel(verts_cup, tris_cup)
mesh_cup.endModel()
print("len_verts_cup:", len(verts_cup))

#fingertip
# obj_fingertip = OBJ( "fingertip.obj")
obj_fingertip = OBJ( "fingertip_part.obj")
verts_fingertip = obj_fingertip.get_vertices()
tris_fingertip = obj_fingertip.get_faces()
print("len_verts_fingertip:", len(verts_fingertip))
print("len_tris_fingertip:", len(tris_fingertip))
# Create mesh geometry

mesh_fingertip = fcl.BVHModel()
mesh_fingertip.beginModel(len(verts_fingertip), len(tris_fingertip))
mesh_fingertip.addSubModel(verts_fingertip, tris_fingertip)
mesh_fingertip.endModel()

#=====================================================================
# Pairwise collision checking
#=====================================================================
print( '='*60)
print( 'Testing Pairwise Collision Checking')
print( '='*60)
print( '')

req = fcl.CollisionRequest(enable_contact=True)
res = fcl.CollisionResult()

pos_R_cup=np.load(file_dir1)
pos_R_fingertip=np.load(file_dir2)
# print("R_cup:", R_cup)
# print("R_fingertip:", R_fingertip)

R_cup = pos_R_cup[0:3, 0:3]
pos_cup = pos_R_cup[0:3, 3]*1000

# R_fingertip = np.array([-0.03486789, 0.64261658, -0.02630733, 0.76494188])
R_fingertip = pos_R_fingertip[0:3, 0:3]
pos_fingertip = pos_R_fingertip[0:3, 3] *1000

print("pos_fingertip:", pos_fingertip)

    # cup_trans:
    #  [[-0.99998416 -0.00266248  0.00495896  0.12111152]
    #  [ 0.00494169  0.00645425  0.99996696 -0.00991389]
    #  [-0.0026944   0.99997563 -0.00644099  0.03512894]
    #  [ 0.          0.          0.          1.        ]]

t_cup = fcl.Transform(R_cup, pos_cup)
# t_cup = fcl.Transform(R_cup, pos_cup)

t_fingertip =  fcl.Transform(R_fingertip, pos_fingertip)
# t_fingertip =  fcl.Transform(R_fingertip, pos_fingertip)


o_cup = fcl.CollisionObject(mesh_cup, t_cup)
o_fingertip = fcl.CollisionObject(mesh_fingertip, t_fingertip)
 
n_contacts = fcl.collide(o_cup, o_fingertip, req, res)
contact = res.contacts[0]
print("contact:", contact)
# res.getContacts(contact)
normals = contact.normal
print("normal:", normals)

# n_contacts = fcl.collide(fcl.CollisionObject(mesh_cup, fcl.Transform(np.array([0.0,0.0,-1.0]))),
#                          fcl.CollisionObject(cyl, fcl.Transform()),
#                          req, res)
print_collision_result('cup', 'fingertip', res)


# print( '='*60)
# print( 'cup and fingertip Testing Pairwise Collision Checking')
# print( '='*60)
# print( '')
#
# req = fcl.CollisionRequest(enable_contact=True)
# res = fcl.CollisionResult()
#

#=====================================================================
# Pairwise distance checking
#=====================================================================
print( '='*60)
print( 'Testing Pairwise Distance Checking')
print( '='*60)
print( '')

req = fcl.DistanceRequest(enable_nearest_points=True)
res = fcl.DistanceResult()


dist = fcl.distance(o_cup,o_fingertip,
                    req, res)
print_distance_result('o_cup', 'o_fingertip', res)

# dist = fcl.distance(fcl.CollisionObject(box, fcl.Transform()),
#                     fcl.CollisionObject(cyl, fcl.Transform(np.array([6.0,0.0,0.0]))),
#                     req, res)
# print_distance_result('Box', 'Cylinder', res)
#
# dist = fcl.distance(fcl.CollisionObject(box, fcl.Transform()),
#                     fcl.CollisionObject(box, fcl.Transform(np.array([1.01,0.0,0.0]))),
#                     req, res)
# print_distance_result('Box', 'Box', res)

#=====================================================================
# Pairwise continuous collision checking
#=====================================================================
# print( '='*60)
# print( 'Testing Pairwise Continuous Collision Checking')
# print( '='*60)
# print( '')
#
# req = fcl.ContinuousCollisionRequest()
# res = fcl.ContinuousCollisionResult()
#
# dist = fcl.continuousCollide(fcl.CollisionObject(box, fcl.Transform()),
#                              fcl.Transform(np.array([5.0, 0.0, 0.0])),
#                              fcl.CollisionObject(cyl, fcl.Transform(np.array([5.0,0.0,0.0]))),
#                              fcl.Transform(np.array([0.0, 0.0, 0.0])),
#                              req, res)
# print_continuous_collision_result('Box', 'Cylinder', res)
#
# #=====================================================================
# # Managed collision checking
# #=====================================================================
# print( '='*60)
# print( 'Testing Managed Collision and Distance Checking')
# print( '='*60)
# print( '')
# objs1 = [fcl.CollisionObject(box, fcl.Transform(np.array([20,0,0]))), fcl.CollisionObject(sphere)]
# objs2 = [fcl.CollisionObject(cone), fcl.CollisionObject(mesh)]
# objs3 = [fcl.CollisionObject(box), fcl.CollisionObject(sphere)]
#
# manager1 = fcl.DynamicAABBTreeCollisionManager()
# manager2 = fcl.DynamicAABBTreeCollisionManager()
# manager3 = fcl.DynamicAABBTreeCollisionManager()
#
# manager1.registerObjects(objs1)
# manager2.registerObjects(objs2)
# manager3.registerObjects(objs3)
#
# manager1.setup()
# manager2.setup()
# manager3.setup()
#
# #=====================================================================
# # Managed internal (n^2) collision checking
# #=====================================================================
# cdata = fcl.CollisionData()
# manager1.collide(cdata, fcl.defaultCollisionCallback)
# print( 'Collision within manager 1?: {}'.format(cdata.result.is_collision))
# print( '')
#
# cdata = fcl.CollisionData()
# manager2.collide(cdata, fcl.defaultCollisionCallback)
# print( 'Collision within manager 2?: {}'.format(cdata.result.is_collision))
# print( '')
#
# ##=====================================================================
# ## Managed internal (n^2) distance checking
# ##=====================================================================
# ddata = fcl.DistanceData()
# manager1.distance(ddata, fcl.defaultDistanceCallback)
# print( 'Closest distance within manager 1?: {}'.format(ddata.result.min_distance))
# print( '')
#
# ddata = fcl.DistanceData()
# manager2.distance(ddata, fcl.defaultDistanceCallback)
# print( 'Closest distance within manager 2?: {}'.format(ddata.result.min_distance))
# print( '')
#
# #=====================================================================
# # Managed one to many collision checking
# #=====================================================================
# req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
# rdata = fcl.CollisionData(request = req)
#
# manager1.collide(fcl.CollisionObject(mesh), rdata, fcl.defaultCollisionCallback)
# print( 'Collision between manager 1 and Mesh?: {}'.format(rdata.result.is_collision))
# print( 'Contacts:')
# for c in rdata.result.contacts:
#     print( '\tO1: {}, O2: {}'.format(c.o1, c.o2))
# print( '')
#
# #=====================================================================
# # Managed many to many collision checking
# #=====================================================================
# rdata = fcl.CollisionData(request = req)
# manager3.collide(manager2, rdata, fcl.defaultCollisionCallback)
# print( 'Collision between manager 2 and manager 3?: {}'.format(rdata.result.is_collision))
# print( 'Contacts:')
# for c in rdata.result.contacts:
#     print( '\tO1: {}, O2: {}'.format(c.o1, c.o2))
# print( '')
