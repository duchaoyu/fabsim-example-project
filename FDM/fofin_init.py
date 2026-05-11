import os
import numpy as np

import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import diags

from compas.numerical.matrices import connectivity_matrix
from compas.datastructures import Mesh
from compas.numerical import fd_numpy
from compas_view2.app import App

import datetime

steps = 15
pressure = 2.2
time_step = 2


def fd_inflate(mesh, pressure):
    xyz = mesh.vertices_attributes(['x', 'y', 'z'])
    edges = list(mesh.edges())
    fixed = list(mesh.vertices_where({'is_anchor': True}))
    free= list(mesh.vertices_where({'is_anchor': False}))

    for _ in range(5):
        for vkey in free:
            v_n = mesh.vertex_normal(vkey)
            v_area = mesh.vertex_area(vkey)
            px = v_n[0] * v_area * pressure
            py = v_n[1] * v_area * pressure
            pz = v_n[2] * v_area * pressure
            mesh.vertex_attributes(vkey, ['px', 'py', 'pz'], [px, py, pz])
        load = - 240 * 0.005 * 0.1
        for vkey in free:
            v_area = mesh.vertex_area(vkey)
            add_z = v_area * load
            pz = mesh.vertex_attribute(vkey, 'pz')
            mesh.vertex_attribute(vkey, 'pz', pz + add_z)
            
        loads = mesh.vertices_attributes(['px', 'py', 'pz'])
        qpre = mesh.edges_attribute('qpre')
        
        xyz, q, f, l, r = fd_numpy(xyz, edges, fixed, qpre, loads) 
        
        for key in mesh.vertices():
            mesh.vertex_attributes(key, ['x', 'y', 'z'], xyz[key])
            mesh.vertex_attribute(key, 'residual', r[key])
        for i, (u, v) in enumerate(mesh.edges()):
            mesh.edge_attribute((u, v), 'fpre', f[i])
            mesh.edge_attribute((u, v), 'qpre', q[i])
            mesh.edge_attribute((u, v), 'lpre', l[i])

        
folder = "/Users/duch/Documents/PhD/inflatable/base_geometries"
file_name = "shell_four_corners.obj"
file_path = os.path.join(folder, file_name)

mesh = Mesh.from_obj(file_path)



mesh.update_default_vertex_attributes(is_anchor=False)
mesh.update_default_vertex_attributes(residual=None)
mesh.update_default_vertex_attributes(px=0, py=0, pz=0)

mesh.update_default_edge_attributes(qpre=1.0)

anchors = [113, 114, 115, 116, 118, 247, 248, 250, 252, 254, 256, 258, 260, 262, 347, 348, 350, 352, 354, 356, 358, 360, 362, 447, 448, 450, 452, 454, 456, 458, 460, 462, 856, 857, 859, 861]
mesh.vertices_attribute('is_anchor', True, anchors)


# step I: compute the initial guess... 
mesh_0 = mesh.copy()
e_bdrs = list(set(mesh.edges_on_boundary()))
for edge in e_bdrs:
    mesh_0.edge_attribute(edge, 'qpre', 20)

for vkey in mesh_0.vertices():
    mesh_0.vertex_attribute(vkey, 'z', 0)
    
fd_inflate(mesh_0, pressure=pressure)
# print(mesh_0.vertices_attributes())
# print(mesh_0.edges_attributes())

# step II: find the cloest mesh face of every vertex on the mesh 
# make the input mesh triangles
if mesh.is_trimesh() is False:
    vertices = mesh.vertices_attributes('xyz')
    tri_faces = []
    for fkey in mesh.faces():
        f_vkeys = mesh.face_vertices(fkey)
        if len(f_vkeys) != 3:
            if len(f_vkeys) == 4:
                tri_faces.append(f_vkeys[:3])
                tri_faces.append(f_vkeys[2:] + [f_vkeys[0]])
            else:
                raise ValueError('The face is not a triangle or quad')
        elif f_vkeys == 3:
            tri_faces.append(f_vkeys)
    mesh = Mesh.from_vertices_and_faces(vertices, tri_faces)
    
# check the trianges where the vertices are most close to 

viewer = App(width=800, height=800)
viewer.view.camera.rz = -30
viewer.view.camera.rx = -60
viewer.view.camera.tx = 0
viewer.view.camera.distance = 8


for _ in range(steps):
    # preprocess
    mesh_0_pts = mesh_0.vertices_attributes('xyz')
    i_k = mesh.index_key()
    fk_fi = {fkey: index for index, fkey in enumerate(mesh.faces())}
    vertices = np.array(mesh.vertices_attributes("xyz"), dtype=np.float64).reshape((-1, 3))
    triangles = np.array([mesh.face_coordinates(fkey) for fkey in mesh.faces()], dtype=np.float64)
    points = np.array(mesh_0_pts, dtype=np.float64).reshape((-1, 3))
    closest_vis = np.argmin(distance_matrix(points, vertices), axis=1)
    # transformation matrices

    planes = np.zeros((points.shape[0], 4))
    # pull every point onto the mesh
    for i in range(points.shape[0]):
        point = points[i]
        closest_vi = closest_vis[i]
        closest_vk = i_k[closest_vi]
        closest_tris = [fk_fi[fk] for fk in mesh.vertex_faces(closest_vk, ordered=True) if fk is not None]
        
        a0, b0, c0, d0 = 0, 0, 0, 0
        min_dis = 100000000000000
        for tri in closest_tris:
            n = mesh.face_normal(tri)
            # triangle = triangles[tri]
            # P1 = triangle[0]
            # u = triangle[1] - P1
            # v = triangle[2] - P1
            # n = np.cross(u, v)
            # if np.dot(n, [0,0,1]) <0: 
            #     n = -n
            a, b, c = n
            d = -np.dot(n, mesh.face_centroid(tri))
            # {a}x + {b}y + {c}z + ({d}) = 0
            distance = (a * point[0] + b * point[1] + c * point[2] + d)**2 / (a**2 + b**2 + c**2)
            if distance < min_dis:
                min_dis = distance
                a0, b0, c0, d0 = a, b, c, d
        planes[i] = [a0, b0, c0, d0]
    # print(planes)

            
    # III. gradient descent 
    # q_t+1 = q_t - lamda * f'(q_t)  
    # argmin(∑(||V - S||)**2)   the distance from the point to the cloest point on the mesh
    # the distance from an arbitrary point to the plane is: 
    # distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    # now minimise ∑  ((ax + by + cz + d)/ sqrt(a^2 + b^2 + c^2)) **2    
    #             = ∑ 1/(a^2 + b^2 + c^2 ) * (ax + by + cz + d)**2
    # let k = 1/(a^2 + b^2 + c^2 )
    # gradient is: ∑ k * 2 *(f(ax + by + cz + d)) * (a*(dx/dq) + b*(dy/dq) + c*(dz/dq))

    qpre = mesh_0.edges_attribute('qpre')
    fixed = list(mesh_0.vertices_where({'is_anchor': True}))
    loads = mesh_0.vertices_attributes(['px', 'py', 'pz'])
    edges = list(mesh_0.edges())

    v = len(vertices)
    free = list(set(range(v)) - set(fixed))
    xyz = np.asarray(mesh_0_pts, dtype=np.float64).reshape((-1, 3))
    q = np.asarray(qpre, dtype=float).reshape((-1, 1))
    p = np.asarray(loads, dtype=float).reshape((-1, 3))
    C = connectivity_matrix(edges, 'csr')
    Ci = C[:, free]
    Cf = C[:, fixed]
    Ct = C.transpose()
    Cit = Ci.transpose()
    Q = diags([q.flatten()], [0])
    # A = Cit.dot(Q).dot(Ci)
    # b = p[free] - Cit.dot(Q).dot(Cf).dot(xyz[fixed])
    Dn = Cit.dot(Q).dot(Ci)

    # print((Dn - Dn.T).nnz == 0) # is symmetric
    # print(scipy.sparse.issparse(Dn)) # is sparse matrix
    # print(all(np.isscalar(Dn.data[i]) for i in range(Dn.data.size))) # all scalar value
    # print(np.linalg.eigvalsh(Dn.todense())) # all positive

    L = scipy.linalg.cholesky(Dn.todense(), lower=True, check_finite=True)

    b_dxq = Cit.dot(diags(C.dot(xyz[:, 0])))
    dx_q = scipy.sparse.linalg.spsolve(Dn, b_dxq)

    b_dyq = Cit.dot(diags(C.dot(xyz[:, 1])))
    dy_q = scipy.sparse.linalg.spsolve(Dn, b_dyq)

    b_dzq = Cit.dot(diags(C.dot(xyz[:, 2])))
    dz_q = scipy.sparse.linalg.spsolve(Dn, b_dzq)

    # b_dyq = Cit.dot(C).dot(xyz[:, 1])
    # y_dyq = scipy.sparse.linalg.spsolve(L, b_dyq)
    # dy_q = scipy.sparse.linalg.spsolve(L.transpose(), y_dyq)

    # b_dzq = Cit.dot(C).dot(xyz[:, 2])
    # y_dzq = scipy.sparse.linalg.spsolve(L, b_dzq)
    # dz_q = scipy.sparse.linalg.spsolve(L.transpose(), y_dzq)

    # ∑ k * 2 *((ax + by + cz + d)) * (a*(dx/dq) + b*(dy/dq) + c*(dz/dq))
    # print(dx_q.shape, dy_q.shape, dz_q.shape, q.shape, xyz.shape)   
    # print(L.shape, b_dzq.shape, dz_q.shape, Dn.shape, len(free))

    sum_gradient = np.zeros(q.shape)

    # print(planes.shape, points.shape)
    for i, key in enumerate(free):
        a, b, c, d = planes[key]
        k = 1/(a**2 + b**2 + c**2)
        x, y, z = points[key]
        # print(i, key)
        gradient_i = k * 2 * (a*x + b*y + c*z + d) * (a*(dx_q[i]) + b*(dy_q[i]) + c*(dz_q[i]))
        sum_gradient += gradient_i.T
    # print(sum_gradient)

    # gradient descent
    # q_t+1 = q_t - lamda * f'(q_t)  
    
    # print(qpre)
    
    qpre = qpre + time_step * sum_gradient
    print(_, qpre.shape)

    mesh_1 = mesh_0.copy()
    for i, edge in enumerate(mesh_1.edges()):
        mesh_1.edge_attribute(edge, 'qpre', qpre[i])    
    fd_inflate(mesh_1, pressure=pressure)
    
    # viewer.add(mesh_0, color=(0, 0, 1)) # blue
    
    if _ != steps-1:
        mesh_0 = mesh_1.copy()


# from compas.geometry import Line
# for edge in mesh_1.edges():
#     q = mesh_1.edge_attribute(edge, 'qpre')
#     line = Line(*mesh.edge_coordinates(*edge))
#     viewer.add(line, linecolor=(1, 0, 0), linewidth=q)




# Get the current time
current_time = datetime.datetime.now()
# Format the time as a string with only numbers (YYYYMMDDHHMMSS)
formatted_time = current_time.strftime("%Y%m%d%H%M%S")


HERE = os.path.dirname(__file__)
FOLDER = os.path.join(HERE, 'data')
mesh.to_json(os.path.join(FOLDER, 'mesh.json'))
# mesh_0.to_json(os.path.join(FOLDER, 'mesh_0.json'))
mesh_1.to_json(os.path.join(FOLDER, 'mesh_out_{}.json'.format(formatted_time)))

# viewer.add(mesh_0, color=(0, 0, 1)) # blue
viewer.add(mesh_1, color=(1, 0, 0)) # red
viewer.add(mesh, color=(0, 1, 0)) # green
viewer.show()
