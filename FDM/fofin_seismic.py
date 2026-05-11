import os
import numpy as np

import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import diags

from compas.numerical.matrices import connectivity_matrix
from compas.datastructures import Mesh
from compas.numerical import fd_numpy
from compas_view2.app import App
from compas.geometry import Line
import datetime

steps = 1
pressure = 2.0
time_step = 5
show_steps = False
show_pipes = True
show_mesh = False

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

        
folder = os.path.abspath("/Users/duch/Downloads")
# file_name = "shell_four_corners.obj"
# file_path = os.path.join(folder, file_name)
file_network = os.path.join(folder, "seismic_vault.json")
mesh_0 = Mesh.from_json(file_network) # network 
# mesh_0.flip_cycles()
# anchors = list(mesh_0.vertices_where({'is_anchor': True}))

file_ref =  os.path.join(folder, "seismic_vault.json")
mesh = Mesh.from_json(file_ref) # reference mesh


mesh_0.update_default_vertex_attributes(is_anchor=False)
mesh_0.update_default_vertex_attributes(residual=None)
mesh_0.update_default_vertex_attributes(px=0, py=0, pz=0)
mesh_0.update_default_edge_attributes(qpre=3.0)

# for key in anchors:
#     mesh_0.vertex_attribute(key, 'is_anchor', True)
    
for key in mesh_0.vertices_on_boundary():
    mesh_0.vertex_attribute(key, 'is_anchor', True)

loop = mesh_0.edge_loop((608, 579))
for edge in loop:
    mesh_0.edge_attribute(edge, 'qpre', 10)
loop2 = mesh_0.edge_loop((134, 345))
for edge in loop2:
    mesh_0.edge_attribute(edge, 'qpre', 10)


# step I: compute the initial guess... 
# e_bdrs = list(set(mesh_0.edges_on_boundary()))
# for edge in e_bdrs:
#     mesh_0.edge_attribute(edge, 'qpre', 20)

# for vkey in mesh_0.vertices():
#     mesh_0.vertex_attribute(vkey, 'z', 0)
    
fd_inflate(mesh_0, pressure=pressure)

viewer = App(width=800, height=800)
viewer.view.camera.rz = -300
viewer.view.camera.rx = -600
viewer.view.camera.tx = 0
viewer.view.camera.distance = 80

# # print(mesh_0.vertices_attributes())
# # print(mesh_0.edges_attributes())

# step II: find the target vertex of every vertex on the mesh 
target_xyzs = {}
for vkey in mesh_0.vertices():
    xyz = mesh_0.vertex_coordinates(vkey)
    target_xyzs[vkey] = xyz
    if mesh_0.is_vertex_on_boundary is False:
        mesh_0.vertex_attribute(vkey, 'z', 0)

# set the vertex xyz z coordinate to 0

for _ in range(steps):

    # III. gradient descent 
    # q_t+1 = q_t - lamda * f'(q_t)  
    # argmin(∑(||V - S||)**2)   the distance from the point to the cloest point on the mesh
    # the distance from an arbitrary point to the plane is: 
    # distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    # now minimise ∑  ((ax + by + cz + d)/ sqrt(a^2 + b^2 + c^2)) **2    
    #             = ∑ 1/(a^2 + b^2 + c^2 ) * (ax + by + cz + d)**2
    # let k = 1/(a^2 + b^2 + c^2 )
    # gradient is: ∑ k * 2 *(f(ax + by + cz + d)) * (a*(dx/dq) + b*(dy/dq) + c*(dz/dq))
    
    # vertices = np.array(mesh.vertices_attributes("xyz"), dtype=np.float64).reshape((-1, 3))
    mesh_0_pts = mesh_0.vertices_attributes('xyz')

    qpre = mesh_0.edges_attribute('qpre')
    fixed = list(mesh_0.vertices_where({'is_anchor': True}))
    loads = mesh_0.vertices_attributes(['px', 'py', 'pz'])
    edges = list(mesh_0.edges())

    v = len(mesh_0_pts)
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

    # L = scipy.linalg.cholesky(Dn.todense(), lower=True, check_finite=True)

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
        x0, y0, z0 = target_xyzs[key]
        x, y, z = xyz[key]
        # print(i, key)
        gradient_i = 2*(x-x0)*dx_q[i] + 2*(y-y0)*dy_q[i] + 2*(z-z0)*dz_q[i]
        # gradient_i = k * 2 * (a*x + b*y + c*z + d) * (a*(dx_q[i]) + b*(dy_q[i]) + c*(dz_q[i]))
        sum_gradient += gradient_i.T
    # print(sum_gradient)

    obj_sum = 0
    for i, key in enumerate(free):
        x0, y0, z0 = target_xyzs[key]
        x, y, z = xyz[key]
        obj_sum += np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
    print("obj", obj_sum)

    # gradient descent
    # q_t+1 = q_t - lamda * f'(q_t)  
    
    # print(qpre)
    
    qpre = qpre + time_step * sum_gradient
    print(_, qpre.shape)

    mesh_1 = mesh_0.copy()
    for i, edge in enumerate(mesh_1.edges()):
        mesh_1.edge_attribute(edge, 'qpre', qpre[i])    
    fd_inflate(mesh_1, pressure=pressure)
    
    if show_steps:
        viewer.add(mesh_0, color=(0, 0, 1)) # blue
    
    if _ != steps-1:
        mesh_0 = mesh_1.copy()


if show_pipes:
    for edge in mesh_1.edges():
        q = mesh_1.edge_attribute(edge, 'qpre')
        line = Line(*mesh_1.edge_coordinates(*edge))
        viewer.add(line, linecolor=(1, 0, 0), linewidth=q)




# Get the current time
current_time = datetime.datetime.now()
# Format the time as a string with only numbers (YYYYMMDDHHMMSS)
formatted_time = current_time.strftime("%Y%m%d%H%M%S")


HERE = os.path.dirname(__file__)
FOLDER = os.path.join(HERE, 'data')
# mesh.to_json(os.path.join(FOLDER, 'mesh.json'))
# mesh_0.to_json(os.path.join(FOLDER, 'mesh_0.json'))
mesh_1.to_json(os.path.join(FOLDER, 'mesh_out_seismic_{}.json'.format(formatted_time)))

viewer.add(mesh_0, color=(0, 0, 1)) # blue
# if show_mesh:
#     viewer.add(mesh_1, color=(1, 0, 0)) # red
viewer.add(mesh, color=(0, 1, 0)) # green
viewer.show()
