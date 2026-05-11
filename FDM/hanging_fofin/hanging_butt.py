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

steps = 50
time_step = 0.05
show_steps = True
show_pipes = True
show_mesh = True

def fd(mesh):
    qpre = mesh.edges_attribute('qpre')
    xyz = mesh.vertices_attributes(['x', 'y', 'z'])        
    
    xyz, q, f, l, r = fd_numpy(xyz, edges, fixed, qpre, loads) 
    
    for key in mesh.vertices():
        mesh.vertex_attributes(key, ['x', 'y', 'z'], xyz[key])
        mesh.vertex_attribute(key, 'residual', r[key])
    for i, (u, v) in enumerate(mesh.edges()):
        mesh.edge_attribute((u, v), 'fpre', f[i])
        mesh.edge_attribute((u, v), 'qpre', q[i])
        mesh.edge_attribute((u, v), 'lpre', l[i])


def update_edgegroups(qpre, group_num):
    count = 0
    qsum = []
    
    for idx in range(num_edges):
        if edge_groups[idx] == group_num: # find the cable 1
            count += 1
            qsum.append(qpre[idx])
    for idx in range(num_edges):
        if edge_groups[idx] == group_num:
            qpre[idx] = sum(qsum) / count
    return qpre
    
    
folder = os.path.abspath("/Users/duch/Documents/Github/phd/pneu_fofin/hanging_fofin/data")
file_network = os.path.join(folder, "butt_quad_hanging.json")
mesh = Mesh.from_json(file_network) # network 
# mesh_0.flip_cycles()
# anchors = list(mesh_0.vertices_where({'is_anchor': True}))

file_ref = os.path.join(folder, "butt_quad_hanging.json")
mesh_ref = Mesh.from_json(file_ref) # reference mesh

mesh.update_default_vertex_attributes(is_anchor=False)
mesh.update_default_vertex_attributes(residual=None)
mesh.update_default_vertex_attributes(px=0, py=0, pz=0)
        
mesh.update_default_edge_attributes(qpre=1.0)


    
# for key in anchors:
#     mesh_0.vertex_attribute(key, 'is_anchor', True)
    
for key in mesh.vertices_on_boundary():
    mesh.vertex_attribute(key, 'is_anchor', True)

free = list(mesh.vertices_where({'is_anchor': False}))
load = - 240 * 0.005 * 0.1

for vkey in free:
    v_area = mesh.vertex_area(vkey)
    add_z = v_area * load
    mesh.vertex_attribute(vkey, 'pz', add_z)

# step I: compute the initial guess... 
# e_bdrs = list(set(mesh_0.edges_on_boundary()))
# for edge in e_bdrs:
#     mesh_0.edge_attribute(edge, 'qpre', 20)

# for vkey in mesh_0.vertices():
#     mesh_0.vertex_attribute(vkey, 'z', 0)

edges = list(mesh.edges())
fixed = list(mesh.vertices_where({'is_anchor': True}))
loads = mesh.vertices_attributes(['px', 'py', 'pz'])
    


viewer = App(width=800, height=800)
viewer.view.camera.rz = -300
viewer.view.camera.rx = -600
viewer.view.camera.tx = 0
viewer.view.camera.distance = 80





# # step II: find the target vertex of every vertex on the mesh 
target_xyzs = {}
for vkey in mesh_ref.vertices():
    xyz = mesh_ref.vertex_coordinates(vkey)
    target_xyzs[vkey] = xyz
    if mesh.is_vertex_on_boundary is False:
        mesh.vertex_attribute(vkey, 'z', 0)


num_edges = len(edges)

# Create a mapping from edges to indices
edge_index = {}
for idx, (a, b) in enumerate(edges):
    edge_index[(a, b)] = idx
    edge_index[(b, a)] = idx

# Identify boundary edges and get their indices
boundary_edges = list(mesh.edges_on_boundary())
boundary_edge_indices = [edge_index[edge] for edge in boundary_edges]

# Initialize an array to hold group indices for each edge
edge_groups = np.zeros(num_edges, dtype=int)

# Assign group indices
for idx in boundary_edge_indices:
    edge_groups[idx] = 1  # Group 1 for boundary edges




        
# Assign group 2 to cable_1
cable_1 = [(8, 75), (8, 149), (30, 38), (30, 84), (38, 149), (57, 126), (64, 75), (64, 139), (84, 126)]
cable1_indices = [edge_index[edge] for edge in cable_1]

cable_2 = [(24, 82), (24, 132), (29, 61), (29, 70), (44, 125), (44, 150), (56, 86), (56, 191), (57, 69), (57, 81), (61, 86), (69, 70), (81, 82), (125, 132)]
cable2_indices = [edge_index[edge] for edge in cable_2]

for edge in cable_1 + cable_2:
    mesh.edge_attribute(edge, 'qpre', 8.0)

for idx in cable1_indices:
    edge_groups[idx] = 2  # Group 2 for internal edges
    
for idx in cable2_indices:
    edge_groups[idx] = 3

# # Initialize force densities for each group
# q_group_values = {
#     0: 1.0,
#     1: 8.0,  # Initial q value for group 1 (boundary edges)
#     2: 1.0,  # Initial q value for group 2 (internal edges)
# }



fd(mesh) # initiate the form finding
if show_steps:
    viewer.add(mesh.copy(), color=(0, 0, 1))  # blue

for _ in range(steps):
    print(_)
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
    qpre = mesh.edges_attribute('qpre')
    xyz = mesh.vertices_attributes(['x', 'y', 'z'])
    loads = mesh.vertices_attributes(['px', 'py', 'pz'])
    fixed = list(mesh.vertices_where({'is_anchor': True}))
    edges = list(mesh.edges())

    v = len(xyz)
    free = list(set(range(v)) - set(fixed))
    xyz = np.asarray(xyz, dtype=np.float64).reshape((-1, 3))
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

    b_dxq = Cit.dot(diags(C.dot(xyz[:, 0])))
    dx_q = scipy.sparse.linalg.spsolve(Dn, b_dxq)

    b_dyq = Cit.dot(diags(C.dot(xyz[:, 1])))
    dy_q = scipy.sparse.linalg.spsolve(Dn, b_dyq)

    b_dzq = Cit.dot(diags(C.dot(xyz[:, 2])))
    dz_q = scipy.sparse.linalg.spsolve(Dn, b_dzq)

    sum_gradient = np.zeros(q.shape)


    # print(planes.shape, points.shape)
    for i, key in enumerate(free):
        x0, y0, z0 = target_xyzs[key]
        x, y, z = xyz[key]
        # print(i, key)
        gradient_i = 2*(x-x0)*dx_q[i] + 2*(y-y0)*dy_q[i] + 2*(z-z0)*dz_q[i]
        # gradient_i = k * 2 * (a*x + b*y + c*z + d) * (a*(dx_q[i]) + b*(dy_q[i]) + c*(dz_q[i]))
        sum_gradient += gradient_i.T
        
    qpre = qpre + time_step * sum_gradient
    print(_, qpre.shape)
    
   
    qpre = update_edgegroups(qpre, 2)
    qpre = update_edgegroups(qpre, 3)
    
    
    for idx in range(num_edges):
        if qpre[idx] <= 0.01:
            qpre[idx] = 0.01
    


    mesh_1 = mesh.copy()  
    for i, edge in enumerate(mesh_1.edges()):
        mesh_1.edge_attribute(edge, 'qpre', qpre[i])    
    fd(mesh_1)
    
    if show_steps:
        viewer.add(mesh.copy(), color=(0, 0, 1))  # blue
        
    if _ != steps-1:
        mesh = mesh_1.copy()


if show_pipes:
    for edge in mesh_1.edges():
        q = mesh_1.edge_attribute(edge, 'qpre')
        print(q)
        line = Line(*mesh_1.edge_coordinates(*edge))
        viewer.add(line, linecolor=(1, 0, 0), linewidth=q*10)




# # Get the current time
# current_time = datetime.datetime.now()
# # Format the time as a string with only numbers (YYYYMMDDHHMMSS)
# formatted_time = current_time.strftime("%Y%m%d%H%M%S")


# HERE = os.path.dirname(__file__)
# FOLDER = os.path.join(HERE, 'data')
# # mesh.to_json(os.path.join(FOLDER, 'mesh.json'))
# # mesh_0.to_json(os.path.join(FOLDER, 'mesh_0.json'))
# mesh_1.to_json(os.path.join(FOLDER, 'mesh_out_butt_{}.json'.format(formatted_time)))

viewer.add(mesh_ref, color=(1, 0, 0)) # red

viewer.add(mesh_1, color=(0, 1, 0)) # green

viewer.show()
