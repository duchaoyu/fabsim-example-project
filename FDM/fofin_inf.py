import os
import numpy as np

from compas.datastructures import Mesh
from compas.numerical import fd_numpy
from compas.geometry import Line

from compas_view2.app import App


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



file_path = "/Users/duch/Documents/Github/phd/pneu_fofin/data/2part.json"

pressure = 1.0

mesh0 = Mesh.from_json(file_path)
mesh = mesh0.copy()
mesh.flip_cycles()

mesh.update_default_vertex_attributes(is_anchor=False)
mesh.update_default_vertex_attributes(residual=None)
mesh.update_default_vertex_attributes(px=0, py=0, pz=0)
mesh.update_default_edge_attributes(qpre=2.0)

for key in mesh.vertices_on_boundary():
    mesh.vertex_attribute(key, 'is_anchor', True)



viewer = App(width=800, height=800)
viewer.view.camera.rz = -300
viewer.view.camera.rx = -600
viewer.view.camera.tx = 0
viewer.view.camera.distance = 80

# middle vertex: 0
nbrs = mesh.vertex_neighbors(0, ordered=True)


for (a, b) in zip(nbrs, nbrs[1:]+[nbrs[0]]):
    edges = mesh.edge_strip((b, a))
    for edge in edges:
        mesh.edge_attribute(edge, 'qpre', 1.0)
    
fd_inflate(mesh, pressure=pressure)

# mesh0 = mesh.copy()
# viewer.add(mesh0)

# # must use iterative method to change the force density to negative
# for q in np.linspace(2, -2, 10):
#     for i, nbr in enumerate(nbrs[::2]):
#         edges = mesh.edge_loop((0, nbr))
#         for edge in edges:
#             mesh.edge_attribute(edge, 'qpre', q)
#     # mesh0 = mesh.copy()
#     # viewer.add(mesh0)

viewer.add(mesh)
for edge in mesh.edges():
    q = mesh.edge_attribute(edge, 'qpre')
    line = Line(*mesh.edge_coordinates(*edge))
    if q > 0: 
        viewer.add(line, linecolor=(1, 0, 0), linewidth=10*q)
    else:
        viewer.add(line,linecolor=(0, 0, 1), linewidth=-10*q)

viewer.add(mesh0)
# viewer.add(mesh, color=(0, 1, 0)) # green
viewer.show()