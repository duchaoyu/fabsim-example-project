import random
import os
import compas
from compas.datastructures import Mesh
from compas.geometry import Line
from compas_view2.app import App

folder = os.path.abspath("/Users/duch/Documents/Github/phd/pneu_fofin/data")
# file_name = "shell_four_corners.obj"
# file_path = os.path.join(folder, file_name)
filepath = os.path.join(folder, "mesh_out_cross_20240323073955.json")
mesh = Mesh.from_json(filepath) # reference mesh


viewer = App(width=800, height=800)
viewer.view.camera.rz = -300
viewer.view.camera.rx = -600
viewer.view.camera.tx = 0
viewer.view.camera.distance = 80

for edge in mesh.edges():
    q = mesh.edge_attribute(edge, 'qpre')[0]
    f = mesh.edge_attribute(edge, 'fpre')[0]
    edge_len = mesh.edge_length(*edge)
    E = 50000 # pa
    # print(1/(1 + f / (edge_len * E)))
    print(q)
    
    

    line = Line(*mesh.edge_coordinates(*edge))
    viewer.add(line, linecolor=(1, 0, 0), linewidth=abs(q))


viewer.add(mesh, color=(0, 1, 0)) # green
viewer.show()