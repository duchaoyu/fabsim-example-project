import os
import numpy as np

import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import diags

from compas.numerical.matrices import connectivity_matrix
from compas.datastructures import Network
# from compas.numerical import fd_numpy, dr_numpy
from compas_view2.app import App
from compas.geometry import Line, Vector, Sphere
import datetime



from numpy import array
from numpy import isnan
from numpy import isinf
from numpy import ones
from numpy import zeros
from scipy.linalg import norm
from scipy.sparse import diags

from compas.numerical import connectivity_matrix
from compas.numerical import normrow

K = [
    [0.0],
    [0.5, 0.5],
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0, 1.0],
]


class Coeff():
    def __init__(self, c):
        self.c = c
        self.a = (1 - c * 0.5) / (1 + c * 0.5)
        self.b = 0.5 * (1 + self.a)


def dr_numpy(vertices, edges, fixed, loads, qpre,
             fpre=None, lpre=None, linit=None, E=None, radius=None,
             callback=None, callback_args=None, **kwargs):
    """Implementation of the dynamic relaxation method for form findong and analysis
    of articulated networks of axial-force members.

    Parameters
    ----------
    vertices : list
        XYZ coordinates of the vertices.
    edges : list
        Connectivity of the vertices.
    fixed : list
        Indices of the fixed vertices.
    loads : list
        XYZ components of the loads on the vertices.
    qpre : list
        Prescribed force densities in the edges.
    fpre : list, optional
        Prescribed forces in the edges.
    lpre : list, optinonal
        Prescribed lengths of the edges.
    linit : list, optional
        Initial length of the edges.
    E : list, optional
        Stiffness of the edges.
    radius : list, optional
        Radius of the edges.
    callback : callable, optional
        User-defined function that is called at every iteration.
    callback_args : tuple, optional
        Additional arguments passed to the callback.

    Returns
    -------
    xyz : array
        XYZ coordinates of the equilibrium geometry.
    q : array
        Force densities in the edges.
    f : array
        Forces in the edges.
    l : array
        Lengths of the edges
    r : array
        Residual forces.

    Notes
    -----
    For more info, see [1]_.

    References
    ----------
    .. [1] De Laet L., Veenendaal D., Van Mele T., Mollaert M. and Block P.,
           *Bending incorporated: designing tension structures by integrating bending-active elements*,
           Proceedings of Tensinet Symposium 2013,Istanbul, Turkey, 2013.

    Examples
    --------
    >>>
    """
    # --------------------------------------------------------------------------
    # callback
    # --------------------------------------------------------------------------
    if callback:
        assert callable(callback), 'The provided callback is not callable.'
    # --------------------------------------------------------------------------
    # configuration
    # --------------------------------------------------------------------------
    kmax = kwargs.get('kmax', 10000)
    dt = kwargs.get('dt', 1.0)
    tol1 = kwargs.get('tol1', 1e-3)
    tol2 = kwargs.get('tol2', 1e-6)
    coeff = Coeff(kwargs.get('c', 0.1))
    ca = coeff.a
    cb = coeff.b
    # --------------------------------------------------------------------------
    # attribute lists
    # --------------------------------------------------------------------------
    num_v = len(vertices)
    num_e = len(edges)
    free = list(set(range(num_v)) - set(fixed))
    # --------------------------------------------------------------------------
    # input processing
    # --------------------------------------------------------------------------

    def init_array(array, length):
        if array is None or len(array) == 0:
            return zeros((length,), dtype=float)
        return array

    qpre = init_array(qpre, num_e)
    fpre = init_array(fpre, num_e)
    lpre = init_array(lpre, num_e)
    linit = init_array(linit, num_e)
    E = init_array(E, num_e)
    radius = init_array(radius, num_e)
    # --------------------------------------------------------------------------
    # attribute arrays
    # --------------------------------------------------------------------------
    x = array(vertices, dtype=float).reshape((-1, 3))                      # m
    p = array(loads, dtype=float).reshape((-1, 3))                         # kN
    qpre = array(qpre, dtype=float).reshape((-1, 1))
    fpre = array(fpre, dtype=float).reshape((-1, 1))                       # kN
    lpre = array(lpre, dtype=float).reshape((-1, 1))                       # m
    linit = array(linit, dtype=float).reshape((-1, 1))                     # m
    E = array(E, dtype=float).reshape((-1, 1))                             # kN/mm2 => GPa
    radius = array(radius, dtype=float).reshape((-1, 1))                   # mm
    # --------------------------------------------------------------------------
    # sectional properties
    # --------------------------------------------------------------------------
    A = 3.14159 * radius ** 2                                              # mm2
    EA = E * A                                                             # kN
    # --------------------------------------------------------------------------
    # create the connectivity matrices
    # after spline edges have been aligned
    # --------------------------------------------------------------------------
    C = connectivity_matrix(edges, 'csr')
    Ct = C.transpose()
    Ci = C[:, free]
    Cit = Ci.transpose()
    Ct2 = Ct.copy()
    Ct2.data **= 2
    # --------------------------------------------------------------------------
    # if none of the initial lengths are set,
    # set the initial lengths to the current lengths
    # --------------------------------------------------------------------------
    if all(linit == 0):
        linit = normrow(C.dot(x))
    # --------------------------------------------------------------------------
    # initial values
    # --------------------------------------------------------------------------
    q = ones((num_e, 1), dtype=float)
    l = normrow(C.dot(x))  # noqa: E741
    f = q * l
    v = zeros((num_v, 3), dtype=float)
    r = zeros((num_v, 3), dtype=float)
    # --------------------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------------------

    def rk(x0, v0, steps=2):
        def a(t, v):
            dx = v * t
            x[free] = x0[free] + dx[free]
            # update residual forces
            r[free] = p[free] - D.dot(x)
            return cb * r / mass

        if steps == 1:
            return a(dt, v0)

        if steps == 2:
            B = [0.0, 1.0]
            K0 = dt * a(K[0][0] * dt, v0)
            K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0)
            dv = B[0] * K0 + B[1] * K1
            return dv

        if steps == 4:
            B = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
            K0 = dt * a(K[0][0] * dt, v0)
            K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0)
            K2 = dt * a(K[2][0] * dt, v0 + K[2][1] * K0 + K[2][2] * K1)
            K3 = dt * a(K[3][0] * dt, v0 + K[3][1] * K0 + K[3][2] * K1 + K[3][3] * K2)
            dv = B[0] * K0 + B[1] * K1 + B[2] * K2 + B[3] * K3
            return dv

        raise NotImplementedError

    # --------------------------------------------------------------------------
    # start iterating
    # --------------------------------------------------------------------------
    for k in range(kmax):
        # print(k)

        q_fpre = fpre / l
        q_lpre = f / lpre
        q_EA = EA * (l - linit) / (linit * l)
        q_lpre[isinf(q_lpre)] = 0
        q_lpre[isnan(q_lpre)] = 0
        q_EA[isinf(q_EA)] = 0
        q_EA[isnan(q_EA)] = 0

        q = qpre + q_fpre + q_lpre + q_EA
        Q = diags([q[:, 0]], [0])
        D = Cit.dot(Q).dot(C)
        mass = 0.5 * dt ** 2 * Ct2.dot(qpre + q_fpre + q_lpre + EA / linit)
        # RK
        x0 = x.copy()
        v0 = ca * v.copy()
        dv = rk(x0, v0, steps=4)
        v[free] = v0[free] + dv[free]
        dx = v * dt
        x[free] = x0[free] + dx[free]
        # update
        u = C.dot(x)
        l = normrow(u)  # noqa: E741
        f = q * l
        r = p - Ct.dot(Q).dot(u)
        # crits
        crit1 = norm(r[free])
        crit2 = norm(dx[free])
        # callback
        if callback:
            callback(k, x, [crit1, crit2], callback_args)
        # convergence
        if crit1 < tol1:
            break
        if crit2 < tol2:
            break
    return x, q, f, l, r


dva = {
    'is_fixed': False,
    'px': 0.0,
    'py': 0.0,
    'pz': 0.0,
    'rx': 0.0,
    'ry': 0.0,
    'rz': 0.0,
}

dea = {
    'qpre': 1.0,
    'fpre': 0.0,
    'lpre': 0.0,
    'linit': 0.0,
    'E': 0.0,
    'radius': 0.0,
}

folder = os.path.abspath("/Users/duch/Documents/Github/phd/pneu_fofin/data")
# file_name = "shell_four_corners.obj"
# file_path = os.path.join(folder, file_name)
file_network = os.path.join(folder, "network.json")

network = Network.from_json(file_network)

network.update_default_node_attributes(dva)
network.update_default_edge_attributes(dea)

for key in network.nodes():
    if network.degree(key) <= 2:
        network.node_attribute(key, 'is_fixed', True)

k_i = network.key_index()

# network.node_attribute(0, 'is_fixed', True)

bending_vs = [49, 60, 55, 42, 16, 41, 0, 15, 52, 59, 50]
bending_edges = zip(bending_vs[:-1], bending_vs[1:])
for u, v in bending_edges:
    if network.has_edge(u, v):
        pass
    else:
        u, v = v, u
    
    # network.edge_attribute((u, v), 'E', 10.0)
    # network.edge_attribute((u, v), 'radius', 0.5)
    network.edge_attribute((u, v), 'lpre', 2.5)
    
for u, v in network.edges():
    if network.has_edge(u, v):
        pass
    else:
        u, v = v, u
    length = network.edge_length(u, v)
    network.edge_attribute((u, v), 'linit', length)

xyz      = network.nodes_attributes(('x', 'y', 'z'))
edges    = [(k_i[u], k_i[v]) for u, v in network.edges()]
fixed    = [k_i[key] for key in network.nodes_where({'is_fixed': True})]
loads    = network.nodes_attributes(('px', 'py', 'pz'))
qpre     = network.edges_attribute('qpre')
fpre     = network.edges_attribute('fpre')
lpre     = network.edges_attribute('lpre')
linit    = network.edges_attribute('linit')
E        = network.edges_attribute('E')
radius   = network.edges_attribute('radius')
print(linit, E, radius)

# xyz = network.nodes_attributes(['x', 'y', 'z'])
# edges = list(network.edges())
# fixed = list(network.nodes_where({'is_fixed': True}))
# free = list(network.nodes_where({'is_fixed': False}))
# loads = network.nodes_attributes(['px', 'py', 'pz'])
# qpre = network.edges_attribute('qpre')

# result: xyz, q, f, l, r 
xyz, q, f, l, r = dr_numpy(xyz, edges, fixed, loads,
                            qpre, fpre, lpre,
                            linit, E, radius,
                            kmax=100)
# xyz, q, f, l, r = fd_numpy(xyz, edges, fixed, qpre, loads) 
        
for i, key in enumerate(network.nodes()):
    network.node_attributes(key, ['x', 'y', 'z'], xyz[i])
    network.node_attribute(key, 'residual', r[i])
for i, (u, v) in enumerate(network.edges()):
    network.edge_attribute((u, v), 'fpre', f[i])
    network.edge_attribute((u, v), 'qpre', q[i])
    network.edge_attribute((u, v), 'lpre', l[i])

viewer = App(width=800, height=800)
viewer.view.camera.rz = -300
viewer.view.camera.rx = -600
viewer.view.camera.tx = 0
viewer.view.camera.distance = 80
viewer.add(network)

for vkey in network.nodes_where({'is_fixed': True}):
    viewer.add(Sphere(network.node_coordinates(vkey), 0.5), color=(1, 0, 0))

viewer.show()