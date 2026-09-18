"""Knit-direction field of the two-pattern disc as OBJ line segments.

Two files: the field on the flat rest disc, and the same material direction
pushed forward onto the inflated surface by each triangle's deformation
gradient - which is what the solver's face directions actually become.
"""
import numpy as np, pandas as pd, os
ROOT="/home/chaoyu/phd_simulation"
SP="/tmp/claude-1001/-home-chaoyu-phd-simulation/9d0d8cd6-113f-4f55-9664-3423baeb9740/scratchpad"
OUT=f"{ROOT}/data/two_pattern"
L=open(f"{ROOT}/data/circular_flat.off").read().split("\n")
nv,nf=int(L[1].split()[0]),int(L[1].split()[1])
V0=np.array([[float(x) for x in L[2+i].split()] for i in range(nv)])
F=np.array([[int(x) for x in L[2+nv+i].split()[1:4]] for i in range(nf)])
D=np.array([[float(v) for v in l.replace(',',' ').split()]
            for l in open(f"{ROOT}/FDM/data/2part/circle_face_directional_field.txt") if l.strip()])
assert len(D)==nf
Vd=pd.read_csv(f"{SP}/runs/z_A_right65_1000_verts.csv").sort_values("vid")[["x","y","z"]].to_numpy()
mean_edge=np.mean([np.linalg.norm(V0[F[:,i]]-V0[F[:,(i+1)%3]],axis=1).mean() for i in range(3)])
half=min(0.8*mean_edge, 0.02)/2
print(f"mean edge {1000*mean_edge:.1f} mm, segment length {2000*half:.1f} mm")

def pushforward(V):
    """Per-face direction and centroid on the surface V."""
    dirs=np.zeros((nf,3)); cen=np.zeros((nf,3))
    for f in range(nf):
        a,b,c=F[f]
        # rest basis
        e1,e2=V0[b]-V0[a], V0[c]-V0[a]
        E1,E2=V[b]-V[a],  V[c]-V[a]
        # barycentric coords of the rest direction in (e1,e2)
        A=np.array([[e1@e1,e1@e2],[e1@e2,e2@e2]])
        rhs=np.array([e1@D[f], e2@D[f]])
        u,v=np.linalg.solve(A,rhs)
        d=u*E1+v*E2
        n=np.linalg.norm(d)
        dirs[f]=d/n if n>1e-12 else np.array([1.,0.,0.])
        cen[f]=(V[a]+V[b]+V[c])/3
    return dirs,cen

for tag,V in (("flat",V0),("p1000",Vd)):
    d,c=pushforward(V)
    P0,P1=c-half*d, c+half*d
    with open(f"{OUT}/disc_knit_dir_{tag}.obj","w") as fo:
        fo.write("# knit (wale) direction, one segment per face, centred on the face centroid\n")
        fo.write(f"# source FDM/data/2part/circle_face_directional_field.txt, {nf} faces, metres\n")
        fo.write("# surface: "+("flat rest disc data/circular_flat.off\n" if tag=="flat" else
                 "inflated two-pattern disc at 1000 Pa (disc_two_pattern_nu065_p1000.obj)\n"))
        for p0,p1 in zip(P0,P1):
            fo.write(f"v {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}\nv {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
        fo.write("g knit_direction\n")
        for i in range(nf): fo.write(f"l {2*i+1} {2*i+2}\n")
    print("wrote", f"{OUT}/disc_knit_dir_{tag}.obj")
