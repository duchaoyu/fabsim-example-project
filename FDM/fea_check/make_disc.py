"""Flat clamped disc, radius 0.5 m, for the pressure benchmark (OFF for fabsim, INP mesh for CalculiX)."""
import numpy as np, triangle, json
a, nb = 0.5, 128
th = np.linspace(0, 2*np.pi, nb, endpoint=False)
P = np.c_[a*np.cos(th), a*np.sin(th)]
S = np.c_[np.arange(nb), (np.arange(nb)+1) % nb]
T = triangle.triangulate({"vertices": P, "segments": S}, "pq30a0.0004")
V = np.c_[T["vertices"], np.zeros(len(T["vertices"]))]; F = T["triangles"]
n = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]]); F[n[:,2] < 0] = F[n[:,2] < 0][:, ::-1]
with open("disc.off","w") as f:
    f.write(f"OFF\n{len(V)} {len(F)} 0\n"); [f.write("%.10f %.10f %.10f\n"%tuple(v)) for v in V]; [f.write(f"3 {t[0]} {t[1]} {t[2]}\n") for t in F]
rim = [i for i in range(len(V)) if abs(np.hypot(*V[i,:2]) - a) < 1e-9]
centre = int(np.argmin(np.hypot(V[:,0], V[:,1])))
json.dump({"rim": rim, "centre": centre, "nv": len(V), "nf": len(F)}, open("disc.json","w"))
json.dump({"face_regions": [0]*len(F), "face_knit_dirs_deg": [0.0]*len(F)}, open("disc_map.json","w"))
with open("disc_mesh.inp","w") as f:
    f.write("*NODE, NSET=NALL\n"); [f.write(f"{i+1}, {v[0]:.10f}, {v[1]:.10f}, {v[2]:.10f}\n") for i,v in enumerate(V)]
    f.write("*ELEMENT, TYPE=S3, ELSET=EALL\n"); [f.write(f"{k+1}, {t[0]+1}, {t[1]+1}, {t[2]+1}\n") for k,t in enumerate(F)]
    f.write("*NSET, NSET=RIM\n"); [f.write(f"{i+1},\n") for i in rim]
    f.write("*ELSET, ELSET=EPRES\n"); [f.write(f"{k+1},\n") for k in range(len(F))]
print(f"disc: {len(V)} nodes, {len(F)} triangles, {len(rim)} rim nodes, centre node {centre} at r={np.hypot(*V[centre,:2]):.2e}")
