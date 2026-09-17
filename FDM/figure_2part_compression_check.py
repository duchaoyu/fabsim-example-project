import os, json, csv, numpy as np, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
D="FDM/data/2part"
L=open(f"{D}/2parts_smooth_tri_m.off").read().split("\n")
nv,nf=int(L[1].split()[0]),int(L[1].split()[1])
V=np.array([[float(x) for x in L[2+i].split()] for i in range(nv)])
F=np.array([[int(x) for x in L[2+nv+i].split()[1:4]] for i in range(nf)])
rows=list(csv.reader(open("FDM/optimisation/2part_K_p2_00599_stress.csv")))
S=np.array([[float(x) for x in r] for r in rows[1:]])
p1,p2=S[:,5],S[:,6]
cen=V[F].mean(1)
# face areas
a=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)/2
A=a.sum()
uni = (p2<0)&(p1>=0)      # one principal compressive -> uniaxial wrinkling
bia = (p1<0)              # both compressive -> slack / collapse
ten = (p2>=0)
for nm,m in [("pure tension",ten),("uniaxial compression (wrinkles)",uni),("biaxial compression (slack)",bia)]:
    print(f"  {nm:34s} {m.mean()*100:5.1f}% of faces, {a[m].sum()/A*100:5.1f}% of area")
fan=cen[:,1]<=-0.33
print(f"\n  in the fan (y<=-0.33): {(p2[fan]<0).mean()*100:.1f}% of faces compressive "
      f"vs {(p2[~fan]<0).mean()*100:.1f}% elsewhere")
print(f"  worst compressive principal stress: {p2.min():.0f} Pa (tensile max {p1.max():.0f})")

tri=Triangulation(V[:,0],V[:,1],F)
fig,axes=plt.subplots(1,2,figsize=(13,6))
ax=axes[0]
c=np.zeros(nf); c[uni]=1; c[bia]=2
tp=ax.tripcolor(tri,facecolors=c,cmap=matplotlib.colors.ListedColormap(["#f7f3e8","#f2a93b","#c0392b"]),vmin=0,vmax=2)
cab=json.load(open(f"{D}/cable_paths_2part_continuous.json"))
for p in cab.values(): ax.plot(V[p][:,0],V[p][:,1],color="#2a78d6",lw=2.0)
ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
ax.set_title("cream = pure tension\norange = wrinkling (one principal < 0)\nred = slack (both < 0)",fontsize=9)
ax=axes[1]
tp=ax.tripcolor(tri,facecolors=np.minimum(p2,0),cmap="Reds_r",vmax=0)
for p in cab.values(): ax.plot(V[p][:,0],V[p][:,1],color="#2a78d6",lw=2.0)
ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
ax.set_title("minor principal stress where compressive, Pa",fontsize=9)
fig.colorbar(tp,ax=ax,fraction=0.045)
fig.suptitle("2-part smooth, best 4-cable fit (6.87 mm) — where the membrane is in compression",fontsize=12)
fig.tight_layout(); fig.savefig(f"{D}/2part_compression_check.png",dpi=150,bbox_inches="tight")
print("\nSaved",f"{D}/2part_compression_check.png")
