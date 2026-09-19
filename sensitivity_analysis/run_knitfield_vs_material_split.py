"""Isolate the two contributions to 183.6 -> 192.1 mm: material split vs knit field.

2x2: {uniform structure 1, two-material split} x {theta=0 uniform, FDM per-face field}.
Region 0 = left half (x<0) = structure 1; region 1 = right half = structure 2,
matching the face grouping in data/two_pattern/disc_two_pattern_nu065_p1000.obj.
"""
import csv, json, os, subprocess
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT="/home/chaoyu/phd_simulation"
SCR="/tmp/claude-1001/-home-chaoyu-phd-simulation/f09b748b-0bf0-48b4-9695-28198fe18583/scratchpad"
MESH=f"{ROOT}/data/circular_flat.off"; NREG=f"{ROOT}/build/fem_batch_nregion"
OUT=f"{SCR}/split_runs"; os.makedirs(OUT, exist_ok=True)

S1=(10300.0,13400.0,0.58); S2=(7700.0,7600.0,0.65)
PRESSURES=[200.,400.,600.,800.,1000.,1200.,1400.,1600.]

L=open(MESH).read().split("\n"); nV,nF,_=map(int,L[1].split())
V=np.array([[float(x) for x in L[2+i].split()] for i in range(nV)])
F=np.array([[int(x) for x in L[2+nV+i].split()[1:4]] for i in range(nF)])
cx=V[F,0].mean(axis=1)
face_reg_split=(cx>=0).astype(int)            # 0 = left = S1, 1 = right = S2
face_reg_uni=np.zeros(nF,int)

fld=np.loadtxt(f"{ROOT}/FDM/data/2part/circle_face_directional_field.txt",delimiter=",")
th_fdm=np.degrees(np.arctan2(fld[:,1],fld[:,0]))
th_uni=np.full(nF,90.0)                        # == theta_knit 0 in the sensitivity convention
print(f"split: {(~(cx>=0)).sum()} left / {(cx>=0).sum()} right faces")

def one(job):
    mat,fieldname,p=job
    regs=face_reg_split if mat=="split" else face_reg_uni
    ang =th_fdm if fieldname=="fdm" else th_uni
    tag=f"{mat}_{fieldname}_p{int(p)}"
    mj=f"{OUT}/{tag}_map.json"; pj=f"{OUT}/{tag}_params.json"; pref=f"{OUT}/{tag}"
    json.dump({"face_regions":regs.tolist(),"face_knit_dirs_deg":ang.tolist()},open(mj,"w"))
    R=[{"sf_wale":1.0,"sf_course":1.0,"knit_dir_deg":90.0,"E1":S1[0],"E2":S1[1],"nu":S1[2]}]
    if mat=="split":
        R.append({"sf_wale":1.0,"sf_course":1.0,"knit_dir_deg":90.0,"E1":S2[0],"E2":S2[1],"nu":S2[2]})
    json.dump({"pressure":p,"motif":1,"E1":S1[0],"E2":S1[1],"nu":S1[2],"regions":R},open(pj,"w"))
    r=subprocess.run([NREG,MESH,mj,pj,pref],capture_output=True,text=True,timeout=600)
    if r.returncode!=0: raise RuntimeError(f"{tag}: {r.stderr[-300:]}")
    row=next(csv.DictReader(open(pref+"_scalars.csv")))
    return {"material":mat,"field":fieldname,"pressure":p,**{k:float(v) for k,v in row.items()}}

jobs=[(m,f,p) for m in ("uniform","split") for f in ("uniform0","fdm") for p in PRESSURES]
rows=[]
with ProcessPoolExecutor(max_workers=12) as ex:
    for fu in as_completed([ex.submit(one,j) for j in jobs]): rows.append(fu.result())
df=pd.DataFrame(rows).sort_values(["material","field","pressure"])
df.to_csv(f"{SCR}/split_compare.csv",index=False)
piv=df.pivot_table(index="pressure",columns=["material","field"],values="crown_height")*1000
print("\ncrown height [mm]"); print(piv.round(2).to_string())
