import sys,json; sys.path.insert(0,'/home/claude/sim')
import numpy as np
from joblib import Parallel, delayed
from descr import *
from geom import geometry
P={'base':dict(),'P1':dict(psi_deg=112.7,V=20.09502054201356,w=-24.55007216238367),
   'P2':dict(psi_deg=102.7,V=21.855273509357964,w=-23.0655798993469),
   'P3':dict(psi_deg=117.7,V=21.84161412098515,w=-22.97593932018246),
   'P4':dict(mstar=0.70,V=18.42869631175722,w=-21.076463671887964),
   'P5':dict(mstar=0.55,V=23.436332692743697,w=-26.82334013128753)}
def job(nm):
    par=Params(Gcut=3.1,**P[nm]); b=Band(par)
    d=describe(par,24)
    G8=geometry(b,8,0.45,7,8); G10=geometry(b,10,0.45,8,8)
    f=np.arange(12)/12; F1,F2=np.meshgrid(f,f,indexing='ij'); e,_=b.top(b.frac2cart(np.stack([F1.ravel(),F2.ravel()],1)),2)
    d.update(gstar8=float(G8['gstar']),tau0_8=float(G8['tau0']),gstar10=float(G10['gstar']),tau0_10=float(G10['tau0']),
             tau2_10=float(G10['tau2']),tau4_10=float(G10['tau4']),bandwidth=float(np.ptp(e[:,0])),iso_gap=float((e[:,0]-e[:,1]).min()),p=P[nm])
    return nm,d
res=dict(Parallel(n_jobs=2)(delayed(job)(n) for n in P))
json.dump(res,open('partners.json','w'),indent=1)
for n,d in res.items():
    print(n,'g*8 %.3f t0_8 %.4f | g*10 %.3f t0_10 %.4f t2 %.4f t4 %.4f | sOm %.3f strg %.3f TV %.3f | F(g) %.3f/%.3f F(M) %.3f/%.3f | W %.1f gap %.1f C %.0f'%(
      d['gstar8'],d['tau0_8'],d['gstar10'],d['tau0_10'],d['tau2_10'],d['tau4_10'],d['sigma_Omega'],d['sigma_trg'],d['trace_violation'],
      *d['F']['g'],*d['F']['M (g/2)'],d['bandwidth'],d['iso_gap'],d['C']))
