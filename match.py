"""Matched-geometry partner bands: solve gbar*(V,w)=gbar*_0 and tau0(V,w)=tau0_0 at a different (psi, m*).
Geometry evaluated exactly as in production (Gcut 3.1, Nmesh 8, q-window 0.45, 7 q, order 8)."""
import numpy as np, sys, json, time
from joblib import Parallel, delayed
from band import Band, Params, chern
from geom import geometry

def geo(V,w,psi,m):
    b=Band(Params(Gcut=3.1,V=V,w=w,psi_deg=psi,mstar=m))
    G=geometry(b,8,0.45,7,8)
    return float(G['gstar']),float(G['tau0'])

def solve(psi,m,target,x0,maxit=8,h=(0.3,0.3),log=None):
    x=np.array(x0,float); hist=[]
    for it in range(maxit):
        pts=[x,x+[h[0],0],x+[0,h[1]]]
        res=Parallel(n_jobs=2)(delayed(geo)(p[0],p[1],psi,m) for p in pts)
        f0=np.array(res[0]); J=np.stack([(np.array(res[1])-f0)/h[0],(np.array(res[2])-f0)/h[1]],1)
        r=f0-np.array(target); rel=np.abs(r)/np.abs(target)
        hist.append(dict(it=it,V=x[0],w=x[1],gstar=f0[0],tau0=f0[1],rel=rel.tolist()))
        if log: print(psi,m,hist[-1],flush=True)
        if rel.max()<2e-3: break
        dx=-np.linalg.solve(J,r)
        step=np.clip(dx,-4,4); x=x+step
    return x,hist

if __name__=='__main__':
    target=(217.574,0.66476)
    cases=json.loads(sys.argv[1])
    out=[]
    for c in cases:
        x,h=solve(c['psi'],c['m'],target,(20.8,-23.8),log=True)
        out.append(dict(case=c,x=x.tolist(),hist=h))
        json.dump(out,open('match_out.json','w'),indent=1)
