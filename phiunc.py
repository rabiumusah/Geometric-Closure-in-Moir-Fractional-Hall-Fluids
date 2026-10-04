import numpy as np,sys,pickle
sys.path.insert(0,'/home/claude/sim')
from band import *; from geom import geometry
out={}
for eps in (0.04,0.06,0.10):
    for phi in (0.0,30.0):
        for N in (8,10):
            b=Band(Params(Gcut=3.1,smap='exp',eps=eps,phi_deg=phi))
            G=geometry(b,N,0.45,7,8,rng=np.random.default_rng(0),boot=300)
            be=G['beta']; sd=G['sd_beta'] if 'sd_beta' in G else None
            a4c,a4s=be[6],be[7]; r2=a4c**2+a4s**2
            d4=0.25*np.sqrt((a4s*sd[6])**2+(a4c*sd[7])**2)/r2 if sd is not None else np.nan
            a2c,a2s=be[4],be[5]; d2=0.5*np.sqrt((a2s*sd[4])**2+(a2c*sd[5])**2)/(a2c**2+a2s**2) if sd is not None else np.nan
            out[(eps,phi,N)]=(float(np.degrees(G['phi2'])),float(np.degrees(d2)),float(np.degrees(G['phi4'])),float(np.degrees(d4)))
            print(eps,phi,N,np.round(out[(eps,phi,N)],2),flush=True)
pickle.dump(out,open('/home/claude/sim/res4/phiunc.pkl','wb'))
