import numpy as np, sys
sys.path.insert(0,'/home/claude/sim')
from post import *
SIZES=(4,5,6,7)

def lg4_fit(run,g0,gstar,qmax_rel=0.55):
    xs=[];ys=[]
    for u,r in run['u'].items():
        if u==0: continue
        if 1e-9<r['q']<=qmax_rel*g0 and r['chi_b'][1]>0:
            xs.append(r['q']/g0); ys.append(1.0/r['chi_b'][1])
    xs=np.array(xs); ys=np.array(ys)
    if len(np.unique(np.round(xs,4)))<4: return None
    A=np.stack([np.ones_like(xs),xs**2,xs**4],1)
    beta,*_=np.linalg.lstsq(A,ys,rcond=None)
    res=ys-A@beta; dof=max(len(xs)-3,1); s2=np.sum(res**2)/dof
    cov=np.linalg.inv(A.T@A)*s2
    c0,c4=beta[0],beta[2]
    l4=c4/c0/g0**4
    # error propagation of ratio
    var=(l4**2)*(cov[2,2]/c4**2+cov[0,0]/c0**2-2*cov[0,2]/(c0*c4))
    return dict(l4=float(l4),sig=float(np.sqrt(abs(var))),Lam4=float(l4/gstar**2),n=int(len(xs)),c0=float(c0))

def bz_obs(run,geo=None):
    ud=run['u']; us=[u for u in ud if u!=0]
    th=[theta_obs(ud[u]) for u in us]
    o={k:float(np.mean([t[k] for t in th])) for k in ('E2','E4','M24','M24F','sig2','f4pt','f4','shift')}
    uR=roton_u(run); po=point_obs(run,uR)
    o.update(E_R=po['E_R'],uR=uR,F4_R=po['F4'],q_R=po['q'],f4_R=po['f4'],M24_R=po['M24'])
    o['kappa_min']=float(min(ud[u]['kappa'] for u in us))
    o['capt']=float(np.mean([ud[u]['captured'][1:].mean() for u in us]))
    o['spread']=float(run['spread']); o['gap']=float(run['gap'])
    if geo is not None:
        l=lg4_fit(run,geo['_g0'],geo['gstar'])
        o['lg4']=l['l4'] if l else np.nan; o['lg4_sig']=l['sig'] if l else np.nan; o['lg4_n']=l['n'] if l else 0
    return o

def collect(name,lam=1.0,fam='A',tw=(0.0,0.0),gm=0):
    d=load(name); g0=None
    geo=dict(d['geo']); 
    from band import Band,Params
    pk=dict(d['spec'].get('p',{})); pk.pop('Gcut_geo',None)
    geo['_g0']=Band(Params(**{**dict(Gcut=3.1),**pk})).g0
    out={}
    for Ne in SIZES:
        key=(lam,fam,Ne,tw,gm)
        if key in d['runs']: out[Ne]=bz_obs(d['runs'][key],geo)
    return d,geo,out
