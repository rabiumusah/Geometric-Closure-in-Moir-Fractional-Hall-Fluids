"""Generality test: twisted WSe2 continuum model (Devakul et al. parameters), nu=1/3, single-band projection."""
import numpy as np,pickle,sys,time,os
sys.path.insert(0,'/home/claude/sim')
from joblib import Parallel,delayed
OUT='/home/claude/sim/res4'
def run(theta,Ne=6,extra=None):
    tag=f'wse2_th{theta}_{Ne}'+('' if extra is None else f'_{extra}')
    if os.path.exists(f'{OUT}/{tag}.pkl'): return tag
    import ana,current; current.System=ana.System
    from ana import Params,Band,System,get_cluster
    from topo import low_eigs,pes,admissible_count,pes_counting
    from descr import describe
    from refband import hom_lam
    from current import manifold_current
    p=Params(theta_deg=theta,V=9.0,psi_deg=128.0,w=-18.0,mstar=0.43,a0=3.317,Gcut=3.1,smap='exp'); b=Band(p)
    out=dict(theta=theta,Ne=Ne)
    f=np.arange(12)/12; F1,F2=np.meshgrid(f,f,indexing='ij'); e,_=b.top(b.frac2cart(np.stack([F1.ravel(),F2.ravel()],1)),2)
    out.update(bw=float(np.ptp(e[:,0])),iso=float((e[:,0]-e[:,1]).min()))
    out['descr']={k:v for k,v in describe(p,24).items() if k!='F'}
    cl=get_cluster('A',Ne); S=System(b,cl)
    lows=np.array([low_eigs(S,K,3)[0] for K in range(cl.Nk)]); o=np.argsort(lows[:,0]); man=[int(x) for x in o[:3]]
    top=lows[o[2],0]; exc=np.array([(lows[K,1] if K in man else lows[K,0])-top for K in range(cl.Nk)])
    out.update(man=man,spread=float(top-lows[o[0],0]),gap=float(exc.min()))
    gsl=[(K,low_eigs(S,K,1)[1][:,0]) for K in man]; NA=3; xi,ks=pes(cl,gsl,NA); cnt=admissible_count(cl.Nk,NA); out['pes']=pes_counting(xi,cnt)
    if out['gap']>3*out['spread'] and out['pes']['largest']:
        avg,mem=manifold_current(b,cl,None,dense_max=1200); out['D']=avg
        avg,mem=manifold_current(b,cl,hom_lam('rms',-1),dense_max=1200); out['DT']=avg
        out['EC']=14.399645*1e3/(10*np.sqrt(b.Auc/(2*np.pi))); out['ell']=float(np.sqrt(b.Auc/(2*np.pi)))
    pickle.dump(out,open(f'{OUT}/{tag}.pkl','wb')); return tag
if __name__=='__main__':
    ths=[float(x) for x in sys.argv[1:]]
    for r in Parallel(n_jobs=1)(delayed(run)(t) for t in ths): print(r,flush=True)
