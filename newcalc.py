"""Robustness calculations: random-gauge audit of D, Chern numbers on finer twist grids, strong screening, mean-|F| twin, N_e=8 subset."""
import numpy as np, pickle, os, sys, time
sys.path.insert(0,'/home/claude/sim')
from joblib import Parallel, delayed
OUT='/home/claude/sim/res4'
def P(name):
    from ana import Params
    spec=pickle.load(open(f'/home/claude/sim/res2/{name}.pkl','rb'))['spec']
    return Params(**{**dict(Gcut=3.1,smap='exp'),**spec.get('p',{})})
def save(tag,obj): pickle.dump(obj,open(f'{OUT}/{tag}.pkl','wb')); return tag
def done(tag): return os.path.exists(f'{OUT}/{tag}.pkl')

def gauge(name,Ne,seed):
    tag=f'gauge_{name}_{Ne}_{seed}'
    if done(tag): return tag
    import ana, current
    from ana import Band, get_cluster
    b=Band(P(name)); cl=get_cluster('A',Ne)
    current.System=ana.System
    if seed>=0:
        Base=ana.System
        class GS(Base):
            def __init__(self,*a,**k):
                k['gauge_rng']=np.random.default_rng(seed); super().__init__(*a,**k)
        current.System=GS
    t=time.time(); avg,mem=current.manifold_current(b,cl,None,dense_max=1200)
    return save(tag,dict(avg=avg,D=[m['D'].tolist() for m in mem],time=time.time()-t))

def chern(name,Ne,M):
    tag=f'chern_{name}_{Ne}_{M}'
    if done(tag): return tag
    from ana import Band, get_cluster
    from topo import manifold_chern
    b=Band(P(name)); cl=get_cluster('A',Ne); t=time.time()
    r=manifold_chern(b,cl,M=M)
    return save(tag,dict(r=r,time=time.time()-t))

def cur(name,fam,Ne,lam=None,d=300.,tag=None):
    tag=tag or f'cur_{name}_{fam}{Ne}_d{int(d)}'
    if done(tag): return tag
    from ana import Band, get_cluster
    import ana, current
    current.System=ana.System
    from current import manifold_current
    from refband import hom_lam
    b=Band(P(name)); cl=get_cluster(fam,Ne); t=time.time()
    lf=None if lam is None else hom_lam(lam,-1)
    avg,mem=manifold_current(b,cl,lf,dense_max=1200,d=d)
    for m in mem:
        for k in ('E','w'): m.pop(k,None)
    return save(tag,dict(avg=avg,mem=mem,time=time.time()-t))

if __name__=='__main__':
    which=sys.argv[1]
    if which=='quick':
        jobs=[delayed(gauge)('A',6,s) for s in (-1,1,2)]+[delayed(gauge)('P3',6,s) for s in (-1,1)]
        jobs+=[delayed(chern)(n,5,M) for n in ('A','P3') for M in (8,10)]
        jobs+=[delayed(cur)(n,'A',6,None,50.) for n in ('A','P3')]
        jobs+=[delayed(cur)('A','A',6,'twin',300.,'cur_A_meantwin6')]
    else:
        jobs=[delayed(cur)(n,'A',8) for n in ('P2','sA0.06','w1.1','psi117.7','w0.8','psi97.7','V1.1','th4.3','w1.2','psi112.7')]
    for r in Parallel(n_jobs=2,verbose=5)(jobs): print(r,flush=True)
