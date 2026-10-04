"""Bragg-replica (umklapp) weight of the magnetoroton for all moire bands and their twins:
R = sum_{|Q|l>qmin} |<low_{K0+Q}|rho_Q|0>|^2 / sum_{|Q|l>qmin} S(Q), lowest non-manifold state of the target sector."""
import numpy as np, pickle, os, sys, glob
sys.path.insert(0,'/home/claude/sim')
from joblib import Parallel, delayed
OUT='/home/claude/sim/res3/rep'; os.makedirs(OUT,exist_ok=True)
def job(name,variant,Ne,qmin=2.6):
    fn=f'{OUT}/{name}_{variant}_{Ne}.pkl'
    if os.path.exists(fn): return fn
    from ana import get_cluster, System, Band, Params
    from refband import hom_lam
    from scipy.sparse.linalg import eigsh
    spec=pickle.load(open(f'/home/claude/sim/res2/{name}.pkl','rb'))['spec']
    par=Params(**{**dict(Gcut=3.1,smap='exp'),**spec.get('p',{})}); b=Band(par)
    lam={'main':(None if spec.get('kind','moire')=='moire' else hom_lam(spec['kind'],-1)),'twin':hom_lam('rms',-1)}[variant]
    cl=get_cluster('A',Ne); S=System(b,cl,lam_fn=lam); ell=np.sqrt(b.Auc/(2*np.pi))
    E={};V={}
    for K in range(cl.Nk):
        e,v=eigsh(S.H(K),k=3,which='SA',tol=1e-12,ncv=30); o=np.argsort(e); E[K]=e[o]; V[K]=v[:,o]
    man=sorted(range(cl.Nk),key=lambda K:E[K][0])[:3]
    rows=[]
    for K0 in man:
        g=V[K0][:,0]
        for iq in range(len(S.Qs)):
            q=S.Qs[iq,4]*ell
            if q<1e-9: continue
            u=int(S.Qs[iq,0]); Kt=cl.add(K0,u); O,_=S.rho_op(K0,S.Lam[iq],u); v=O@g
            ex=[0] if Kt in man else []
            for i in ex: v=v-V[Kt][:,i]*np.vdot(V[Kt][:,i],v)
            lo=1 if Kt in man else 0
            rows.append((q,np.vdot(v,v).real/Ne,abs(np.vdot(V[Kt][:,lo],v))**2/Ne,E[Kt][lo]-E[K0][0]))
    rows=np.array(rows); m=rows[:,0]>qmin
    out=dict(name=name,variant=variant,Ne=Ne,rows=rows,R=float(rows[m,2].sum()/rows[m,1].sum()),Rabs=float(rows[m,2].mean()))
    pickle.dump(out,open(fn,'wb')); return fn
if __name__=='__main__':
    names=sorted(os.path.basename(f)[:-4] for f in glob.glob('/home/claude/sim/res2/*.pkl') if not f.endswith('LL.pkl'))
    Ne=int(sys.argv[1]); nj=int(sys.argv[2]) if len(sys.argv)>2 else 1
    jobs=[(n,v,Ne) for n in names for v in ('main','twin')]+[('LLL','main',Ne)]
    for r in Parallel(n_jobs=nj,verbose=5)(delayed(job)(*j) for j in jobs): pass
    print('done')
