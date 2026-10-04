"""Microscopic neutral dispersion, projected static structure factor S(Q), oscillator strength f(Q), SMA energy and
dynamic structure factor S(Q,w) (Lanczos continued fraction) for moire bands, their homogeneous twins and the LLL.
Manifold states are projected out of rho_Q|0> (topological-sector weight, which vanishes in the thermodynamic limit)."""
import numpy as np, pickle, os, sys, time
sys.path.insert(0,'/home/claude/sim')
from joblib import Parallel, delayed
OUT='/home/claude/sim/res3'

def lanczos(H,v,E0,m=160):
    nrm=np.linalg.norm(v); q=v/nrm; qp=np.zeros_like(q); b=0.; al=[];be=[]
    for j in range(m):
        w=H@q-E0*q; a=np.vdot(q,w).real; w=w-a*q-b*qp
        al.append(a); b=np.linalg.norm(w); be.append(b)
        if b<1e-10: break
        qp=q; q=w/b
    return np.array(al),np.array(be[:-1]),nrm**2

def job(name,variant,Ne,Qmax=5.0):
    fn=f'{OUT}/{name}_{variant}_{Ne}.pkl'
    if os.path.exists(fn): return fn
    from ana import get_cluster, System, Band, Params
    from refband import hom_lam
    from scipy.sparse.linalg import eigsh
    t0=time.time()
    spec=pickle.load(open(f'/home/claude/sim/res2/{name}.pkl','rb'))['spec']
    par=Params(**{**dict(Gcut=3.1,smap='exp'),**spec.get('p',{})}); b=Band(par)
    lam={'main':(None if spec.get('kind','moire')=='moire' else hom_lam(spec['kind'],-1)),'twin':hom_lam('rms',-1)}[variant]
    cl=get_cluster('A',Ne); S=System(b,cl,lam_fn=lam)
    l2=b.Auc/(2*np.pi); ell=np.sqrt(l2)
    # lowest states per sector
    lows={}; vecs={}
    for K in range(cl.Nk):
        H=S.H(K); e,v=eigsh(H,k=4,which='SA',tol=1e-12,ncv=40); o=np.argsort(e); lows[K]=e[o]; vecs[K]=v[:,o[0]]
    order=sorted(range(cl.Nk),key=lambda K:lows[K][0]); man=order[:3]
    E0s={K:lows[K][0] for K in man}; Etop=max(E0s.values())
    disp=[]; 
    for K in range(cl.Nk):
        disp.append((K,(lows[K][1] if K in man else lows[K][0])-Etop))
    # momentum transfers
    sel=[iq for iq in range(len(S.Qs)) if 1e-9<S.Qs[iq,4]*ell<Qmax]
    pairs={}   # target sector -> list of (K0,iq)
    for K0 in man:
        for iq in sel:
            u=int(S.Qs[iq,0]); Kt=cl.add(K0,u); pairs.setdefault(Kt,[]).append((K0,iq))
    res=[]
    for Kt,lst in pairs.items():
        H=S.H(Kt)
        for (K0,iq) in lst:
            u=int(S.Qs[iq,0]); O,_=S.rho_op(K0,S.Lam[iq],u); v=O@vecs[K0]
            raw=np.vdot(v,v).real
            if Kt in man: v=v-vecs[Kt]*np.vdot(vecs[Kt],v)
            Sq=np.vdot(v,v).real; fq=np.vdot(v,H@v).real-E0s[K0]*Sq
            al,be,n2=lanczos(H,v,E0s[K0]) if Sq>1e-14 else (np.zeros(0),np.zeros(0),0.)
            res.append(dict(K0=K0,Kt=Kt,iq=iq,Q=S.Qs[iq,5:7]*ell,q=S.Qs[iq,4]*ell,S_raw=raw/Ne,S=Sq/Ne,f=fq/Ne,al=al,be=be,n2=n2/Ne))
    out=dict(name=name,variant=variant,Ne=Ne,ell=ell,man=man,E0=E0s,lows=lows,disp=disp,res=res,
             Kcart=[(b.frac2cart(np.array([[cl.X1[K]/cl.D,cl.X2[K]/cl.D]]))[0]*ell).tolist() for K in range(cl.Nk)],
             gb=[(b.b1*ell).tolist(),(b.b2*ell).tolist()],time=time.time()-t0)
    pickle.dump(out,open(fn,'wb')); return fn

if __name__=='__main__':
    jobs=[(n,v,Ne) for Ne in (6,7) for n,v in (('A','main'),('A','twin'),('P3','main'),('P3','twin'),('LLL','main'))]
    jobs+=[(n,v,8) for n,v in (('A','main'),('P3','main'),('A','twin'),('P3','twin'),('LLL','main'))]
    for r in Parallel(n_jobs=2,verbose=10)(delayed(job)(*j) for j in jobs): print(r,flush=True)
