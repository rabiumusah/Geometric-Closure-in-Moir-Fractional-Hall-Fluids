"""Topological certification of the FQAH manifold on the torus:
 - boundary twists in the cluster's flux convention  t = S^{-1} theta/(2 pi)  (b-basis fractions)
 - spectral flow under flux insertion
 - many-body Chern number of the three-fold manifold (Fukui-Hatsugai-Suzuki plaquettes on the twist torus,
   many-body overlaps built from cell-periodic orbital overlaps)
 - particle entanglement spectrum (PES) and (1,3)-admissible counting."""
import numpy as np, numba as nb, itertools, math
from scipy.sparse.linalg import eigsh
from ed import *

def twist_frac(cl,th1,th2):
    return np.linalg.solve(np.array(cl.S,float),np.array([th1,th2])/(2*np.pi))

def low_eigs(S,K,k):
    H=S.H(K); dim=H.shape[0]
    if dim<=600:
        e,v=np.linalg.eigh(H.toarray()); return e[:k],v[:,:k]
    e,v=eigsh(H,k=k,which='SA',tol=1e-11,ncv=max(30,4*k)); o=np.argsort(e); return e[o],v[:,o]

def make_system(band,cl,twist,lam_fn=None,v_ref=None,**kw):
    return System(band,cl,twist=tuple(twist),lam_fn=lam_fn,v_ref=v_ref,**kw)

# ------------------------------------------------------------------ spectral flow
def spectral_flow(band,cl,nflux=3,nstep=6,lam_fn=None,direction=0,k=3,**kw):
    ths=np.linspace(0,2*np.pi*nflux,nflux*nstep+1); out=[]
    for th in ths:
        t=twist_frac(cl,th if direction==0 else 0.0,th if direction==1 else 0.0)
        S=make_system(band,cl,t,lam_fn,**kw)
        row=[low_eigs(S,K,k)[0] for K in range(cl.Nk)]
        out.append(np.array(row))
    return ths,np.array(out)   # (nth, Nk, k)

# ------------------------------------------------------------------ many-body overlaps
@nb.njit(cache=True)
def mb_overlap(states,c1,c2,O,Nk):
    tot=0j
    for i in range(len(states)):
        s=states[i]; p=1.0+0j
        for c in range(Nk):
            if (s>>c)&1: p*=O[c]
        tot+=np.conj(c1[i])*c2[i]*p
    return tot

def orbital_overlap(S1,S2,hom_sig=None):
    """O_c = <u_{k_c+t1}|u_{k_c+t2}> (cell-periodic parts). For homogeneous (magnetic-algebra) bands:
    <u_{k+t1}|u_{k+t2}> = F_{k+t2, t1-t2} -> phase exp(i sig l^2 ((t1-t2) x (k+t2))/2) (|.|->1)."""
    if hom_sig is None:
        return np.sum(np.conj(S1.v)*S2.v,axis=1)
    l2=S1.band.Auc/(2*np.pi); d=S1.kcart-S2.kcart; k2=S2.kcart
    return np.exp(1j*hom_sig*0.5*l2*(d[:,0]*k2[:,1]-d[:,1]*k2[:,0]))

def manifold_chern(band,cl,M=6,lam_fn=None,hom_sig=None,**kw):
    """returns C_total (sum over the three GS sectors), per-sector fluxes, min gap along the torus"""
    S0=make_system(band,cl,(0.,0.),lam_fn,**kw)
    low=[low_eigs(S0,K,2)[0] for K in range(cl.Nk)]
    order=np.argsort([l[0] for l in low]); Ks=[int(K) for K in order[:3]]
    grid={}; gaps=[]
    ths=np.linspace(0,2*np.pi,M+1)
    for i,t1 in enumerate(ths):
        for j,t2 in enumerate(ths):
            S=make_system(band,cl,twist_frac(cl,t1,t2),lam_fn,**kw)
            vecs={}; es=[]
            for K in Ks:
                e,v=low_eigs(S,K,2); vecs[K]=v[:,0]; es.append(e)
            # gap: lowest excited level of all sectors vs highest manifold level (cheap proxy: within GS sectors + others k=1)
            grid[(i,j)]=(S,vecs,es)
    flux={K:0.0 for K in Ks}
    for i in range(M):
        for j in range(M):
            for K in Ks:
                st=cl.sector(K)
                def ov(a,b):
                    Sa,va,_=grid[a]; Sb,vb,_=grid[b]
                    O=orbital_overlap(Sa,Sb,hom_sig)
                    return mb_overlap(st,va[K],vb[K],O.astype(np.complex128),cl.Nk)
                U=ov((i,j),(i+1,j))*ov((i+1,j),(i+1,j+1))*ov((i+1,j+1),(i,j+1))*ov((i,j+1),(i,j))
                flux[K]+=np.angle(U)
    Ctot=sum(flux.values())/(2*np.pi)
    return Ctot,{K:f/(2*np.pi) for K,f in flux.items()},Ks

# ------------------------------------------------------------------ particle entanglement spectrum
def admissible_count(Nk,NA,gap=2):
    """number of ring configurations of NA particles on Nk sites with >= gap empty sites between neighbours"""
    cnt=0
    for comb in itertools.combinations(range(Nk),NA):
        ok=True
        for a in range(NA):
            d=(comb[(a+1)%NA]-comb[a])%Nk if NA>1 else Nk
            if d<=gap: ok=False; break
        if ok: cnt+=1
    return cnt

def pes(cl,gs_list,NA):
    """gs_list: list of (sector K, vector). Returns xi (sorted) and momentum labels."""
    Nk=cl.Nk; Ne=cl.Ne; NB=Ne-NA
    Aconf=[sum(1<<c for c in comb) for comb in itertools.combinations(range(Nk),NA)]
    Bconf=[sum(1<<c for c in comb) for comb in itertools.combinations(range(Nk),NB)]
    Aidx={a:i for i,a in enumerate(Aconf)}; Bidx={b:i for i,b in enumerate(Bconf)}
    from scipy.sparse import coo_matrix
    rho=np.zeros((len(Aconf),len(Aconf)),complex)
    for K,psi in gs_list:
        st=cl.sector(K); rows=[];cols=[];vals=[]
        for n,s in enumerate(st):
            occ=[c for c in range(Nk) if (s>>c)&1]
            for comb in itertools.combinations(range(Ne),NA):
                A=[occ[x] for x in comb]; B=[occ[x] for x in range(Ne) if x not in comb]
                inv=sum(1 for a in A for b in B if b<a)
                ma=sum(1<<c for c in A); mb=sum(1<<c for c in B)
                rows.append(Aidx[ma]); cols.append(Bidx[mb]); vals.append((-1)**inv*psi[n])
        Mm=coo_matrix((vals,(rows,cols)),shape=(len(Aconf),len(Bconf))).tocsr()
        rho+=(Mm@Mm.getH()).toarray()
    rho/=np.trace(rho).real
    # momentum blocks
    KA=np.array([cl.grid[sum(cl.X1[c] for c in range(Nk) if (a>>c)&1)%cl.D, sum(cl.X2[c] for c in range(Nk) if (a>>c)&1)%cl.D] for a in Aconf])
    xs=[];ks=[]
    for K in np.unique(KA):
        idx=np.where(KA==K)[0]; ev=np.linalg.eigvalsh(rho[np.ix_(idx,idx)])
        ev=ev[ev>1e-15]; xs+=list(-np.log(ev)); ks+=[K]*len(ev)
    o=np.argsort(xs); return np.array(xs)[o],np.array(ks)[o]

def pes_counting(xi,count):
    """entanglement gap at the predicted counting and whether it is the largest gap among the low levels"""
    if len(xi)<=count: return dict(count=count,n=len(xi),gap=np.nan,largest=False)
    d=np.diff(xi[:min(len(xi),2*count+10)])
    g=xi[count]-xi[count-1]
    return dict(count=count,n=len(xi),gap=float(g),largest=bool(np.argmax(d)==count-1),xi_c=float(xi[count-1]))
