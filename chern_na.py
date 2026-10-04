"""Non-Abelian (determinant-link) many-body Chern number of the three-fold ground-state manifold on the
twist torus, with exact boundary sewing (Fukui-Hatsugai-Suzuki for a degenerate multiplet).

Grid theta_i = 2 pi i / M, i = 0..M-1 in each direction (periodic, M x M plaquettes).
Link U_mu(theta) = det M / |det M|,  M_ab = <Psi_a(theta)|Psi_b(theta + Delta_mu)>, a,b over the three
manifold states (lowest state of each of the three manifold momentum sectors).
Links that cross theta_mu = 2 pi use the large-gauge (relabelling) map R_mu: the system at theta + 2 pi e_mu
equals the system at theta with orbital c relabelled to pi(c) and its Bloch vector shifted by the reciprocal
vector s_c (k_c + t(2 pi e_mu) = k_pi(c) + s_c).  Many-body states are mapped with the fermionic reordering sign.
With this sewing the lattice field strength sums to an exact integer; we report it together with the largest
|plaquette flux| (branch safety) and the smallest singular value of any link overlap matrix."""
import numpy as np, numba as nb, sys, pickle, time, os
sys.path.insert(0,'/home/claude/sim')
from ed import *
from topo import twist_frac, low_eigs, make_system

@nb.njit(cache=True)
def mb_overlap_map(states_a, ca, states_b, cb, perm, Oc, Nk):
    """<Psi_a | R Psi_b> where R relabels orbital c' of Psi_b to perm_inv: orbital c' -> c with perm[c]=c'.
    Oc[c] = <u^a_c | shifted u^b_{perm[c]}>.  states_a sorted ascending (binary search)."""
    inv=np.empty(Nk,np.int64)
    for c in range(Nk): inv[perm[c]]=c
    tot=0j
    na=len(states_a)
    for j in range(len(states_b)):
        sb=states_b[j]
        # occupied labels of sb, mapped to new labels, in original (ascending c') order
        lab=np.empty(64,np.int64); n=0
        for cp in range(Nk):
            if (sb>>cp)&1:
                lab[n]=inv[cp]; n+=1
        # sign of sorting lab[0:n]
        sgn=1.0
        for x in range(n):
            for y in range(x+1,n):
                if lab[x]>lab[y]: sgn=-sgn
        s=0
        p=1.0+0j
        for x in range(n):
            s|=(1<<lab[x]); p*=Oc[lab[x]]
        lo=0; hi=na-1; idx=-1
        while lo<=hi:
            mid=(lo+hi)//2
            if states_a[mid]==s: idx=mid; break
            elif states_a[mid]<s: lo=mid+1
            else: hi=mid-1
        if idx>=0:
            tot+=np.conj(ca[idx])*cb[j]*sgn*p
    return tot

def relabel(cl,mu):
    """perm[c]=c', s[c] with k_c + t(2 pi e_mu) = k_{c'} + s_c (fractional b-basis)"""
    th=[0.,0.]; th[mu]=2*np.pi
    t=twist_frac(cl,*th)
    perm=np.empty(cl.Nk,np.int64); s=np.zeros((cl.Nk,2),np.int64)
    for c in range(cl.Nk):
        f=np.array([cl.X1[c]/cl.D,cl.X2[c]/cl.D])+t
        fl=np.floor(f+1e-9); r=f-fl
        x1=int(round(r[0]*cl.D))%cl.D; x2=int(round(r[1]*cl.D))%cl.D
        cp=cl.grid[x1,x2]; assert cp>=0
        perm[c]=cp; s[c]=fl.astype(int)
        assert np.allclose(np.array([cl.X1[cp]/cl.D,cl.X2[cp]/cl.D])+fl,f,atol=1e-9)
    return perm,s

def shift_vec(band,v,s):
    """coefficients of the Bloch function stored in v (base momentum k) re-expanded about k+s: v'(G)=v(G+s)"""
    n=band.nG; out=np.zeros_like(v)
    for i,(a,b) in enumerate(band.mn):
        j=band.index.get((int(a+s[0]),int(b+s[1])))
        if j is not None:
            out[...,i]=v[...,j]; out[...,n+i]=v[...,n+j]
    return out

def chern_na(band,cl,M=6,lam_fn=None,verbose=False):
    S0=make_system(band,cl,(0.,0.),lam_fn)
    low=[low_eigs(S0,K,2)[0] for K in range(cl.Nk)]
    Ks=[int(K) for K in np.argsort([l[0] for l in low])[:3]]
    ths=2*np.pi*np.arange(M)/M
    G={}; gapmin=np.inf
    for i in range(M):
        for j in range(M):
            S=make_system(band,cl,twist_frac(cl,ths[i],ths[j]),lam_fn)
            vec={}; e3=[]
            for K in Ks:
                e,v=low_eigs(S,K,2); vec[K]=v[:,0]; e3.append(e)
            G[(i,j)]=(S.v.copy(),vec)
    R=[relabel(cl,0),relabel(cl,1)]
    sec={K:cl.sector(K) for K in range(cl.Nk)}
    def link(p,mu):
        i,j=p; q=[i,j]; q[mu]+=1; wrap=(q[mu]==M); q[mu]%=M; q=tuple(q)
        va,ca=G[p]; vb,cb=G[q]
        if not wrap:
            perm=np.arange(cl.Nk,dtype=np.int64); vbs=vb
        else:
            perm,s=R[mu]
            vbs=np.array([shift_vec(band,vb[perm[c]],s[c]) for c in range(cl.Nk)])
        Oc=np.sum(np.conj(va)*vbs,axis=1).astype(np.complex128)
        Mab=np.zeros((3,3),complex)
        for a,Ka in enumerate(Ks):
            for b,Kb in enumerate(Ks):
                Mab[a,b]=mb_overlap_map(sec[Ka],ca[Ka],sec[Kb],cb[Kb],perm,Oc,cl.Nk)
        sv=np.linalg.svd(Mab,compute_uv=False)
        d=np.linalg.det(Mab)
        return d/abs(d),sv.min()
    U={}; smin=np.inf
    for i in range(M):
        for j in range(M):
            for mu in (0,1):
                u,s_=link((i,j),mu); U[(i,j,mu)]=u; smin=min(smin,s_)
    F=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            ip=(i+1)%M; jp=(j+1)%M
            P=U[(i,j,0)]*U[(ip,j,1)]*np.conj(U[(i,jp,0)])*np.conj(U[(i,j,1)])
            F[i,j]=np.angle(P)
    C=F.sum()/(2*np.pi)
    return dict(C=C,Fmax=float(np.abs(F).max()),smin=float(smin),Ks=Ks,M=M,F=F)

if __name__=='__main__':
    from joblib import Parallel,delayed
    def job(name,M,Ne=5):
        fn=f'/home/claude/sim/res4/chernNA_{name}_{Ne}_{M}.pkl'
        if os.path.exists(fn): return fn
        from ana import Params,Band,get_cluster
        spec=pickle.load(open(f'/home/claude/sim/res2/{name}.pkl','rb'))['spec']
        b=Band(Params(**{**dict(Gcut=3.1,smap='exp'),**spec.get('p',{})})); cl=get_cluster('A',Ne)
        t=time.time(); r=chern_na(b,cl,M); r['time']=time.time()-t
        pickle.dump(r,open(fn,'wb')); print(name,M,r['C'],r['Fmax'],r['smin'],round(r['time']),flush=True); return fn
    which=sys.argv[1]
    if which=='test': print(job('A',4))
    elif which=='conv': Parallel(n_jobs=2)(delayed(job)(n,M) for n in ('A','P3') for M in (5,8,10))
    else:
        import glob
        names=sorted(os.path.basename(f)[:-4] for f in glob.glob('/home/claude/sim/res2/*.pkl') if not f.endswith('LL.pkl'))
        Parallel(n_jobs=2)(delayed(job)(n,6) for n in names)
