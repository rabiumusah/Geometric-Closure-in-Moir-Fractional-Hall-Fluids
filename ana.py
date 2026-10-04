import numpy as np, time
from scipy.sparse.linalg import eigsh
from ed import *

FAM = {'C':{4:[[4,2],[2,4]],7:[[5,1],[4,5]],9:[[6,3],[3,6]],12:[[8,4],[4,8]]},
       'A':{4:[[6,0],[2,2]],5:[[5,0],[1,3]],6:[[6,0],[1,3]],7:[[21,0],[4,1]],8:[[6,0],[1,4]],9:[[9,0],[3,3]]},
       'B':{4:[[3,0],[0,4]],5:[[3,0],[0,5]],6:[[3,0],[0,6]],7:[[3,0],[0,7]],8:[[3,0],[0,8]]}}
_CL={}
def get_cluster(fam,Ne):
    key=(fam,Ne)
    if key not in _CL: _CL[key]=Cluster(FAM[fam][Ne],Ne)
    return _CL[key]

def sector_eigs(S,K,nev,full_max=1200):
    H=S.H(K); dim=H.shape[0]
    if dim<=full_max:
        e,v=np.linalg.eigh(H.toarray()); return e,v,H,True
    nv=min(nev,dim-2)
    e,v=eigsh(H,k=nv,which='SA',ncv=min(dim-1,max(3*nv,40)),tol=1e-10)
    o=np.argsort(e); return e[o],v[:,o],H,False

def channel_coefs(S,u,ell,radial='F',spins=(0,2,4)):
    """Lam_a[c] for channels [rho, 2+,2-, 4+,4-]; returns array (5,Nk)"""
    sel=np.where((S.Qs[:,0]==u)&(S.Qs[:,4]>1e-9))[0] if True else None
    # u==0: exclude Q=0 (Qs[:,4]>0)
    Qn=S.Qs[sel,4]; ph=np.arctan2(S.Qs[sel,6],S.Qs[sel,5]); Lm=S.Lam[sel]   # (nsel,Nk)
    if radial=='F': wF=S.meanF[sel]
    elif radial=='gauss': wF=np.exp(-0.5*np.einsum('qi,ij,qj->q',S.Qs[sel,5:7],S.gbar_ref,S.Qs[sel,5:7]))
    else: wF=np.ones_like(Qn)
    x=Qn*ell
    out=[]
    for s,sgn in ((0,0),(2,1),(2,-1),(4,1),(4,-1)):
        coef=(x**s)*wF*np.exp(1j*sgn*s*ph)
        out.append(coef@Lm)
    return np.array(out)

def orthonormalize(G,mode='hier',tol=1e-9):
    """G[a,b]=<v_a|v_b>, channels [0],[1,2],[3,4]. returns C (nout x 5), block index list"""
    n=G.shape[0]; blocks=[[0],[1,2],[3,4]]
    if mode=='lowdin':
        ev,U=np.linalg.eigh(G); keep=ev>tol*ev.max()
        X=(U[:,keep]/np.sqrt(ev[keep]))@U[:,keep].conj().T      # G^{-1/2} (v-basis coefficient)
        # rows: u_alpha = sum_b C[alpha,b] v_b with C = X^T-ish; use conj for orientation
        C=X.T   # u_a = sum_b X[b,a] v_b
        # block membership remains channel label
        return C,[0,1,1,2,2][:C.shape[0]] if C.shape[0]==5 else None
    rows=[];lab=[]
    for bi,blk in enumerate(blocks):
        Cr=[]
        for b in blk:
            e=np.zeros(n,complex); e[b]=1
            for Ci in rows:
                e=e-Ci*(np.conj(Ci)@G[:,b])
            Cr.append(e)
        Cr=np.array(Cr); R=Cr.conj()@G@Cr.T   # R[i,j]=<r_i|r_j>
        # <r_i|r_j> = sum conj(Cr_i,a) G_ab Cr_j,b
        ev,U=np.linalg.eigh((R+R.conj().T)/2); keep=ev>max(1e-8*ev.max(),1e-14*max(1.0,np.abs(np.diag(G)).max()))  # relative (block) rank threshold
        if keep.sum()==0: continue
        X=(U[:,keep]/np.sqrt(ev[keep]))@U[:,keep].conj().T
        Cn=X.T@Cr
        for r in Cn: rows.append(r); lab.append(bi)
    return np.array(rows),lab

def orthonormal_vectors(Vm,mode='hier',rtol=1e-6):
    """numerically stable orthonormal multipole states from raw vectors Vm (dim,5).
    hier: blocks [rho],[2+,2-],[4+,4-] processed in order; each block is projected (twice) on the complement of
    the accepted states and Lowdin(polar)-normalized; directions with singular value < rtol * (original block norm)
    are discarded (the block is then rank deficient, e.g. at high-symmetry momenta where +/- helicities coincide).
    lowdin: polar decomposition of the whole set. Returns U (dim,k), labels (k,)"""
    blocks=[[0],[1,2],[3,4]]
    if mode=='lowdin':
        X,sv,Wh=np.linalg.svd(Vm,full_matrices=False); keep=sv>rtol*sv.max()
        if keep.all(): U=X@Wh
        else: U=X[:,keep]
        lab=[0,1,1,2,2] if keep.all() else [-1]*int(keep.sum())
        return U,np.array(lab)
    cols=[];lab=[]
    for bi,blk in enumerate(blocks):
        R=Vm[:,blk].copy(); nrm=np.linalg.norm(Vm[:,blk],axis=0).max()
        if nrm==0: continue
        for _ in range(2):
            if cols:
                Q=np.array(cols).T; R=R-Q@(Q.conj().T@R)
        X,sv,Wh=np.linalg.svd(R,full_matrices=False); keep=sv>rtol*nrm
        if keep.sum()==0: continue
        U=X@Wh if keep.all() else X[:,keep]
        for j in range(U.shape[1]): cols.append(U[:,j]); lab.append(bi)
    return np.array(cols).T,np.array(lab)

def analyze_u(S,cl,K_gs,gs,E0,manifold,eigs,u,ell,mode='hier',radial='F',full_check=False,nstate=24):
    """spectral weights of channels from GS to sector K_gs+u"""
    Kt=cl.add(K_gs,u)
    Lam=channel_coefs(S,u,ell,radial)
    vecs=[]
    for a in range(5):
        O,_=S.rho_op(K_gs,Lam[a],u); vecs.append(O@gs)
    Vm=np.array(vecs).T                            # (dim_t,5)
    e,V,H,full=eigs[Kt]
    excl=[i for (K,i) in manifold if K==Kt]
    keepn=[i for i in range(len(e)) if i not in excl]
    for i in excl:
        Vm=Vm-np.outer(V[:,i],V[:,i].conj()@Vm)
    if np.linalg.norm(Vm,axis=0).max()<1e-7: return None
    U,lab=orthonormal_vectors(Vm,mode)
    A=V[:,keepn].conj().T@U                          # (n, nout): <n|u_alpha>
    En=e[keepn]-E0
    nb_=np.array([max(1,int(np.sum(lab==b))) for b in range(3)])
    W=np.array([np.sum(np.abs(A[:,lab==b])**2,axis=1)/nb_[b] if np.any(lab==b) else np.zeros(len(En)) for b in range(3)])
    out=dict(u=u,Kt=Kt,E=En[:nstate],W=W[:,:nstate],captured=np.array([np.sum(W[b]) for b in range(3)]),full=full)
    Th=U.conj().T@(H@U)-E0*np.eye(U.shape[1])      # operator-space Hamiltonian (first-moment matrix), exact
    out['Theta']=(Th+Th.conj().T)/2; out['lab']=lab; out['Ec_max']=float(En[-1]); out['rank']=int(U.shape[1])
    out['G']=Vm.conj().T@Vm
    chi=2*np.real((A.conj().T*(1/np.maximum(En,1e-9))[None,:])@A)
    ev=np.linalg.eigvalsh((chi+chi.T)/2); out['kappa']=1.0/ev.max() if ev.max()>0 else np.nan
    out['chi_b']=np.array([2*np.sum(W[b]/np.maximum(En,1e-9)) for b in range(3)])
    if full_check and full:
        Z=A.conj().T@A
        out['r0']=float(np.abs(Z-np.eye(Z.shape[0])).max())
        first=(A.conj().T*En[None,:])@A
        out['r1']=float(np.abs(first-out['Theta']).max()/max(np.abs(out['Theta']).max(),1e-30))
    return out

def run_point(band,fam,Ne,twist=(0.,0.),disp=False,d=300.,epsr=10.,lam=1.0,gbar=None,ell=None,nev=24,
              mode='hier',radial='F',full_u=(),nstate=24,gs_member=0,variants=(),lam_fn=None):
    cl=get_cluster(fam,Ne)
    S=System(band,cl,twist=twist,disp=disp,d=d,epsr=epsr,lam=lam,gbar=gbar,lam_fn=lam_fn); S.gbar_ref=gbar
    eigs={}
    t0=time.time()
    for K in range(cl.Nk):
        eigs[K]=sector_eigs(S,K,nev)
    low=np.array([eigs[K][0][0] for K in range(cl.Nk)]); order=np.argsort(low)
    manifold=[(int(K),0) for K in order[:3]]
    K_gs,_=manifold[gs_member]; E0=low[K_gs]; gs=eigs[K_gs][1][:,0]
    spread=low[order[2]]-low[order[0]]
    cand=[]
    for K in range(cl.Nk):
        e=eigs[K][0]
        for i in range(len(e)):
            if (K,i) not in manifold: cand.append(e[i]); break
    gap=min(cand)-low[order[2]]
    for u in full_u:   # dense re-diagonalisation of selected target sectors (sum-rule certification)
        Kt=cl.add(K_gs,u)
        if not eigs[Kt][3]: eigs[Kt]=sector_eigs(S,Kt,nev,full_max=10**9)
    res=dict(Ne=Ne,fam=fam,D=cl.D,K_gs=K_gs,E0=E0,manifold=manifold,spread=spread,gap=gap,
             manifold_X=[(int(cl.X1[K]),int(cl.X2[K])) for K,_ in manifold],dims=[cl.dim(K) for K in range(cl.Nk)],
             t_eig=time.time()-t0,lowest_per_sector=low.tolist())
    def do(mode_,radial_):
        out={}
        for u in range(cl.Nk):
            r=analyze_u(S,cl,K_gs,gs,E0,manifold,eigs,u,ell,mode_,radial_,full_check=(u in full_u),nstate=nstate)
            if r is not None:
                qs=S.Qs[(S.Qs[:,0]==u)&(S.Qs[:,4]>1e-9)]
                r['q']=float(qs[:,4].min()); out[u]=r
        return out
    res['u']=do(mode,radial)
    res['var']={v:do(*v) for v in variants}
    res['time']=time.time()-t0
    return res
