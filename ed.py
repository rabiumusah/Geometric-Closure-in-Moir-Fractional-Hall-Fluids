import numpy as np, numba as nb, time
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from band import *

KE2 = 14399.64   # e^2/(4 pi eps0) in meV*A

@nb.njit(cache=True)
def popc(x):
    c=0
    while x:
        x&=x-1; c+=1
    return c
@nb.njit(cache=True)
def sgn_below(s,k):
    return -1.0 if (popc(s&((1<<k)-1))&1) else 1.0

@nb.njit(cache=True)
def all_states(Nk,Ne,X1,X2,D,grid):
    n=1
    for i in range(Ne): n=n*(Nk-i)//(i+1)
    st=np.empty(n,np.int64); sec=np.empty(n,np.int64)
    s=(1<<Ne)-1
    for i in range(n):
        st[i]=s; a=0; b=0
        for k in range(Nk):
            if (s>>k)&1:
                a+=X1[k]; b+=X2[k]
        sec[i]=grid[a%D,b%D]
        c=s&-s; r=s+c
        s=(((r^s)>>2)//c)|r
    return st,sec

@nb.njit(cache=True)
def build_H(states, Ms, X1, X2, D, grid, Nk, Ne, eps, maxnnz):
    dim=len(states)
    rows=np.empty(maxnnz,np.int64); cols=np.empty(maxnnz,np.int64); vals=np.empty(maxnnz,np.complex128)
    nnz=0; occ=np.empty(Ne,np.int64)
    for i in range(dim):
        s=states[i]; m=0
        for k in range(Nk):
            if (s>>k)&1:
                occ[m]=k; m+=1
        e=0.0
        for t in range(Ne): e+=eps[occ[t]]
        rows[nnz]=i; cols[nnz]=i; vals[nnz]=e; nnz+=1
        for x in range(Ne):
            c=occ[x]; s1=s^(1<<c); sg1=sgn_below(s,c)
            for y in range(x+1,Ne):
                d=occ[y]; sg2=sgn_below(s1,d); s2=s1^(1<<d)
                P1=X1[c]+X1[d]; P2=X2[c]+X2[d]
                for a in range(Nk):
                    b=grid[(P1-X1[a])%D,(P2-X2[a])%D]
                    if a<b:
                        if (s2>>a)&1 or (s2>>b)&1: continue
                        sg3=sgn_below(s2,b); s3=s2|(1<<b)
                        sg4=sgn_below(s3,a); s4=s3|(1<<a)
                        v=0.5*Ms[a,b,c,d]*sg1*sg2*sg3*sg4
                        # locate s4 in states
                        lo=0; hi=dim-1; idx=-1
                        while lo<=hi:
                            mid=(lo+hi)//2
                            if states[mid]==s4: idx=mid; break
                            elif states[mid]<s4: lo=mid+1
                            else: hi=mid-1
                        if idx>=0:
                            rows[nnz]=idx; cols[nnz]=i; vals[nnz]=v; nnz+=1
    return rows,cols,vals,nnz

@nb.njit(cache=True)
def build_rho(states_src, states_dst, Lam, tgt, Nk, Ne):
    """operator sum_c Lam[c] a+_{tgt[c]} a_c : sparse (dst x src)"""
    dim=len(states_src); nd=len(states_dst)
    maxnnz=dim*Ne
    rows=np.empty(maxnnz,np.int64); cols=np.empty(maxnnz,np.int64); vals=np.empty(maxnnz,np.complex128)
    nnz=0
    for i in range(dim):
        s=states_src[i]
        for c in range(Nk):
            if not (s>>c)&1: continue
            a=tgt[c]
            sg1=sgn_below(s,c); s1=s^(1<<c)
            if (s1>>a)&1: continue
            sg2=sgn_below(s1,a); s2=s1|(1<<a)
            lo=0; hi=nd-1; idx=-1
            while lo<=hi:
                mid=(lo+hi)//2
                if states_dst[mid]==s2: idx=mid; break
                elif states_dst[mid]<s2: lo=mid+1
                else: hi=mid-1
            if idx>=0:
                rows[nnz]=idx; cols[nnz]=i; vals[nnz]=Lam[c]*sg1*sg2; nnz+=1
    return rows,cols,vals,nnz

class Cluster:
    """generic torus: superlattice rows L1=(S[0]), L2=(S[1]) in units of (a1,a2); momenta f=S^-1 n mod 1"""
    def __init__(self,S,Ne):
        S=np.array(S,int); D=abs(int(round(np.linalg.det(S)))); self.S=S; self.D=D; self.Nk=D; self.Ne=Ne
        adj=np.array([[S[1,1],-S[0,1]],[-S[1,0],S[0,0]]])*(1 if round(np.linalg.det(S))>0 else -1)
        pts=set()
        for n1 in range(-3*D,3*D+1):
            for n2 in range(-3*D,3*D+1):
                x=adj.T@np.array([n1,n2]) if False else (adj@np.array([n1,n2]))
                pts.add((int(x[0]%D),int(x[1]%D)))
                if len(pts)==D: break
            if len(pts)==D: break
        # f = S^-1 n  => numerators over D: S^-1 = adj/det ; f_i = sum_j (S^-1)_{ij} n_j
        pts=sorted(pts)
        assert len(pts)==D, (len(pts),D)
        self.X1=np.array([p[0] for p in pts],np.int64); self.X2=np.array([p[1] for p in pts],np.int64)
        self.grid=-np.ones((D,D),np.int64)
        for i,p in enumerate(pts): self.grid[p]=i
        st,sec=all_states(self.Nk,Ne,self.X1,self.X2,D,self.grid)
        order=np.lexsort((st,sec)); self.states=st[order]; sec=sec[order]
        self.sec_start=np.searchsorted(sec,np.arange(self.Nk)); self.sec_end=np.searchsorted(sec,np.arange(self.Nk),side='right')
    def sector(self,K): return self.states[self.sec_start[K]:self.sec_end[K]]
    def dim(self,K): return self.sec_end[K]-self.sec_start[K]
    def add(self,K,u):  # sector K + mesh vector u (orbital index)
        return self.grid[(self.X1[K]+self.X1[u])%self.D,(self.X2[K]+self.X2[u])%self.D]
    def shape_info(self):
        L=self.S@np.array([[1,0],[0.5,np.sqrt(3)/2]]); return np.linalg.norm(L,axis=1)

def best_clusters(D,top=4):
    """HNF sublattices of index D ranked by compactness (Lagrange-reduced shortest vector, then squareness)"""
    A=np.array([[1,0],[0.5,np.sqrt(3)/2]]); out=[]
    for a in range(1,D+1):
        if D%a: continue
        d=D//a
        for c in range(a):
            S=np.array([[a,0],[c,d]]); L=S@A
            u,v=L[0].copy(),L[1].copy()
            for _ in range(50):
                if np.linalg.norm(u)>np.linalg.norm(v): u,v=v,u
                m=np.round(u@v/(u@u)); 
                if m==0: break
                v=v-m*u
            if np.linalg.norm(u)>np.linalg.norm(v): u,v=v,u
            ang=np.degrees(np.arccos(abs(u@v)/np.linalg.norm(u)/np.linalg.norm(v)))
            out.append((np.linalg.norm(u),np.linalg.norm(v)/np.linalg.norm(u),ang,S.tolist()))
    out.sort(key=lambda t:(-round(t[0],3),t[1]))
    return out[:top]

class System:
    """single-band projected system on an N1 x N2 mesh"""
    def __init__(self, band, cl, twist=(0.,0.), disp=False, d=300., epsr=10., lam=1.0, gbar=None,
                 Qcut=2.3, gauge_rng=None, lam_fn=None, v_ref=None, Qs_ref=None):
        self.band=band; self.cl=cl; D=cl.D; Nk=cl.Nk
        fr=np.stack([cl.X1/D+twist[0],cl.X2/D+twist[1]],1)
        self.kcart=band.frac2cart(fr)
        e,v=band.top(self.kcart,1); e=e[:,0]; v=v[:,:,0]
        if gauge_rng is not None: v=v*np.exp(2j*np.pi*gauge_rng.random(Nk))[:,None]
        if v_ref is not None:   # parallel-transport gauge: <v_ref_c|v_c> real positive
            ov=np.sum(np.conj(v_ref)*v,axis=1); v=v*np.exp(-1j*np.angle(ov))[:,None]
        self.v=v
        self.eps=(-(e-e.mean())) if disp else np.zeros(Nk)
        self.Ebands=e
        # shifts needed
        R=int(np.ceil(Qcut/ band.g0*1.0))+3
        m_off=np.arange(-R-1,R+1)
        # enumerate Q = (u1/N1+m1, u2/N2+m2) b-basis ; collect (u1,u2,m1,m2,|Q|)
        Qs=[]
        for u in range(Nk):
            for m1 in m_off:
                for m2 in m_off:
                    Qc=(cl.X1[u]/D+m1)*band.b1+(cl.X2[u]/D+m2)*band.b2
                    nq=np.linalg.norm(Qc)
                    if nq<=Qcut*band.g0: Qs.append((u,0,m1,m2,nq,Qc[0],Qc[1]))
        self.Qs=np.array(Qs)
        if Qs_ref is not None:   # keep the same Q label set (u,m1,m2) as a reference system; recompute cartesian parts
            Qs=[]
            for (u,_,m1,m2,_,_,_) in Qs_ref:
                u=int(u); Qc=(cl.X1[u]/D+m1)*band.b1+(cl.X2[u]/D+m2)*band.b2
                Qs.append((u,0,m1,m2,np.linalg.norm(Qc),Qc[0],Qc[1]))
            self.Qs=np.array(Qs)
        # form factors  Lam[Qidx, c] = <psi_{k_c+Q}|psi_c>
        n2=2*band.nG
        smax=int(np.ceil(Qcut*1.5))+3
        shift_cache={}
        def Lam_s(s):
            if s in shift_cache: return shift_cache[s]
            src=[];dst=[]
            for i,(a,b) in enumerate(band.mn):
                j=band.index.get((a+s[0],b+s[1]))
                if j is not None: src.append(j); dst.append(i)
            src=np.array(src,int);dst=np.array(dst,int)
            fs=np.concatenate([src,src+band.nG]); fd=np.concatenate([dst,dst+band.nG])
            # Lam[a,c]=sum_G conj(v_a[G+s]) v_c[G]
            M=np.conj(v[:,fs])@v[:,fd].T
            shift_cache[s]=M; return M
        nQ=len(self.Qs); self.Lam=np.zeros((nQ,Nk),complex); self.tgt=np.zeros((nQ,Nk),np.int64); self.svec=np.zeros((nQ,Nk,2),np.int64)
        for iq,(u,_,m1,m2,_,_,_) in enumerate(self.Qs):
            u=int(u);m1=int(m1);m2=int(m2)
            for c in range(Nk):
                x1=cl.X1[c]+cl.X1[u]; x2=cl.X2[c]+cl.X2[u]
                a=cl.grid[x1%D,x2%D]; self.tgt[iq,c]=a
                s=(int(x1//D)+m1,int(x2//D)+m2); self.svec[iq,c]=s
                self.Lam[iq,c]=Lam_s(s)[a,c]
        self.meanF_band=np.mean(np.abs(self.Lam),axis=1)
        if lam_fn is not None: self.Lam=lam_fn(self)
        # lambda-path: rescale non-Gaussian remainder of cluster-averaged |F|
        self.lam=lam
        Qn=self.Qs[:,4]; Qv=self.Qs[:,5:7]
        self.meanF=np.mean(np.abs(self.Lam),axis=1)
        if lam!=1.0:
            assert gbar is not None
            R_=np.log(self.meanF)+0.5*np.einsum('qi,ij,qj->q',Qv,gbar,Qv)
            f=np.exp((lam-1.0)*R_); self.Lam=self.Lam*f[:,None]; self.meanF=self.meanF*f
        # interaction tensor
        A=Nk*band.Auc
        M=np.zeros((Nk,Nk,Nk,Nk),complex)
        cgrid,dgrid=np.meshgrid(np.arange(Nk),np.arange(Nk),indexing='ij')
        # index lookup (u1,u2,m1,m2)->iq
        look={(int(q[0]),int(q[2]),int(q[3])):i for i,q in enumerate(self.Qs)}
        for iq,(u,_,m1,m2,nq,_,_) in enumerate(self.Qs):
            if nq<1e-12: continue
            u=int(u);m1=int(m1);m2=int(m2)
            # -Q : mesh part -u mod, integer part chosen so that (-u/D - m)=(mu/D + mm)
            xm1=(-cl.X1[u]); xm2=(-cl.X2[u])
            mu=cl.grid[xm1%D,xm2%D]
            mm1=-m1+int(xm1//D); mm2=-m2+int(xm2//D)
            iqm=look.get((int(mu),mm1,mm2))
            if iqm is None: continue
            V=2*np.pi*KE2*np.tanh(nq*d)/(epsr*nq)/A
            a=self.tgt[iq]; b=self.tgt[iqm]
            M[a[:,None],b[None,:],cgrid,dgrid]+=V*np.outer(self.Lam[iq],self.Lam[iqm])
        self.M=M
        Ms=M-M.transpose(1,0,2,3)-M.transpose(0,1,3,2)+M.transpose(1,0,3,2)
        self.Ms=np.ascontiguousarray(Ms)
        self.look=look

    def H(self,K):
        cl=self.cl; st=cl.sector(K); dim=len(st)
        maxnnz=dim*(cl.Ne*(cl.Ne-1)//2*(cl.Nk//2+1)+cl.Ne+2)
        r,c,v,n=build_H(st,self.Ms,cl.X1,cl.X2,cl.D,cl.grid,cl.Nk,cl.Ne,self.eps,maxnnz)
        return csr_matrix((v[:n],(r[:n],c[:n])),shape=(dim,dim))

    def rho_op(self,K,Lam_c,u):
        """sum_c Lam_c a+_{c+u} a_c : sector K -> sector K+u"""
        cl=self.cl; Kd=cl.add(K,u)
        tgt=cl.grid[(cl.X1+cl.X1[u])%cl.D,(cl.X2+cl.X2[u])%cl.D]
        src=cl.sector(K); dst=cl.sector(Kd)
        r,c,v,n=build_rho(src,dst,Lam_c.astype(np.complex128),tgt.astype(np.int64),cl.Nk,cl.Ne)
        return csr_matrix((v[:n],(r[:n],c[:n])),shape=(len(dst),len(src))),Kd
