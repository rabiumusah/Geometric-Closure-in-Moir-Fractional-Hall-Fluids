"""Strained twisted-MoTe2 continuum model (K valley), plane-wave basis.
Reciprocal-vector-only strain model: g_j -> (1-E) g_j with E = eps*(cos2phi sz + sin2phi sx)."""
import numpy as np
HBAR2_2ME = 3809.98   # hbar^2/(2 m_e) in meV A^2

class Params:
    def __init__(self, theta_deg=3.89, V=20.8, psi_deg=107.7, w=-23.8, mstar=0.62, a0=3.52,
                 eps=0.0, phi_deg=0.0, Gcut=4.1, swap=True, Eextra=None, smap='lin'):
        self.theta=theta_deg; self.V=V; self.psi=np.deg2rad(psi_deg); self.w=w; self.m=mstar; self.a0=a0
        self.eps=eps; self.phi=np.deg2rad(phi_deg); self.Gcut=Gcut; self.swap=swap
        self.Eextra=Eextra; self.smap=smap   # extra strain tensor; 'lin': b->(1-E)b, 'exp': b->exp(-E)b (area preserving)
    def copy(self, **kw):
        d=dict(theta_deg=self.theta,V=self.V,psi_deg=np.rad2deg(self.psi),w=self.w,mstar=self.m,a0=self.a0,
               eps=self.eps,phi_deg=np.rad2deg(self.phi),Gcut=self.Gcut,swap=self.swap,Eextra=self.Eextra,smap=self.smap); d.update(kw)
        return Params(**d)

class Band:
    def __init__(self, p: Params):
        self.p=p
        aM = p.a0/(2*np.sin(np.deg2rad(p.theta)/2)); self.aM=aM
        g0 = 4*np.pi/(np.sqrt(3)*aM); self.g0=g0
        E = p.eps*np.array([[np.cos(2*p.phi),np.sin(2*p.phi)],[np.sin(2*p.phi),-np.cos(2*p.phi)]])
        if p.Eextra is not None: E = E + np.asarray(p.Eextra,float)
        if p.smap=='exp':
            w_,U_=np.linalg.eigh(E); S=U_@np.diag(np.exp(-w_))@U_.T
        else:
            S = np.eye(2)-E
        b1 = S@np.array([g0,0.0]); b2 = S@np.array([g0/2,g0*np.sqrt(3)/2])
        self.b1,self.b2=b1,b2
        self.Auc = (2*np.pi)**2/abs(b1[0]*b2[1]-b1[1]*b2[0])
        # plane waves
        R=int(np.ceil(p.Gcut*1.3))+1
        mm,nn=np.meshgrid(np.arange(-R,R+1),np.arange(-R,R+1),indexing='ij')
        mm,nn=mm.ravel(),nn.ravel()
        Gu = mm[:,None]*np.array([g0,0.0])+nn[:,None]*np.array([g0/2,g0*np.sqrt(3)/2])
        keep = np.linalg.norm(Gu,axis=1)<=p.Gcut*g0*1.0001
        self.mn = np.stack([mm[keep],nn[keep]],1); self.nG=len(self.mn)
        self.G = self.mn[:,0:1]*b1+self.mn[:,1:2]*b2
        self.index = {(int(a),int(b)):i for i,(a,b) in enumerate(self.mn)}
        kp = (b1+b2)/3; km=(2*b1-b2)/3
        if p.swap: kp,km=km,kp
        self.kp,self.km=kp,km
        self._const()
    def _const(self):
        p=self.p; n=self.nG
        Hc=np.zeros((2*n,2*n),complex)
        # V: j=1,3,5 -> shifts b1, b2-b1, -b2 ; <G'|Delta_l|G> = V e^{i l psi} if G'-G=+g_j, V e^{-i l psi} if -g_j
        shifts=[(1,0),(-1,1),(0,-1)]
        for l,layer in ((+1,0),(-1,1)):
            for (sa,sb) in shifts:
                for i,(a,b) in enumerate(self.mn):
                    j=self.index.get((a+sa,b+sb))
                    if j is not None:
                        Hc[layer*n+j,layer*n+i]+=p.V*np.exp(1j*l*p.psi)
                    j=self.index.get((a-sa,b-sb))
                    if j is not None:
                        Hc[layer*n+j,layer*n+i]+=p.V*np.exp(-1j*l*p.psi)
        # tunnelling top<-bottom: G'-G in {0,-g2,-g3}; g2=b2, g3=b2-b1
        for (sa,sb) in [(0,0),(0,-1),(1,-1)]:
            for i,(a,b) in enumerate(self.mn):
                j=self.index.get((a+sa,b+sb))
                if j is not None:
                    Hc[j,n+i]+=p.w; Hc[n+i,j]+=np.conj(p.w)
        self.Hc=Hc
    def H(self,k):
        k=np.atleast_2d(k); c=HBAR2_2ME/self.p.m
        H=np.broadcast_to(self.Hc,(len(k),)+self.Hc.shape).copy()
        for layer,kap in ((0,self.kp),(1,self.km)):
            d=k[:,None,:]+self.G[None,:,:]-kap[None,None,:]
            kin=-c*np.sum(d*d,axis=2)
            idx=np.arange(self.nG)+layer*self.nG
            H[:,idx,idx]+=kin
        return H
    def top(self,k,nb=1):
        """energies and eigenvectors of the top nb bands (descending)"""
        e,v=np.linalg.eigh(self.H(k))
        return e[:,::-1][:,:nb], v[:,:,::-1][:,:,:nb]
    def frac2cart(self,f):
        f=np.atleast_2d(f); return f[:,0:1]*self.b1+f[:,1:2]*self.b2

def shifted_overlap(v1, v2, band, s):
    """<psi_a(+s)|psi_c> where psi_a(+s)[G]=psi_a[G+s]; v arrays (...,2*nG)."""
    n=band.nG
    src=[];dst=[]
    for i,(a,b) in enumerate(band.mn):
        j=band.index.get((a+s[0],b+s[1]))
        if j is not None: src.append(j); dst.append(i)
    src=np.array(src);dst=np.array(dst)
    full_s=np.concatenate([src,src+n]); full_d=np.concatenate([dst,dst+n])
    return np.sum(np.conj(v1[...,full_s])*v2[...,full_d],axis=-1)

def chern(band, N=18):
    f=(np.arange(N)/N)
    F1,F2=np.meshgrid(f,f,indexing='ij'); fr=np.stack([F1.ravel(),F2.ravel()],1)
    _,v=band.top(band.frac2cart(fr)); v=v[:,:,0].reshape(N,N,-1)
    def ov(a,b): return np.sum(np.conj(a)*b,axis=-1)
    # periodic gauge links: wrap uses shifted index
    def link(i,j,di,dj):
        i2,j2=i+di,j+dj; s1=s2=0
        if i2==N: i2=0; s1=1
        if j2==N: j2=0; s2=1
        return v[i,j],v[i2,j2],(s1,s2)
    tot=0; 
    for i in range(N):
        for j in range(N):
            a,b,s=link(i,j,1,0); U1=shifted_overlap(b,a,band,s) if s!=(0,0) else ov(b,a)
            a2=v[(i+1)%N,j]; s_=(1 if i+1==N else 0,0)
            b2=v[(i+1)%N,(j+1)%N]; s_b=(1 if i+1==N else 0,1 if j+1==N else 0)
            c2=v[i,(j+1)%N]; s_c=(0,1 if j+1==N else 0)
            def L(x,y,sx,sy):  # <x|y> with x at k+d, y at k, relative shift sx-sy
                return shifted_overlap(x,y,band,(sx[0]-sy[0],sx[1]-sy[1]))
            U1=L(a2,v[i,j],s_,(0,0)); U2=L(b2,a2,s_b,s_); U3=L(c2,b2,s_c,s_b); U4=L(v[i,j],c2,(0,0),s_c)
            tot+=np.angle(U1*U2*U3*U4)
    return tot/(2*np.pi)
