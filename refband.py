"""Homogeneous ('Landau-level-like') reference bands on the moire lattice.
Form factors obey the magnetic-translation algebra with one flux quantum per moire cell:
  lambda_c(Q) = f(|Q|) exp(i sig l^2 (Q x k_c)/2) exp(i sig xi_s(k_a)),  k_c+Q = k_a + s,
  xi_s(k) = l^2 (s x k)/2 + pi s1 s2 ,  l^2 = A_uc/(2 pi).
Such a band has |F_{k,Q}| independent of k and uniform Berry curvature: all of its information is in f(Q)."""
import numpy as np

def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def hom_lam(f_kind='twin',sig=+1,f_custom=None):
    def fn(S):
        band=S.band; cl=S.cl; l2=band.Auc/(2*np.pi)
        Q=S.Qs[:,5:7]; Qn=S.Qs[:,4]
        if f_kind=='twin': f=S.meanF_band.copy()                    # cluster-averaged |F| of the parent band
        elif f_kind=='rms': f=np.sqrt(np.mean(np.abs(S.Lam)**2,axis=1))   # rms |F| of the parent band (smooth in parameters)
        elif f_kind=='LLL': f=np.exp(-l2*Qn**2/4)
        elif f_kind=='1LL': f=(1-l2*Qn**2/2)*np.exp(-l2*Qn**2/4)
        elif f_kind=='custom': f=f_custom(Qn,l2)
        kc=S.kcart                                                    # (Nk,2) includes twist
        ka=kc[S.tgt]                                                  # (nQ,Nk,2)
        s=S.svec; svec=s[...,0:1]*band.b1+s[...,1:2]*band.b2          # (nQ,Nk,2)
        ph=0.5*l2*cross(Q[:,None,:],kc[None,:,:])+0.5*l2*cross(svec,ka)
        ph=sig*ph+np.pi*s[...,0]*s[...,1]
        # parallel-transport gauge with respect to the boundary twist t (orbital c at k0_c+t aligned to k0_c):
        k0=band.frac2cart(np.stack([cl.X1/cl.D,cl.X2/cl.D],1)); t=kc-k0
        pt=sig*0.5*l2*cross(t,k0)                                     # (Nk,)
        ph=ph+pt[None,:]-pt[S.tgt]
        return f[:,None]*np.exp(1j*ph)
    return fn

def triangle_phase(S,nmax=6):
    """gauge-invariant mean phase of lambda_c(Q1) lambda_{c+Q1}(Q2) conj(lambda_c(Q1+Q2)) for the shortest Q's (chirality)"""
    cl=S.cl; Qs=S.Qs; look={(int(q[0]),int(q[2]),int(q[3])):i for i,q in enumerate(Qs)}
    idx=[i for i in np.argsort(Qs[:,4]) if Qs[i,4]>1e-9][:nmax]
    vals=[]
    for i in idx:
        for j in idx:
            Q1=Qs[i,5:7];Q2=Qs[j,5:7]; c12=cross(Q1,Q2)
            if abs(c12)<1e-9: continue
            # find index of Q1+Q2: mesh part u1+u2, integer part m1+m2+carry
            u1,u2=int(Qs[i,0]),int(Qs[j,0]); x1=cl.X1[u1]+cl.X1[u2]; x2=cl.X2[u1]+cl.X2[u2]
            u3=cl.grid[x1%cl.D,x2%cl.D]; k3=look.get((int(u3),int(Qs[i,2]+Qs[j,2]+x1//cl.D),int(Qs[i,3]+Qs[j,3]+x2//cl.D)))
            if k3 is None: continue
            L1=S.Lam[i]; a=S.tgt[i]; L2=S.Lam[j][a]; L3=S.Lam[k3]
            w=np.mean(L1*L2*np.conj(L3)); vals.append((c12,np.angle(w),abs(w)))
    return vals
