"""Distance of a Chern band from the homogeneous (Landau-level) class on the cluster, and perturbative analysis.
eta^2 = min_gauge sum_Q w_Q sum_c |lambda_c(Q) - lambda^hom_c(Q)|^2 / sum_Q w_Q sum_c |lambda^hom_c(Q)|^2,
lambda^hom = twin (same cluster-averaged |lambda|(Q), magnetic-algebra phases). Gauge fixed by phase synchronization."""
import numpy as np
from ana import *
from refband import hom_lam
from ed import KE2

def weights(S,d=300.,epsr=10.):
    Qn=S.Qs[:,4]; w=np.where(Qn>1e-12,2*np.pi*KE2*np.tanh(Qn*d)/(epsr*np.maximum(Qn,1e-12)),0.0)
    return w

def sync_gauge(S_m,S_h,iters=200):
    """z_c maximizing Re sum w conj(lam_h) lam_m z_c conj(z_a)"""
    Nk=S_m.cl.Nk; w=weights(S_m)
    B=np.zeros((Nk,Nk),complex)
    for iq in range(len(S_m.Qs)):
        a=S_m.tgt[iq]; B[a,np.arange(Nk)]+=w[iq]*np.conj(S_h.Lam[iq])*S_m.Lam[iq]
    Bh=(B+B.conj().T)/2
    ev,U=np.linalg.eigh(Bh); z=U[:,-1]; z=z/np.abs(z)
    for _ in range(iters):   # projected power iteration on the unit-modulus manifold
        z2=Bh@z; z2=z2/np.abs(z2)
        if np.allclose(z2,z): break
        z=z2
    return z

def eta(S_m,S_h,z):
    w=weights(S_m)[:,None]
    Lm=S_m.Lam*z[None,:]*np.conj(z[S_m.tgt])
    return float(np.sqrt(np.sum(w*np.abs(Lm-S_h.Lam)**2)/np.sum(w*np.abs(S_h.Lam)**2))),Lm

def eta_mag_phase(S_m,S_h,z):
    """split eta^2 into a magnitude part (|lam|-<|lam|>) and a phase part"""
    w=weights(S_m)[:,None]; Lm=S_m.Lam*z[None,:]*np.conj(z[S_m.tgt]); den=np.sum(w*np.abs(S_h.Lam)**2)
    mag=np.sum(w*(np.abs(Lm)-np.abs(S_h.Lam))**2)/den
    tot=np.sum(w*np.abs(Lm-S_h.Lam)**2)/den
    return float(mag),float(tot-mag)

def best_origin(S_m,S_h,n=12):
    """scan the guiding-centre origin r0 of the homogeneous reference over the moire cell; return (eta,z,r0,Lam_h(r0))"""
    best=None; b=S_m.band
    a1=2*np.pi*np.array([b.b2[1],-b.b2[0]])/(b.b1[0]*b.b2[1]-b.b1[1]*b.b2[0])
    a2=2*np.pi*np.array([-b.b1[1],b.b1[0]])/(b.b1[0]*b.b2[1]-b.b1[1]*b.b2[0])
    Q=S_m.Qs[:,5:7]; Lh0=S_h.Lam.copy()
    for i in range(n):
        for j in range(n):
            r0=i/n*a1+j/n*a2
            S_h.Lam=Lh0*np.exp(-1j*(Q@r0))[:,None]
            z=sync_gauge(S_m,S_h,50); e,_=eta(S_m,S_h,z)
            if best is None or e<best[0]: best=(e,z,r0,(i,j))
    S_h.Lam=Lh0*np.exp(-1j*(Q@best[2]))[:,None]
    z=sync_gauge(S_m,S_h,500); e,_=eta(S_m,S_h,z)
    return e,z,best[2],best[3]

def cluster_ILambda(band,cl,lam_fn=None,h=None,d=300.,epsr=10.):
    """single-particle inhomogeneity functional of the covariant form-factor derivative on the cluster:
    I = (1/Nk) sum_Q V(Q)^2 sum_a sum_c |D_a lam_c(Q) - Lbar_a(Q) lam_c(Q)|^2,  Lbar_a(Q) = <lam|D_a lam>/<lam|lam>,
    D_a lam = d lam/dt_a in the parallel-transport gauge (boundary twist).  Vanishes for homogeneous bands."""
    if h is None: h=1e-3*band.g0
    S0=System(band,cl,lam_fn=lam_fn); w=weights(S0,d,epsr)/(cl.Nk*band.Auc)   # V(Q)/A as in H
    I=0.0; Ia=[]
    B=np.stack([band.b1,band.b2],1)
    for e in (np.array([1.,0]),np.array([0,1.])):
        L={}
        for m in (-1,1):
            tf=np.linalg.solve(B,m*h*e)
            L[m]=System(band,cl,twist=tuple(tf),lam_fn=lam_fn,v_ref=S0.v,Qs_ref=S0.Qs).Lam
        DL=(L[1]-L[-1])/(2*h)
        Lb=np.sum(np.conj(S0.Lam)*DL,axis=1)/np.maximum(np.sum(np.abs(S0.Lam)**2,axis=1),1e-300)
        res=DL-Lb[:,None]*S0.Lam
        Ii=float(np.sum(w[:,None]**2*np.abs(res)**2)/cl.Nk); Ia.append(Ii); I+=Ii
    return I,Ia
