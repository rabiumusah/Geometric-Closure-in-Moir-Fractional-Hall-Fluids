import numpy as np
from band import *

def mesh_cart(band,N):
    f=np.arange(N)/N; F1,F2=np.meshgrid(f,f,indexing='ij')
    return band.frac2cart(np.stack([F1.ravel(),F2.ravel()],1))

def logF_samples(band, Nmesh, qs, phis, chunk=500):
    """y[iq,iphi]=log <|F_{k,q}|>_BZ on a Nmesh^2 mesh (physical q in 1/A)."""
    K=mesh_cart(band,Nmesh); _,v0=band.top(K); v0=v0[:,:,0]
    out=np.zeros((len(qs),len(phis)))
    for ip,ph in enumerate(phis):
        d=np.array([np.cos(ph),np.sin(ph)])
        for iq,q in enumerate(qs):
            _,v1=band.top(K+q*d); v1=v1[:,:,0]
            out[iq,ip]=np.log(np.mean(np.abs(np.sum(np.conj(v1)*v0,axis=1))))
    return out

def design(q,phi,order):
    qx,qy=q*np.cos(phi),q*np.sin(phi)
    cols=[-0.5*qx**2,-0.5*qy**2,-qx*qy,
          q**4/24,q**4*np.cos(2*phi)/24,q**4*np.sin(2*phi)/24,q**4*np.cos(4*phi)/24,q**4*np.sin(4*phi)/24]
    if order>=6: cols+=[q**6,q**6*np.cos(2*phi),q**6*np.sin(2*phi),q**6*np.cos(4*phi),q**6*np.sin(4*phi),q**6*np.cos(6*phi),q**6*np.sin(6*phi)]
    if order>=8: cols+=[q**8,q**8*np.cos(2*phi),q**8*np.sin(2*phi),q**8*np.cos(4*phi),q**8*np.sin(4*phi)]
    return np.stack(cols,1)

def fit(qs,phis,y,order,ell,rng=None,boot=0):
    Q,P=np.meshgrid(qs,phis,indexing='ij'); Q=Q.ravel();P=P.ravel();Y=y.ravel()
    # rescale to dimensionless q*ell for conditioning
    x=Q*ell; A=design(x,P,order); w=1/x**2
    beta=np.linalg.lstsq(A*w[:,None],Y*w,rcond=None)[0]
    res=Y-A@beta; sd=None
    if boot and rng is not None:
        bs=[]
        for _ in range(boot):
            yb=A@beta+rng.choice(res,res.size)
            bs.append(np.linalg.lstsq(A*w[:,None],yb*w,rcond=None)[0])
        sd=np.std(bs,0)
    return beta,sd,np.sqrt(np.mean(res**2))

def summarize(beta,ell):
    g=np.array([[beta[0],beta[2]],[beta[2],beta[1]]])*ell**2     # A^2
    a0,a2c,a2s,a4c,a4s=beta[3:8]
    gstar=np.sqrt(np.linalg.det(g)); 
    T0=a0*ell**4; T2=np.hypot(a2c,a2s)*ell**4; T4=np.hypot(a4c,a4s)*ell**4
    return dict(g=g,gstar=gstar,T0=T0,T2=T2,T4=T4,tau0=T0/gstar**2,tau2=T2/gstar**2,tau4=T4/gstar**2,
                phi2=0.5*np.arctan2(a2s,a2c),phi4=0.25*np.arctan2(a4s,a4c),beta=beta)

def geometry(band,Nmesh=10,qrel=0.45,nq=8,order=8,phis=None,rng=None,boot=0,ell=None):
    """qrel: window q_max*ell. ell=sqrt(gstar) estimated from small-q pre-fit if None."""
    if phis is None: phis=np.arange(8)*np.pi/8
    if ell is None:
        qs0=np.array([0.02,0.03])*band.g0/0.14*0.14 # tiny q
        y0=logF_samples(band,Nmesh,qs0,phis[::3])
        gdir=-2*y0[0]/qs0[0]**2; gstar_est=np.mean(gdir)
        ell=1.0/np.sqrt(gstar_est)   # units 1/A -> ell variable multiplies q => ell=1/sqrt(g)? we want x=q*sqrt(g)
        ell=np.sqrt(gstar_est)
    qs=np.linspace(0.25,1.0,nq)*qrel/ell
    y=logF_samples(band,Nmesh,qs,phis)
    beta,sd,rms=fit(qs,phis,y,order,ell,rng,boot)
    S=summarize(beta,ell); S['ell']=ell; S['rms']=rms; S['sd_beta']=sd; S['qs']=qs; S['y']=y
    return S
