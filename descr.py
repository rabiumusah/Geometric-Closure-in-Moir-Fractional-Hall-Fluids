"""Gauge-invariant k-space inhomogeneity descriptors of a Chern band (beyond BZ-averaged cumulants)."""
import numpy as np
from band import *

def mesh_states(band,N,shift=(0,0)):
    f=(np.arange(N)+0.0)/N; F1,F2=np.meshgrid(f+shift[0]/N,f+shift[1]/N,indexing='ij')
    fr=np.stack([F1.ravel(),F2.ravel()],1); _,v=band.top(band.frac2cart(fr)); return fr,v[:,:,0].reshape(N,N,-1)

def berry(band,N=24):
    """plaquette Berry curvature on an N x N mesh (periodic-gauge links); returns Omega_k (A^2) array"""
    _,v=mesh_states(band,N)
    def link(a,b,s):  # <a|b> with b displaced by reciprocal shift s relative to a
        return shifted_overlap(a,b,band,s) if s!=(0,0) else np.sum(np.conj(a)*b,axis=-1)
    Om=np.zeros((N,N))
    Ab=abs(band.b1[0]*band.b2[1]-band.b1[1]*band.b2[0])/N**2
    for i in range(N):
        for j in range(N):
            i1=(i+1)%N; j1=(j+1)%N; si=1 if i+1==N else 0; sj=1 if j+1==N else 0
            # <u_k|u_{k+d}> where u_{k+b} index shifted: shifted_overlap(x,y,s) = <x(+s)|y>
            U1=shifted_overlap(v[i1,j],v[i,j],band,(si,0))
            U2=shifted_overlap(v[i1,j1],v[i1,j],band,(0,sj))
            U3=np.conj(shifted_overlap(v[i1,j1],v[i,j1],band,(si,0)))
            U4=np.conj(shifted_overlap(v[i,j1],v[i,j],band,(0,sj)))
            # note: shifted_overlap(a,b,s) uses a shifted by s; for wrap links this implements u_{k+b}=e^{-ibr}u_k
            Om[i,j]=-np.angle(U1*U2*U3*U4)/Ab
    return Om

def metric_trace(band,N=24,h=1e-3):
    """tr g_k from 1-|<u_k|u_{k+h e}>|^2 = h^2 g_ee, averaged over x,y"""
    fr,v=mesh_states(band,N); K=band.frac2cart(fr)
    tr=np.zeros(len(K))
    for e in (np.array([1,0.]),np.array([0,1.])):
        _,v2=band.top(K+h*e); v2=v2[:,:,0]
        tr+=(1-np.abs(np.sum(np.conj(v.reshape(len(K),-1))*v2,axis=1))**2)/h**2
    return tr.reshape(N,N)

def form_factor_stats(band,N=18,shells=None):
    """mean and relative std over k of |F_{k,Q}| for representative Q (cartesian)"""
    fr,v=mesh_states(band,N); K=band.frac2cart(fr); v=v.reshape(len(K),-1)
    g1=band.b1; out={}
    if shells is None:
        shells={'M (g/2)':0.5*g1,'K (|g|/sqrt3)':(2*band.b1-band.b2)/3*0+ (band.b1+band.b2)/3,'g':g1,'sqrt3 g':band.b1+band.b2}
    for nm,Q in shells.items():
        _,v2=band.top(K+Q); v2=v2[:,:,0]
        F=np.abs(np.sum(np.conj(v2)*v,axis=1)); out[nm]=(float(F.mean()),float(F.std()/F.mean()))
    return out

def describe(par,N=24):
    b=Band(par)
    Om=berry(b,N); trg=metric_trace(b,N)
    Ob=Om.mean(); Auc=b.Auc
    d=dict(C=float(Om.sum()*abs(b.b1[0]*b.b2[1]-b.b1[1]*b.b2[0])/N**2/(2*np.pi)),
           Omega_mean=float(Ob),sigma_Omega=float(Om.std()/abs(Ob)),
           trg_mean=float(trg.mean()),sigma_trg=float(trg.std()/trg.mean()),
           trace_violation=float((trg-np.abs(Om)).mean()/abs(Ob)),
           local_trace_min=float((trg-np.abs(Om)).min()/abs(Ob)))
    d['F']=form_factor_stats(b,18)
    return d
