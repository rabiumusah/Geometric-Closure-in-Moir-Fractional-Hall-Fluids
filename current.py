"""Uniform (q->0) current of the band-projected many-body problem: J_a = dH/dt_a, the derivative of the projected
Hamiltonian with respect to a uniform shift k -> k + t (boundary twist / vector potential), with Bloch states in the
parallel-transport gauge and fixed Q labels.  For homogeneous (GMP-class) bands H(t) = T(t) H T(t)^dagger with T a
many-body guiding-centre translation, so J|0> has no weight on excited states of the sector: the O(q) current and the
q^2 terms of the projected oscillator strength and structure factor vanish.  For inhomogeneous bands they do not.
  f(q) = sum_n (E_n-E0)|<n|rho_q|0>|^2 / N  ->  q_a q_b D_ab / N,   D_ab = sum_n Re J^a_{0n} J^b_{n0}/(E_n-E0)
  S(q) = sum_n |<n|rho_q|0>|^2 / N        ->  q_a q_b K_ab / N,   K_ab = sum_n Re J^a_{0n} J^b_{n0}/(E_n-E0)^2
Ward identity: d2E0/dt_a^2 = <0|d2H/dt_a^2|0> - 2 D_aa."""
import numpy as np
from scipy.sparse.linalg import eigsh, minres, LinearOperator
from ana import *

def twisted(band,cl,t_cart,lam_fn=None,S0=None,tw0=(0.,0.),d=300.,epsr=10.):
    t_frac=np.linalg.solve(np.stack([band.b1,band.b2],1),t_cart)+np.array(tw0)
    return System(band,cl,twist=tuple(t_frac),lam_fn=lam_fn,v_ref=None if S0 is None else S0.v,Qs_ref=None if S0 is None else S0.Qs,d=d,epsr=epsr)

def current_response(band,cl,K,lam_fn=None,h=None,dense_max=1200,tw0=(0.,0.),d=300.,epsr=10.,keep_spec=False):
    if h is None: h=1e-3*band.g0
    S0=System(band,cl,lam_fn=lam_fn,twist=tuple(tw0),d=d,epsr=epsr)
    H0=S0.H(K); dim=H0.shape[0]
    Hs={}
    for a,e in ((0,np.array([1.,0])),(1,np.array([0,1.]))):
        for m in (-1,1):
            Hs[(a,m)]=twisted(band,cl,m*h*e,lam_fn,S0,tw0,d,epsr).H(K)
    J=[(Hs[(a,1)]-Hs[(a,-1)])/(2*h) for a in (0,1)]
    D2=[(Hs[(a,1)]+Hs[(a,-1)]-2*H0)/h**2 for a in (0,1)]
    if dim<=dense_max:
        e,V=np.linalg.eigh(H0.toarray()); E0=e[0]; g=V[:,0]
        low=lambda Hm: np.linalg.eigvalsh(Hm.toarray())[0]
    else:
        e,V=eigsh(H0,k=4,which='SA',tol=1e-13); o=np.argsort(e); e=e[o]; V=V[:,o]; E0=e[0]; g=V[:,0]
        low=lambda Hm: eigsh(Hm,k=1,which='SA',tol=1e-13)[0][0]
    out=dict(dim=dim,E0=E0)
    jv=[]
    for a in (0,1):
        v=J[a]@g; out[f'meanJ{a}']=float(np.vdot(g,v).real); v=v-np.vdot(g,v)*g; jv.append(v)
    out['WJ']=float(sum(np.vdot(v,v).real for v in jv))       # sum_n |J_n0|^2 (both components)
    if dim<=dense_max:
        A=[V[:,1:].conj().T@v for v in jv]; En=e[1:]-E0
        out['D']=np.array([[np.sum((A[a].conj()*A[b]).real/En) for b in (0,1)] for a in (0,1)])
        out['Kss']=np.array([[np.sum((A[a].conj()*A[b]).real/En**2) for b in (0,1)] for a in (0,1)])
        out['E']=En; out['w']=np.abs(A[0])**2+np.abs(A[1])**2
    else:
        from linsolve import hsolve
        solve=lambda v: hsolve(H0,E0,g,v)[0]
        X=[solve(v) for v in jv]
        out['D']=np.array([[np.vdot(jv[a],X[b]).real for b in (0,1)] for a in (0,1)])
        X2=[solve(x) for x in X]
        out['Kss']=np.array([[np.vdot(jv[a],X2[b]).real for b in (0,1)] for a in (0,1)])
    # Ward identity (charge stiffness): d2E0/dt2 = <D2> - 2 D_aa
    out['stiff']=[]; out['ward']=[]
    for a in (0,1):
        d2E=(low(Hs[(a,1)])+low(Hs[(a,-1)])-2*E0)/h**2
        kub=np.vdot(g,D2[a]@g).real-2*out['D'][a,a]
        out['stiff'].append(float(d2E)); out['ward'].append(float(abs(d2E-kub)/max(abs(np.vdot(g,D2[a]@g).real),1e-12)))
    out['D_iso']=float(np.trace(out['D'])/2); out['K_iso']=float(np.trace(out['Kss'])/2)
    return out

def manifold_current(band,cl,lam_fn=None,dense_max=1200,tw0=(0.,0.),d=300.,epsr=10.,keep_spec=False):
    S0=System(band,cl,lam_fn=lam_fn,twist=tuple(tw0),d=d,epsr=epsr)
    low=[]
    for K in range(cl.Nk):
        H=S0.H(K); low.append(np.linalg.eigvalsh(H.toarray())[0] if H.shape[0]<=1200 else eigsh(H,k=1,which='SA',tol=1e-12)[0][0])
    Ks=[int(K) for K in np.argsort(low)[:3]]
    mem=[current_response(band,cl,K,lam_fn,dense_max=dense_max,tw0=tw0,d=d,epsr=epsr) for K in Ks]
    avg=dict(D_iso=float(np.mean([m['D_iso'] for m in mem])),K_iso=float(np.mean([m['K_iso'] for m in mem])),
             WJ=float(np.mean([m['WJ'] for m in mem])),ward=float(max(max(m['ward']) for m in mem)),
             stiff=float(np.mean([np.mean(m['stiff']) for m in mem])))
    return avg,mem
