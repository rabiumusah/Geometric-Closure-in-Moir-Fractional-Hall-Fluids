import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
def hsolve(H0,E0,g,v,rtol=1e-11,maxiter=20000):
    """solve (H0-E0) x = v on the complement of the ground state g (v assumed orthogonal to g); x orthogonal to g.
    Operator (H0-E0)P + P0 is Hermitian positive definite -> conjugate gradients (complex)."""
    dim=H0.shape[0]
    def mv(x):
        c=np.vdot(g,x); xp=x-c*g; y=H0@xp-E0*xp; y=y-np.vdot(g,y)*g; return y+c*g
    Lo=LinearOperator((dim,dim),matvec=mv,dtype=complex)
    try: x,info=cg(Lo,v,rtol=rtol,maxiter=maxiter)
    except TypeError: x,info=cg(Lo,v,tol=rtol,maxiter=maxiter)
    return x-np.vdot(g,x)*g,info

def hsolve_multi(H0,E0,Gs,v,rtol=1e-11,maxiter=20000):
    """as hsolve but projecting out a set of (near-)degenerate manifold states Gs (dim,m), orthonormal"""
    dim=H0.shape[0]
    def P(x): return x-Gs@(Gs.conj().T@x)
    def mv(x):
        c=Gs.conj().T@x; xp=x-Gs@c; y=H0@xp-E0*xp; y=P(y); return y+Gs@c
    Lo=LinearOperator((dim,dim),matvec=mv,dtype=complex)
    try: x,info=cg(Lo,P(v),rtol=rtol,maxiter=maxiter)
    except TypeError: x,info=cg(Lo,P(v),tol=rtol,maxiter=maxiter)
    return P(x),info
