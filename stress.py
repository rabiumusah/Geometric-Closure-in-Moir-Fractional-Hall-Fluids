"""Physical uniform strain (shear) vertex of the projected many-body Hamiltonian.
H(eps) is the projected Hamiltonian on the comoving cluster (fixed fractional momenta, fixed Q labels) after an
area-preserving deformation b_i -> exp(-eps E_phi) b_i of the moire lattice; Bloch states are put in the
parallel-transport gauge <u_k(0)|u_k(eps)> > 0.  Pi = dH/deps, K2 = d2H/deps2 (contact term).
Kubo identity checked:  E0''(0) = <0|K2|0> - 2 sum_n |<n|Pi|0>|^2/(E_n-E0)."""
import numpy as np
from scipy.sparse.linalg import eigsh, minres, LinearOperator
from ana import *
from refband import hom_lam

def Ephi(phi_deg):
    p=np.deg2rad(phi_deg); return np.array([[np.cos(2*p),np.sin(2*p)],[np.sin(2*p),-np.cos(2*p)]])

def systems(par,cl,phi_deg,dlt,lam_fn=None,base_E=None):
    """systems at eps = -2d,-d,0,+d,+2d along E_phi (exp map) on top of base strain base_E"""
    out={}
    p0=par.copy(Gcut=3.1,smap='exp',Eextra=base_E)
    S0=System(Band(p0),cl,lam_fn=lam_fn)
    out[0]=S0
    for m in (-2,-1,1,2):
        Ex=m*dlt*Ephi(phi_deg)+(0 if base_E is None else base_E)
        pm=par.copy(Gcut=3.1,smap='exp',Eextra=Ex)
        out[m]=System(Band(pm),cl,lam_fn=lam_fn,v_ref=S0.v,Qs_ref=S0.Qs)
    return out

def vertex(par,cl,K,phi_deg=0.0,dlt=2e-3,lam_fn=None,dense_max=6000,want_spec=True,g0=None):
    Ss=systems(par,cl,phi_deg,dlt,lam_fn)
    H={m:Ss[m].H(K) for m in Ss}
    Pi=(H[1]-H[-1])/(2*dlt); Pi4=(-H[2]+8*H[1]-8*H[-1]+H[-2])/(12*dlt)
    K2=(H[1]+H[-1]-2*H[0])/dlt**2
    dim=H[0].shape[0]
    def lowest(Hm):
        if dim<=dense_max: return np.linalg.eigvalsh(Hm.toarray())[0]
        return eigsh(Hm,k=1,which='SA',tol=1e-13)[0][0]
    if dim<=dense_max:
        e,V=np.linalg.eigh(H[0].toarray()); E0=e[0]; g=V[:,0]
    else:
        e,V=eigsh(H[0],k=6,which='SA',tol=1e-13); o=np.argsort(e); e=e[o]; V=V[:,o]; E0=e[0]; g=V[:,0]
    if g0 is not None:   # common ground-state phase for different strain channels (needed for chirality)
        ph=np.vdot(g0,g); g=g*np.conj(ph)/abs(ph)
        if dim<=dense_max: V[:,0]=g
    Em=lowest(H[-1]); Ep=lowest(H[1]); Em2=lowest(H[-2]); Ep2=lowest(H[2])
    d2E=(Ep+Em-2*E0)/dlt**2
    d2E4=(-Ep2+16*Ep-30*E0+16*Em-Em2)/(12*dlt**2)
    pi=Pi4@g; meanPi=np.vdot(g,pi).real; pi=pi-meanPi*g
    contact=np.vdot(g,K2@g).real
    out=dict(dim=dim,E0=E0,meanPi=meanPi,contact=contact,d2E=d2E,d2E4=d2E4,norm2=float(np.vdot(pi,pi).real))
    if dim<=dense_max:
        A=V[:,1:].conj().T@pi; En=e[1:]-E0
        out['chi']=2*float(np.sum(np.abs(A)**2/En)); out['E']=En; out['w']=np.abs(A)**2
        out['m1']=float(np.sum(En*np.abs(A)**2))
    else:
        from linsolve import hsolve
        Hs=H[0]; x,info=hsolve(Hs,E0,g,pi)
        out['chi']=2*float(np.vdot(pi,x).real); out['cg']=info
        y=Hs@pi-E0*pi; out['m1']=float(np.vdot(pi,y).real)
    out['mu_kubo']=out['contact']-out['chi']
    out['ward_rel']=abs(out['mu_kubo']-out['d2E4'])/max(abs(out['d2E4']),1e-12)
    out['pi']=pi; out['gs']=g; out['S0']=Ss[0]; out['V']=V if dim<=dense_max else None; out['e']=e
    return out

def multipole_fidelity(S,cl,K,gs,pi,ell,radial='F'):
    """fraction of the physical stress state pi captured by the umklapp-shell multipole states at u=0"""
    Lam=channel_coefs(S,0,ell,radial)
    V=np.array([S.rho_op(K,Lam[a],0)[0]@gs for a in range(5)]).T
    V=V-np.outer(gs,gs.conj()@V)
    res={}
    for name,cols in (('rho',[0]),('2',[1,2]),('4',[3,4]),('0+2',[0,1,2]),('2+4',[1,2,3,4]),('all',[0,1,2,3,4])):
        X,sv,_=np.linalg.svd(V[:,cols],full_matrices=False); X=X[:,sv>1e-6*sv.max()]
        res[name]=float(np.linalg.norm(X.conj().T@pi)**2/np.vdot(pi,pi).real)
    return res

def manifold_vertex(par,cl,lam_fn=None,dlt=2e-3,dense_max=6000,want_fid=False,ell=None):
    """both shear channels for each member of the three-fold manifold; chirality of stress weight;
    manifold-averaged contact, paramagnetic and Kubo modulus; Ward-identity residual."""
    S0=System(Band(par.copy(Gcut=3.1,smap='exp')),cl,lam_fn=lam_fn)
    low=[]
    for K in range(cl.Nk):
        H=S0.H(K); low.append(np.linalg.eigvalsh(H.toarray())[0] if H.shape[0]<=1200 else eigsh(H,k=1,which='SA',tol=1e-12)[0][0])
    Ks=[int(K) for K in np.argsort(low)[:3]]
    mem=[]
    for K in Ks:
        r0=vertex(par,cl,K,0.0,dlt,lam_fn,dense_max); r45=vertex(par,cl,K,45.0,dlt,lam_fn,dense_max,g0=r0['gs'])
        p0,p45=r0['pi'],r45['pi']
        Wp=np.vdot(p0+1j*p45,p0+1j*p45).real; Wm=np.vdot(p0-1j*p45,p0-1j*p45).real
        m=dict(K=K,meanPi0=r0['meanPi'],meanPi45=r45['meanPi'],mu0=r0['mu_kubo'],mu45=r45['mu_kubo'],
               contact0=r0['contact'],contact45=r45['contact'],chi0=r0['chi'],chi45=r45['chi'],
               ward=max(r0['ward_rel'],r45['ward_rel']),Wp=Wp,Wm=Wm,chir=(Wp-Wm)/(Wp+Wm),dim=r0['dim'])
        if r0['V'] is not None:
            V=r0['V']; e=r0['e']; En=e[1:]-e[0]
            ap=np.abs(V[:,1:].conj().T@(p0+1j*p45))**2; am=np.abs(V[:,1:].conj().T@(p0-1j*p45))**2
            m['E']=En; m['wp']=ap; m['wm']=am
            m['m1p']=float(np.sum(En*ap)/np.sum(ap)); m['m1m']=float(np.sum(En*am)/np.sum(am))
        if want_fid:
            m['fid0']=multipole_fidelity(r0['S0'],cl,K,r0['gs'],p0,ell)
        mem.append(m)
    avg={k:float(np.mean([m[k] for m in mem])) for k in ('meanPi0','meanPi45','mu0','mu45','contact0','contact45','chi0','chi45','Wp','Wm')}
    avg['chir']=(avg['Wp']-avg['Wm'])/(avg['Wp']+avg['Wm']); avg['ward']=max(m['ward'] for m in mem)
    avg['mu_iso']=0.5*(avg['mu0']+avg['mu45'])/cl.Ne; avg['contact_iso']=0.5*(avg['contact0']+avg['contact45'])/cl.Ne
    avg['chi_iso']=0.5*(avg['chi0']+avg['chi45'])/cl.Ne
    return avg,mem
