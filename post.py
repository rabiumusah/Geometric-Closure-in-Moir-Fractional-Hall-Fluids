import numpy as np, pickle, os, glob
from scipy import stats
RES='/home/claude/sim/res'
def load(name): return pickle.load(open(f'{RES}/{name}.pkl','rb'))

def active_idx(E,W,thr=0.05,nmax=8):
    tot=W[0]+W[1]+W[2]
    return [i for i in range(min(nmax,len(E))) if tot[i]>thr]

def f4_of(W): return W[2]/np.maximum(W[1]+W[2],1e-30)

def mixing(E,W,act):
    """coupling-strength-independent two-state mixing angle from the two lowest active states"""
    if len(act)<2: return np.nan,np.nan
    i,j=act[0],act[1]
    R1=W[2][i]/max(W[1][i],1e-30); R2=W[2][j]/max(W[1][j],1e-30)
    t2=np.sqrt(R1/R2) if R2>0 and R1>0 else 0.0
    t=np.sqrt(t2); s2=2*t/(1+t2)
    return 0.5*(E[j]-E[i])*s2, s2

def roton_u(run):
    ud=run['u']; us=[k for k in ud if k!=0]
    return min(us,key=lambda k:ud[k]['E'][0])

def obs(run,u=None,thr=0.05):
    ud=run['u']
    if u is None: u=roton_u(run)
    E,W=ud[u]['E'],ud[u]['W']
    act=active_idx(E,W,thr)
    o=dict(u=u,E_R=float(E[0]),q=float(ud[u]['q']))
    if act:
        a=act[0]; o.update(E_A=float(E[a]),W0_A=float(W[0][a]),W2_A=float(W[1][a]),W4_A=float(W[2][a]),f4_A=float(f4_of(W)[a]))
        f4=f4_of(W); b=max(act[:6],key=lambda i:f4[i]); o.update(E_B=float(E[b]),f4_B=float(f4[b]),W4_B=float(W[2][b]))
    M,s2=mixing(E,W,act); o.update(M24=float(M),sin2th=float(s2))
    return o

def fss(Ne,y,sig,level=0.95,kaps=(0.5,1.0,1.5,2.0)):
    Ne=np.asarray(Ne,float); y=np.asarray(y,float); sig=np.asarray(sig,float); n=len(Ne)
    out=[]
    for k in kaps:
        A=np.stack([np.ones(n),Ne**-k],1); W=1/sig**2
        cov=np.linalg.inv(A.T@(A*W[:,None])); b=cov@(A.T@(W*y))
        chi2=float(((y-A@b)**2*W).sum()); aicc=chi2+4+(12/(n-3) if n>3 else 0)
        out.append((b[0],cov[0,0],aicc,chi2,b[1]))
    o=np.array(out); wt=np.exp(-0.5*(o[:,2]-o[:,2].min())); wt/=wt.sum()
    m=(wt*o[:,0]).sum(); s=(wt*np.sqrt(o[:,1]+(o[:,0]-m)**2)).sum()
    nu=max(n-2,1); t=stats.t.ppf(0.5+level/2,nu)
    return dict(mean=float(m),sigma=float(s),half=float(t*s),w=wt.tolist(),per=o[:,0].tolist(),nu=nu)

# ---------------------------------------------------------------- operator-space (Theta) observables
def theta_obs(r):
    """r = analysis dict for one target momentum u. returns E2,E4,M24,f4 of lowest 4x4 eigvec, pole shift"""
    Th=r['Theta']; lab=np.array(r['lab']); i2=np.where(lab==1)[0]; i4=np.where(lab==2)[0]
    if len(i2)==0 or len(i4)==0: return None
    T4=Th[np.ix_(np.r_[i2,i4],np.r_[i2,i4])]; T4=(T4+T4.conj().T)/2
    D2=T4[:len(i2),:len(i2)]; D4=T4[len(i2):,len(i2):]; M=T4[:len(i2),len(i2):]
    E2=np.real(np.trace(D2))/len(i2); E4=np.real(np.trace(D4))/len(i4)
    M24=np.linalg.svd(M,compute_uv=False)[0]; M24F=float(np.linalg.norm(M)); dE=max(E4-E2,1e-6)
    sig2=M24F**2/dE; f4pt=M24F**2/dE**2
    w,v=np.linalg.eigh(T4); f4=float(np.sum(np.abs(v[len(i2):,0])**2))
    w2=np.linalg.eigvalsh((D2+D2.conj().T)/2)
    return dict(E2=float(E2),E4=float(E4),M24=float(M24),M24F=M24F,sig2=float(sig2),f4pt=float(f4pt),f4=f4,w_low=float(w[0]),w_hi=float(w[-1]),shift=float(w[0]-w2[0]),
                E_rho=float(np.real(Th[0,0])),Theta=Th)

def window_F4(r,fac=1.4):
    E,W=r['E'],r['W']; Ec=fac*E[0]
    m=E<=Ec
    ok=(E[-1]>=Ec)
    w2=W[1][m].sum(); w4=W[2][m].sum()
    return float(w4/max(w2+w4,1e-30)),bool(ok)

def point_obs(run,u=None):
    ud=run['u']
    if u is None: u=roton_u(run)
    r=ud[u]; o=theta_obs(r); F4,ok=window_F4(r)
    o=dict(o) if o else {}
    o.update(u=u,E_R=float(r['E'][0]),F4=F4,F4_ok=ok,q=float(r['q']))
    o.pop('Theta',None)
    return o
