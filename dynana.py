import numpy as np, pickle, glob, os
R='/home/claude/sim/res3'
def load(n,v,Ne):
    fn=f'{R}/{n}_{v}_{Ne}.pkl'
    return pickle.load(open(fn,'rb')) if os.path.exists(fn) else None
def minq(d,K):
    Kc=np.array(d['Kcart']); g1,g2=map(np.array,d['gb']); best=9
    for K0 in d['man']:
        dq=Kc[K]-Kc[K0]
        for m1 in range(-3,4):
            for m2 in range(-3,4):
                best=min(best,np.linalg.norm(dq+m1*g1+m2*g2))
    return best
def dispersion(d):
    out=[(minq(d,K),e) for K,e in d['disp'] if K not in d['man']]
    return np.array(sorted(out))
def binned(d,key='S',tol=1e-3):
    q=np.array([r['q'] for r in d['res']]); y=np.array([r[key] for r in d['res']])
    o=np.argsort(q); q=q[o]; y=y[o]; qs=[];ys=[]
    i=0
    while i<len(q):
        j=i
        while j<len(q) and q[j]-q[i]<tol: j+=1
        qs.append(q[i:j].mean()); ys.append(y[i:j].mean()); i=j
    return np.array(qs),np.array(ys)
def Sqw(d,w,eta=0.4,qsel=None):
    """continued-fraction S(q,w) for each transfer; returns q list and spectra (averaged over degenerate |q|)"""
    rows=[]
    for r in d['res']:
        if len(r['al'])==0: continue
        z=w+1j*eta; al,be=r['al'],r['be']; G=np.zeros_like(z)
        for j in range(len(al)-1,-1,-1):
            G=1.0/(z-al[j]-(be[j]**2*G if j<len(be) else 0))
        rows.append((r['q'],-G.imag/np.pi*r['n2']))
    return rows
if __name__=='__main__':
    for Ne in (6,7,8):
        for n,v in (('LLL','main'),('A','twin'),('A','main'),('P3','twin'),('P3','main')):
            d=load(n,v,Ne)
            if d is None: continue
            D=dispersion(d); qS,S=binned(d,'S'); qf,f=binned(d,'f')
            k=np.argmin(D[:,1]); 
            print(f'{n:4s}{v:5s} Ne={Ne} gap={D[:,1].min():.3f} at q={D[k,0]:.2f}  |  S(q) first:',' '.join(f'{a:.2f}:{b:.4f}' for a,b in list(zip(qS,S))[:6]),' Smax=%.3f@%.2f'%(S.max(),qS[S.argmax()]))
