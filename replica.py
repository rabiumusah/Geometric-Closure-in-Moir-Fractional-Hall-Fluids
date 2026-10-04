import numpy as np,sys
from dynana import load
def poles(r):
    al,be=r['al'],r['be']; T=np.diag(al)+np.diag(be,1)+np.diag(be,-1); e,V=np.linalg.eigh(T); return e,np.abs(V[0])**2*r['n2']
def replica(d,qmin=2.6,Ecut=7.0):
    tot=0; Stot=0; n=0; rows=[]
    for r in d['res']:
        if r['q']<qmin or len(r['al'])==0: continue
        e,w=poles(r); lw=w[(e>0.5)&(e<Ecut)].sum(); tot+=lw; Stot+=r['S']; n+=1; rows.append((r['q'],lw,r['S']))
    return tot/n, tot/Stot, np.array(rows)
if __name__=='__main__':
    for Ne in (6,7):
        for n,v in (('LLL','main'),('A','twin'),('A','main'),('P3','twin'),('P3','main')):
            d=load(n,v,Ne); a,b,rows=replica(d); print(Ne,n,v,f'mean low-energy weight/N={a:.3e}  fraction of S={b:.3e}')

def replica_lowest(d,qmin=2.6,tol=0.05):
    """weight of rho_Q|0> on the lowest non-manifold state of the target sector (the reduced-momentum neutral mode)"""
    man=d['man']; num=0; den=0; n=0; rows=[]
    for r in d['res']:
        if r['q']<qmin or len(r['al'])==0: continue
        Kt=r['Kt']; El=d['lows'][Kt][1 if Kt in man else 0]-d['E0'][r['K0']]
        e,w=poles(r); lw=w[np.abs(e-El)<tol].sum(); num+=lw; den+=r['S']; n+=1; rows.append((r['q'],lw/max(r['S'],1e-30)))
    return num/n, num/den, np.array(rows)
