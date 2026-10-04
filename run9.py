import sys,json,pickle,time,gc,numpy as np
sys.path.insert(0,'/home/claude/sim')
from ana import get_cluster,System,Band,Params
from bigH import sector_H_T
from scipy.sparse.linalg import eigsh
from linsolve import hsolve_multi
from topo import pes, admissible_count, pes_counting
specs={s['name']:s for s in json.load(open('/home/claude/sim/specs2.json'))}
nm=sys.argv[1]; Ne=int(sys.argv[2]); fam=sys.argv[3] if len(sys.argv)>3 else 'C'
par=Params(**{**dict(Gcut=3.1,smap='exp'),**specs[nm].get('p',{})}); b=Band(par)
cl=get_cluster(fam,Ne); t0=time.time(); S0=System(b,cl); D=cl.D
def rot(K): x1,x2=cl.X1[K],cl.X2[K]; return int(cl.grid[(-x1-x2)%D,x1%D])
symm=all(rot(K)>=0 for K in range(D)) and fam=='C'
orb={K:(min(K,rot(K),rot(rot(K))) if symm else K) for K in range(D)}
reps=sorted(set(orb.values())); print('representatives',len(reps),flush=True)
lowr={}
for K in reps:
    H=sector_H_T(S0,K); e=np.sort(eigsh(H,k=4,which='SA',tol=1e-9,ncv=40)[0]); lowr[K]=e; del H; gc.collect()
    print(nm,'sector',K,np.round(e,4),'%.0fs'%(time.time()-t0),flush=True)
lev=sorted([(lowr[orb[K]][i],K,i) for K in range(D) for i in range(4)])
man=lev[:3]; top=man[-1][0]; gap=lev[3][0]-top; spread=top-man[0][0]
out=dict(lowr=lowr,orb=orb,man=[(int(K),int(i)) for _,K,i in man],spread=float(spread),gap=float(gap),mem=[])
print('manifold',out['man'],'gap %.4f spread %.2e'%(gap,spread),flush=True)
h=1e-3*b.g0; Bm=np.stack([b.b1,b.b2],1); gsl=[]
secs=sorted(set(K for _,K,_ in man))
for K in secs:
    m_in=[i for _,KK,i in man if KK==K]
    H0=sector_H_T(S0,K); e,V=eigsh(H0,k=max(m_in)+3,which='SA',tol=1e-11,ncv=40); o=np.argsort(e); e=e[o]; V=V[:,o]
    Gs=V[:,m_in]
    jv_all=[]
    for a,ev in ((0,np.array([1.,0])),(1,np.array([0,1.]))):
        vv=[]
        for m in (1,-1):
            tf=np.linalg.solve(Bm,m*h*ev); Sm=System(b,cl,twist=tuple(tf),v_ref=S0.v,Qs_ref=S0.Qs)
            Hm=sector_H_T(Sm,K); vv.append(Hm@Gs); del Hm,Sm; gc.collect()
        jv_all.append((vv[0]-vv[1])/(2*h))
    for j,i in enumerate(m_in):
        g=Gs[:,j]; E0=e[i]; gsl.append((K,g.copy()))
        jv=[ja[:,j]-Gs@(Gs.conj().T@ja[:,j]) for ja in jv_all]
        X=[hsolve_multi(H0,E0,Gs,v)[0] for v in jv]
        Dm=np.array([[np.vdot(jv[p],X[q]).real for q in (0,1)] for p in (0,1)])
        X2=[hsolve_multi(H0,E0,Gs,x)[0] for x in X]
        Km=np.array([[np.vdot(jv[p],X2[q]).real for q in (0,1)] for p in (0,1)])
        out['mem'].append(dict(K=K,i=i,D=Dm,Kss=Km,D_iso=float(np.trace(Dm)/2),K_iso=float(np.trace(Km)/2)))
        print(nm,'member',K,i,'D/N %.3f K/N %.4f'%(np.trace(Dm)/2/Ne,np.trace(Km)/2/Ne),'%.0fs'%(time.time()-t0),flush=True)
    del H0; gc.collect()
out['D_iso']=float(np.mean([m['D_iso'] for m in out['mem']])); out['K_iso']=float(np.mean([m['K_iso'] for m in out['mem']]))
fn=f'/home/claude/sim/res2x/n{Ne}{fam}_{nm}.pkl'; pickle.dump(out,open(fn,'wb'))
try:
    xi,ks=pes(cl,gsl,3); cnt=admissible_count(cl.Nk,3); out['pes']=dict(**pes_counting(xi,cnt),xi=xi[:cnt+80])
except Exception as ex: out['pes_err']=str(ex)
pickle.dump(out,open(fn,'wb'))
print('done',nm,Ne,out['gap'],out['spread'],out['D_iso']/Ne,out.get('pes',{}).get('largest'),'%.0fs'%(time.time()-t0))
