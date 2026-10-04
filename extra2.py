"""post-production calculations: I_Lambda for all bands, N_e=7 vertex (common GS phase), optical spectra, S(q) check,
multipole-proxy analysis with homogeneous references, twist/interaction/cluster-shape systematics of D."""
import sys,json,pickle,os,time,numpy as np
sys.path.insert(0,'/home/claude/sim')
from joblib import Parallel, delayed
OUT='/home/claude/sim/res2x'; os.makedirs(OUT,exist_ok=True)
specs={s['name']:s for s in json.load(open('/home/claude/sim/specs2.json'))}

def job(task):
    from ana import get_cluster,System,Band,Params,run_point,channel_coefs
    from refband import hom_lam
    from inhom import cluster_ILambda
    from current import manifold_current, current_response
    from stress import manifold_vertex
    kind,arg=task; fn=f'{OUT}/{kind}_{arg}.pkl'
    if os.path.exists(fn): return fn,'cached'
    t0=time.time(); res=None
    if kind=='ilam':
        sp=specs[arg]; par=Params(**{**dict(Gcut=3.1,smap='exp'),**sp.get('p',{})}); b=Band(par)
        res={Ne:cluster_ILambda(b,get_cluster('A',Ne))[0] for Ne in (5,6,7)}
    elif kind=='vert7':
        sp=specs[arg]; par=Params(**{**dict(Gcut=3.1,smap='exp'),**sp.get('p',{})})
        lf=None if sp.get('kind','moire')=='moire' else hom_lam(sp['kind'],-1)
        avg,mem=manifold_vertex(par,get_cluster('A',7),lf,dense_max=1200)
        res=dict(avg=avg,mem=[{k:v for k,v in m.items() if k not in ('E','wp','wm')} for m in mem])
        if sp.get('kind','moire')=='moire':
            avg,mem=manifold_vertex(par,get_cluster('A',7),hom_lam('rms',-1),dense_max=1200)
            res['twin']=dict(avg=avg)
    elif kind=='spec':      # optical (current) spectra and stress spectra at N_e=6 for figures
        nm=arg; sp=specs[nm.replace('_h','')]; par=Params(**{**dict(Gcut=3.1,smap='exp'),**sp.get('p',{})}); b=Band(par)
        lf=hom_lam('rms',-1) if nm.endswith('_h') else (None if sp.get('kind','moire')=='moire' else hom_lam(sp['kind'],-1))
        cl=get_cluster('A',6); avg,mem=manifold_current(b,cl,lf)
        av2,mem2=manifold_vertex(par,cl,lf,dense_max=1200)
        res=dict(cur=[dict(E=m['E'],w=m['w'],D=m['D']) for m in mem],vert=[dict(E=m.get('E'),wp=m.get('wp'),wm=m.get('wm')) for m in mem2],avg=avg,vavg=av2)
    elif kind=='sq':        # direct S(q), f(q) at smallest Q of each cluster momentum, moire vs twin
        nm,Ne=arg.split('|'); Ne=int(Ne); sp=specs[nm]; par=Params(**{**dict(Gcut=3.1,smap='exp'),**sp.get('p',{})}); b=Band(par); cl=get_cluster('A',Ne)
        from topo import low_eigs
        res={}
        for lab,lf in (('main',None),('twin',hom_lam('rms',-1))):
            S=System(b,cl,lam_fn=lf); lows=[low_eigs(S,K,1) for K in range(cl.Nk)]; o=np.argsort([l[0][0] for l in lows]); K0=int(o[0]); g=lows[K0][1][:,0]; E0=lows[K0][0][0]
            rows=[]
            for u in range(1,cl.Nk):
                sel=np.where((S.Qs[:,0]==u))[0]; iq=sel[np.argmin(S.Qs[sel,4])]
                O,Kd=S.rho_op(K0,S.Lam[iq],u); v=O@g; H=S.H(Kd)
                Sq=np.vdot(v,v).real/Ne; fq=(np.vdot(v,H@v).real-E0*np.vdot(v,v).real)/Ne
                rows.append((S.Qs[iq,4],S.Qs[iq,5],S.Qs[iq,6],Sq,fq))
            res[lab]=np.array(rows)
    elif kind=='legacy':    # multipole Theta analysis: moire lam=1, lam=0, twin, LLL
        nm,Ne=arg.split('|'); Ne=int(Ne)
        G=pickle.load(open('/home/claude/sim/res2/A.pkl','rb'))['geo']; gbar=np.array(G['g']); ell=G['ell']
        b=Band(Params(Gcut=3.1,smap='exp'))
        lf={'moire':None,'lam0':None,'twin':hom_lam('rms',-1),'LLL':hom_lam('LLL',-1)}[nm]
        r=run_point(b,'A',Ne,lam=(0.0 if nm=='lam0' else 1.0),gbar=gbar,ell=ell,nev=24,full_u=(tuple(range(3*Ne)) if (Ne<=6 and nm in ('lam0','moire')) else ()),
                    variants=(),lam_fn=lf)
        res=r
    elif kind=='sys':       # systematics of D: proper twists, interaction parameters, cluster family B
        nm,what,Ne=arg.split('|'); Ne=int(Ne); sp=specs[nm]; par=Params(**{**dict(Gcut=3.1,smap='exp'),**sp.get('p',{})}); b=Band(par)
        if what.startswith('tw'):
            th=dict(twpi0=(np.pi,0),tw0pi=(0,np.pi),twpipi=(np.pi,np.pi))[what]; cl=get_cluster('A',Ne)
            tw=np.linalg.solve(np.array(cl.S,float),np.array(th)/(2*np.pi)); avg,_=manifold_current(b,cl,None,tw0=tuple(tw))
        elif what.startswith('eps') or what.startswith('d'):
            kw={'epsr7':dict(epsr=7.),'epsr14':dict(epsr=14.),'d200':dict(d=200.),'d500':dict(d=500.)}[what]
            avg,_=manifold_current(b,get_cluster('A',Ne),None,**kw)
        elif what=='famB':
            from ana import FAM, Cluster
            cl=Cluster(FAM['B'][Ne],Ne); avg,_=manifold_current(b,cl,None)
        elif what=='h':
            cl=get_cluster('A',Ne); from current import current_response
            avg=dict(vals=[])
            S=System(b,cl); 
            from topo import low_eigs
            K=int(np.argmin([low_eigs(S,K_,1)[0][0] for K_ in range(cl.Nk)]))
            for hh in (5e-4,1e-3,2e-3,4e-3):
                r=current_response(b,cl,K,h=hh*b.g0); avg['vals'].append((hh,r['D_iso'],r['ward']))
        res=avg
    pickle.dump(res,open(fn,'wb')); return fn,round(time.time()-t0,1)

if __name__=='__main__':
    tasks=[]
    tasks+=[('vert7','A'),('vert7','P3')]
    tasks+=[('ilam',n) for n,s in specs.items() if s.get('kind','moire')=='moire']
    tasks+=[('spec',n) for n in ('A','P3','A_h','LLL')]
    tasks+=[('sq',f'{n}|{Ne}') for n in ('A','P3') for Ne in (6,8)]
    tasks+=[('legacy',f'{n}|{Ne}') for n in ('moire','lam0','twin','LLL') for Ne in (4,5,6,7)]
    tasks+=[('sys',f'{n}|{w}|{Ne}') for n in ('A','P3') for w in ('twpi0','tw0pi','twpipi','epsr7','epsr14','d200','d500','famB','h') for Ne in (5,6)]
    tasks+=[('sys',f'{n}|famB|7') for n in ('A','P3')]
    res=Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv)>1 else 2)(delayed(job)(t) for t in tasks)
    print(res)
