"""Production v2: moire bands vs their homogeneous (Landau-level-class) twins; LL references; topology; stress vertex."""
import numpy as np, pickle, os, sys, time, json
from joblib import Parallel, delayed
OUT='/home/claude/sim/res2'; os.makedirs(OUT,exist_ok=True)

def spectrum(S,cl,k=4):
    from topo import low_eigs
    lows=np.array([low_eigs(S,K,k)[0] for K in range(cl.Nk)])
    o=np.argsort(lows[:,0]); man=[int(x) for x in o[:3]]
    top=lows[o[2],0]
    exc=np.array([(lows[K,1] if K in man else lows[K,0])-top for K in range(cl.Nk)])
    return dict(lows=lows,man=man,E0=float(lows[o[0],0]),spread=float(top-lows[o[0],0]),gap=float(exc.min()),disp=exc)

def do_band(spec):
    fn=f"{OUT}/{spec['name']}.pkl"
    if os.path.exists(fn) and not spec.get('force'): return spec['name'],'cached'
    sys.path.insert(0,'/home/claude/sim')
    from ana import get_cluster, System, Band, Params, run_point
    from refband import hom_lam
    from inhom import best_origin, eta_mag_phase
    from topo import manifold_chern, pes, admissible_count, pes_counting, low_eigs
    from stress import manifold_vertex
    from descr import describe
    from geom import geometry
    t0=time.time()
    par=Params(**{**dict(Gcut=3.1,smap='exp'),**spec.get('p',{})})
    b=Band(par)
    kind=spec.get('kind','moire')     # 'moire' | 'LLL' | '1LL'
    out=dict(spec=spec,sizes={})
    if kind=='moire':
        G=geometry(b,8,0.45,7,8); out['geo']={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in G.items() if k not in ('y','qs','sd_beta','beta')}
        out['descr']=describe(par,24)
        f=np.arange(12)/12; F1,F2=np.meshgrid(f,f,indexing='ij'); e,_=b.top(b.frac2cart(np.stack([F1.ravel(),F2.ravel()],1)),2)
        out['geo'].update(bandwidth=float(np.ptp(e[:,0])),iso_gap=float((e[:,0]-e[:,1]).min()),Auc=float(b.Auc))
    lam_main=None if kind=='moire' else hom_lam(kind,-1)
    for Ne in spec.get('sizes',(4,5,6,7)):
        cl=get_cluster('A',Ne); r={}
        Sm=System(b,cl,lam_fn=lam_main); r['main']=spectrum(Sm,cl)
        if kind=='moire':
            Sh=System(b,cl,lam_fn=hom_lam('rms',-1)); r['twin']=spectrum(Sh,cl)
            et,z,r0,ij=best_origin(Sm,Sh,12); mg,ph=eta_mag_phase(Sm,Sh,z)
            r['eta']=dict(eta=et,mag=mg,phase=ph,origin=ij)
        if Ne in spec.get('vertex_sizes',(4,5,6)):
            avg,mem=manifold_vertex(par,cl,lam_main,dense_max=1200,want_fid=(kind=='moire' and Ne<=6),ell=(out['geo']['ell'] if kind=='moire' else None))
            for m in mem:
                for k in ('E','wp','wm'): m.pop(k,None)
            r['vertex']=dict(avg=avg,mem=mem)
            if kind=='moire':
                avg,mem=manifold_vertex(par,cl,hom_lam('rms',-1),dense_max=1200)
                for m in mem:
                    for k in ('E','wp','wm'): m.pop(k,None)
                r['vertex_twin']=dict(avg=avg,mem=mem)
        if Ne in spec.get('current_sizes',(4,5,6,7)):
            from current import manifold_current
            avg,mem=manifold_current(b,cl,lam_main,dense_max=1200)
            for m in mem:
                for k in ('E','w'): m.pop(k,None)
            r['current']=dict(avg=avg,mem=mem)
            if kind=='moire':
                avg,mem=manifold_current(b,cl,hom_lam('rms',-1),dense_max=1200)
                for m in mem:
                    for k in ('E','w'): m.pop(k,None)
                r['current_twin']=dict(avg=avg,mem=mem)
        if Ne in spec.get('chern_sizes',(5,)):
            C,per,Ks=manifold_chern(b,cl,M=5,lam_fn=lam_main,hom_sig=(None if kind=='moire' else -1)); r['chern']=dict(C=C,per=per)
        if Ne in spec.get('pes_sizes',(6,)):
            gsl=[]
            for K in r['main']['man']:
                e,v=low_eigs(Sm,K,1); gsl.append((K,v[:,0]))
            NA=3; xi,ks=pes(cl,gsl,NA); cnt=admissible_count(cl.Nk,NA); r['pes']=dict(NA=NA,**pes_counting(xi,cnt),xi=xi[:min(len(xi),cnt+80)])
        out['sizes'][Ne]=r
        pickle.dump(out,open(fn+'.part','wb'))
        print(spec['name'],Ne,'%.0fs'%(time.time()-t0),flush=True)
    out['time']=time.time()-t0
    pickle.dump(out,open(fn,'wb'))
    return spec['name'],round(out['time'],1)

if __name__=='__main__':
    specs=json.load(open(sys.argv[1]))
    nj=int(sys.argv[2]) if len(sys.argv)>2 else 2
    res=Parallel(n_jobs=nj)(delayed(do_band)(s) for s in specs)
    print(res)
