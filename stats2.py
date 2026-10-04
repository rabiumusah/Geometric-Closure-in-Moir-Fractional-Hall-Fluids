"""Grouped statistics for the D-sigma_Omega association (SM Sec. S15)."""
import numpy as np, pickle, json, itertools
from scipy import stats
o=pickle.load(open('/home/claude/sim/analysis2.pkl','rb'))['rows']
fam={'A':'ref','P1':'matched','P2':'matched','P3':'matched','V0.9':'V','V1.1':'V','psi97.7':'psi','psi102.7':'psi','psi112.7':'psi','psi117.7':'psi',
     'w0.8':'w','w0.9':'w','w1.1':'w','w1.2':'w','th3.5':'theta','th4.3':'theta','sA0.02':'strainA','sA0.04':'strainA','sA0.06':'strainA','sA0.08':'strainA','sA0.1':'strainA',
     'sB0.04':'strainB','sB0.08':'strainB'}
F=sorted(set(fam.values())-{'ref'})
rng=np.random.default_rng(1)
def Dt(n,Ne): return o[n][f'D{Ne}']/(o[n]['EC']*o[n]['ell']**2)
def fitp(x,y): return np.polyfit(np.log(x),np.log(y),1)[0]
def partial_spearman(x,y,Z):
    rx=stats.rankdata(x); ry=stats.rankdata(y); RZ=np.column_stack([np.ones(len(x))]+[stats.rankdata(z) for z in Z])
    ex=rx-RZ@np.linalg.lstsq(RZ,rx,rcond=None)[0]; ey=ry-RZ@np.linalg.lstsq(RZ,ry,rcond=None)[0]
    return stats.pearsonr(ex,ey)[0]
out={}
for Ne in (5,6,7):
    names=[n for n in fam if f'D{Ne}' in o[n]]
    x=np.array([o[n]['sOm'] for n in names]); y=np.array([Dt(n,Ne) for n in names]); g=np.array([fam[n] for n in names])
    r=dict(n=len(names),rho=stats.spearmanr(x,y)[0],p=fitp(x,y))
    # bootstrap over bands and over families (block bootstrap; A always kept)
    bb=[];bp=[];fb=[];fp=[]
    for _ in range(4000):
        i=rng.integers(0,len(x),len(x))
        if len(set(i))>3: bb.append(stats.spearmanr(x[i],y[i])[0]); bp.append(fitp(x[i],y[i]))
        fs=rng.choice(F,len(F)); idx=[list(names).index('A')]+[k for f in fs for k in np.where(g==f)[0]]
        if len(set(idx))>3: fb.append(stats.spearmanr(x[idx],y[idx])[0]); fp.append(fitp(x[idx],y[idx]))
    q=lambda a:[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
    r.update(rho_CI_band=q(bb),p_CI_band=q(bp),rho_CI_family=q(fb),p_CI_family=q(fp))
    # leave-one-family-out
    lofo={}
    for f in F:
        m=g!=f; lofo[f]=(float(stats.spearmanr(x[m],y[m])[0]),float(fitp(x[m],y[m])))
    r['lofo']=lofo; r['lofo_rho_min']=min(v[0] for v in lofo.values()); r['lofo_p_range']=[min(v[1] for v in lofo.values()),max(v[1] for v in lofo.values())]
    # without P3
    m=np.array([n!='P3' for n in names]); r['noP3']=(float(stats.spearmanr(x[m],y[m])[0]),float(fitp(x[m],y[m])))
    # within-family (A included as family centre)
    wf={}
    for f in F:
        m=(g==f)|(g=='ref')
        if m.sum()>=3: wf[f]=(int(m.sum()),float(stats.spearmanr(x[m],y[m])[0]),float(fitp(x[m],y[m])))
    r['within']=wf
    # sign test: are within-family slopes all positive?
    k=sum(1 for v in wf.values() if v[2]>0); r['sign_test']=(k,len(wf),float(stats.binomtest(k,len(wf),0.5,alternative='greater').pvalue))
    # block permutation: permute sigma_Omega values among families' centred residual structure -> permute family-mean x across families and within-family deviations within families
    obs=r['rho']; cnt=0; nperm=20000
    fm={f:x[g==f].mean() for f in set(g)}
    for _ in range(nperm):
        xp=x.copy()
        # within-family shuffles (keeps between-family structure) -> tests within-family association
        for f in set(g):
            ii=np.where(g==f)[0]; xp[ii]=x[rng.permutation(ii)]
        # shuffle family means among families of equal treatment
        perm=dict(zip(F,rng.permutation([fm[f] for f in F])))
        for f in F:
            ii=np.where(g==f)[0]; xp[ii]=xp[ii]-fm[f]+perm[f]
        if stats.spearmanr(xp,y)[0]>=obs: cnt+=1
    r['block_perm_p']=(cnt+1)/(nperm+1)
    # partial correlations
    Z={'strg':[o[n]['strg'] for n in names],'iso':[o[n]['iso'] for n in names],'EC':[o[n]['EC'] for n in names],'tau0':[o[n]['tau0'] for n in names],'gstar':[o[n]['gstar'] for n in names]}
    r['partial']={k:float(partial_spearman(x,y,[v])) for k,v in Z.items()}
    r['partial_all']=float(partial_spearman(x,y,[Z['strg'],Z['iso'],Z['EC']]))
    # reverse: metric dispersion controlling for sigma_Omega
    r['partial_strg_given_sOm']=float(partial_spearman(np.array(Z['strg']),y,[x]))
    out[Ne]=r
json.dump(out,open('/home/claude/sim/res4/stats2.json','w'),indent=1,default=float)
for Ne,r in out.items():
    print(Ne,'rho',round(r['rho'],3),'CI band',np.round(r['rho_CI_band'],3),'CI fam',np.round(r['rho_CI_family'],3),'p',round(r['p'],2),'pCI',np.round(r['p_CI_band'],2),np.round(r['p_CI_family'],2))
    print('   lofo min rho',round(r['lofo_rho_min'],3),'p range',np.round(r['lofo_p_range'],2),' noP3',np.round(r['noP3'],3),' sign',r['sign_test'],' blockperm p',r['block_perm_p'])
    print('   within',{k:(v[0],round(v[1],2),round(v[2],2)) for k,v in r['within'].items()})
    print('   partial',{k:round(v,2) for k,v in r['partial'].items()},'all',round(r['partial_all'],2),' strg|sOm',round(r['partial_strg_given_sOm'],2))
