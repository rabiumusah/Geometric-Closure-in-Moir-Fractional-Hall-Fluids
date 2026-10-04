import sys,json,pickle,glob,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
from scipy import stats
KE2=14399.64
R={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2/*.pkl')}
X={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2x/*.pkl')}
P=json.load(open('partners.json'))
out={}
def ell(d): return np.sqrt(d['geo']['Auc']/(2*np.pi)) if 'geo' in d else np.sqrt(R['A']['geo']['Auc']/(2*np.pi))
def EC(d,epsr=10.): return KE2/(epsr*ell(d))
moire=[n for n,d in R.items() if d['spec'].get('kind','moire')=='moire']
rows={}
for n in moire:
    d=R[n]; s=d['sizes']; l=ell(d); ec=EC(d)
    r=dict(name=n,gstar=d['geo']['gstar'],tau0=d['geo']['tau0'],tau2=d['geo']['tau2'],tau4=d['geo']['tau4'],ell=l,EC=ec,
           sOm=d['descr']['sigma_Omega'],strg=d['descr']['sigma_trg'],TV=d['descr']['trace_violation'],
           Fg=d['descr']['F']['g'][0],sFg=d['descr']['F']['g'][1],sFM=d['descr']['F']['M (g/2)'][1],
           bw=d['geo']['bandwidth'],iso=d['geo']['iso_gap'])
    for Ne,x in s.items():
        r[f'gap{Ne}']=x['main']['gap']; r[f'spread{Ne}']=x['main']['spread']; r[f'gapT{Ne}']=x['twin']['gap']; r[f'spreadT{Ne}']=x['twin']['spread']
        r[f'eta{Ne}']=x['eta']['eta']; r[f'etamag{Ne}']=x['eta']['mag']; r[f'etaph{Ne}']=x['eta']['phase']
        r[f'E0{Ne}']=x['main']['E0']/Ne; r[f'E0T{Ne}']=x['twin']['E0']/Ne
        if 'current' in x:
            c=x['current']['avg']; r[f'D{Ne}']=c['D_iso']/Ne; r[f'K{Ne}']=c['K_iso']/Ne; r[f'WJ{Ne}']=c['WJ']/Ne; r[f'DT{Ne}']=x['current_twin']['avg']['D_iso']/Ne
            r[f'wardJ{Ne}']=c['ward']
            Dm=[m['D'] for m in x['current']['mem']]; Dav=np.mean(Dm,axis=0); r[f'dich{Ne}']=float((Dav[0,0]-Dav[1,1])/(Dav[0,0]+Dav[1,1]))
            r[f'Dxy{Ne}']=float(Dav[0,1]/(Dav[0,0]+Dav[1,1])*2)
        if 'vertex' in x:
            v=x['vertex']['avg']; r[f'chir{Ne}']=v['chir']; r[f'wardS{Ne}']=v['ward']; r[f'contact{Ne}']=v['contact_iso']; r[f'chi{Ne}']=v['chi_iso']
            r[f'meanPi{Ne}']=(v['meanPi0'],v['meanPi45']); r[f'chirT{Ne}']=x['vertex_twin']['avg']['chir']; r[f'wardST{Ne}']=x['vertex_twin']['avg']['ward']
            if 'fid0' in x['vertex']['mem'][0]:
                for k in ('2','4','2+4','all','rho'): r[f'fid{k}_{Ne}']=float(np.mean([m['fid0'][k] for m in x['vertex']['mem']]))
        if 'chern' in x: r[f'C{Ne}']=x['chern']['C']
        if 'pes' in x: r[f'pes{Ne}']=(x['pes']['count'],x['pes']['gap'],x['pes']['largest'])
    if f'ilam_{n}' in X:
        for Ne,v in X[f'ilam_{n}'].items(): r[f'I{Ne}']=v
    rows[n]=r
out['rows']=rows
# homogeneous references
ref={}
for n in ('LLL','1LL'):
    d=R[n]; ref[n]={Ne:dict(gap=x['main']['gap'],spread=x['main']['spread'],D=(x['current']['avg']['D_iso']/Ne if 'current' in x else None),
                     chir=(x['vertex']['avg']['chir'] if 'vertex' in x else None),C=(x['chern']['C'] if 'chern' in x else None),
                     pes=((x['pes']['count'],x['pes']['gap'],x['pes']['largest']) if 'pes' in x else None),
                     contact=(x['vertex']['avg']['contact_iso'] if 'vertex' in x else None),chi=(x['vertex']['avg']['chi_iso'] if 'vertex' in x else None),
                     ward=(x['vertex']['avg']['ward'] if 'vertex' in x else None)) for Ne,x in d['sizes'].items()}
out['ref']=ref
# dose-response at fixed theta (exclude th variants for raw; include all with dimensionless D)
def dimless(r,Ne): return r[f'D{Ne}']/(r['EC']*r['ell']**2)
cand=['sOm','strg','sFg','sFM','tau0','gstar','bw','iso','TV']
corr={}
for Ne in (5,6,7):
    names=[n for n in rows if f'D{Ne}' in rows[n]]
    y=np.array([dimless(rows[n],Ne) for n in names])
    for k in cand+[f'eta{Ne}',f'I{Ne}']:
        x=np.array([rows[n].get(k,np.nan) for n in names]); m=np.isfinite(x)
        if m.sum()>4:
            rs=stats.spearmanr(x[m],y[m]); pr=stats.pearsonr(np.log(np.abs(x[m])+1e-300),np.log(y[m])) if np.all(x[m]>0) else (np.nan,np.nan)
            corr[(Ne,k)]=(float(rs[0]),float(rs[1]),float(pr[0]) if pr[0]==pr[0] else np.nan,int(m.sum()))
out['corr']={f'{a}|{b}':v for (a,b),v in corr.items()}
for k,v in sorted(out['corr'].items()): print(k,np.round(v,3))
pickle.dump(out,open('analysis2.pkl','wb'))
