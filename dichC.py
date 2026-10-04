import sys,json,pickle,os,numpy as np; sys.path.insert(0,'/home/claude/sim')
from ana import get_cluster,Band,Params
from current import manifold_current
specs={s['name']:s for s in json.load(open('specs2.json'))}
out={}
fn='res2x/dichC.pkl'
if os.path.exists(fn): out=pickle.load(open(fn,'rb'))
for nm in ['A','P3','sA0.02','sA0.04','sA0.06','sA0.08','sA0.1','sB0.04','sB0.08']:
    for Ne in (4,7):
        if (nm,Ne) in out: continue
        par=Params(**{**dict(Gcut=3.1,smap='exp'),**specs[nm].get('p',{})})
        avg,mem=manifold_current(Band(par),get_cluster('C',Ne))
        Dm=np.mean([m['D'] for m in mem],axis=0)
        out[(nm,Ne)]=dict(D=Dm,avg=avg); pickle.dump(out,open(fn,'wb'))
        print(nm,Ne,'D/N %.3f dich %.4f Dxy %.4f'%(avg['D_iso']/Ne,(Dm[0,0]-Dm[1,1])/(Dm[0,0]+Dm[1,1]),2*Dm[0,1]/(Dm[0,0]+Dm[1,1])),flush=True)
