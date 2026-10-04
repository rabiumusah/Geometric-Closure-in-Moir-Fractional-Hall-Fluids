import numpy as np,pickle,glob,json
from scipy import stats
o=pickle.load(open('/home/claude/sim/analysis2.pkl','rb'))['rows']
REP={}
for f in glob.glob('/home/claude/sim/res3/rep/*.pkl'):
    d=pickle.load(open(f,'rb')); REP[(d['name'],d['variant'],d['Ne'])]=d['rows']
def R(rows,qmin): m=rows[:,0]>qmin; return rows[m,2].sum()/rows[m,1].sum()
out={}
for Ne in (6,7):
    names=[n for n in o if (n,'main',Ne) in REP]
    for qm in (2.4,2.5,2.6,2.7,2.8):
        y=np.array([R(REP[(n,'main',Ne)],qm) for n in names]); Dt=np.array([o[n][f'D{Ne}']/(o[n]['EC']*o[n]['ell']**2) for n in names])
        hom=[R(REP[k],qm) for k in REP if k[2]==Ne and (k[1]=='twin' or k[0]=='LLL')]
        out[f'{Ne}|{qm}']=dict(range=[float(y.min()),float(y.max())],A=float(R(REP[('A','main',Ne)],qm)),P3=float(R(REP[('P3','main',Ne)],qm)),
            hom_max=float(max(hom)),rho_eta=float(stats.spearmanr([o[n][f'eta{Ne}'] for n in names],y)[0]),rho_sOm=float(stats.spearmanr([o[n]['sOm'] for n in names],y)[0]),rho_D=float(stats.spearmanr(Dt,y)[0]))
for k,v in out.items(): print(k,{a:(np.round(b,3) if not isinstance(b,list) else np.round(b,3).tolist()) for a,b in v.items()})
json.dump(out,open('/home/claude/sim/res4/repsens.json','w'),indent=1)
