import sys,json,pickle,numpy as np; sys.path.insert(0,'/home/claude/sim')
from descr import berry, mesh_states
from band import Band,Params
P=json.load(open('partners.json'))
out={}
for nm,pp in (('A',{}),('P3',P['P3']['p'])):
    b=Band(Params(Gcut=3.1,smap='exp',**pp)); Om=berry(b,30)
    # BZ-averaged |F| along b1 direction (q in units of g0) and its k-spread
    fr,v=mesh_states(b,15); K=b.frac2cart(fr); v=v.reshape(len(K),-1)
    qs=np.linspace(0.02,1.3,40); mF=[];sF=[]
    for q in qs:
        for ang in (0.0,):
            d=q*b.g0*np.array([np.cos(ang),np.sin(ang)])
            _,v2=b.top(K+d); F=np.abs(np.sum(np.conj(v2[:,:,0])*v,axis=1)); mF.append(F.mean()); sF.append(F.std())
    out[nm]=dict(Om=Om,Omean=Om.mean(),qs=qs,mF=np.array(mF),sF=np.array(sF),l2=b.Auc/(2*np.pi),g0=b.g0)
pickle.dump(out,open('figgeo2.pkl','wb')); print({k:(v['Om'].std()/abs(v['Omean'])) for k,v in out.items()})
