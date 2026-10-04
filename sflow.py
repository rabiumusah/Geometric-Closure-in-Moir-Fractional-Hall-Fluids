import sys,json,pickle,numpy as np; sys.path.insert(0,'/home/claude/sim')
from topo import spectral_flow
from ana import get_cluster
from band import Band,Params
P=json.load(open('partners.json')); out={}
for nm,pp in (('A',{}),('P3',P['P3']['p'])):
    b=Band(Params(Gcut=3.1,smap='exp',**pp)); ths,E=spectral_flow(b,get_cluster('A',6),nflux=3,nstep=6,k=3)
    out[nm]=(ths,E); print(nm,'done',flush=True)
pickle.dump(out,open('sflow.pkl','wb'))
