import numpy as np,pickle,sys
sys.path.insert(0,'/home/claude/sim')
from ana import Params,Band,System,get_cluster
from prod2 import spectrum
from newcalc import P
out={}
for n in ('A','P3'):
    for Ne in (4,7):
        b=Band(P(n)); cl=get_cluster('C',Ne); S=System(b,cl); r=spectrum(S,cl); out[(n,Ne)]=dict(gap=r['gap'],spread=r['spread'],dim=len(cl.sector(0)))
        print(n,Ne,out[(n,Ne)],flush=True)
pickle.dump(out,open('/home/claude/sim/res4/cspec.pkl','wb'))
