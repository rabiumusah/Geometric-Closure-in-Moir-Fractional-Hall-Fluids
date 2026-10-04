import sys,pickle,numpy as np
sys.path.insert(0,'/home/claude/sim')
from ana import *; from newcalc import P
from refband import hom_lam
from inhom import best_origin
out={}
for n in ('A','P3'):
    b=Band(P(n)); cl=get_cluster('A',6)
    Sm=System(b,cl); Sh=System(b,cl,lam_fn=hom_lam('rms',-1)); e12=best_origin(Sm,Sh,12)[0]
    Sh=System(b,cl,lam_fn=hom_lam('rms',-1)); e24=best_origin(Sm,Sh,24)[0]
    out[n]=(e12,e24); print(n,e12,e24,flush=True)
pickle.dump(out,open('/home/claude/sim/res4/etascan.pkl','wb'))
