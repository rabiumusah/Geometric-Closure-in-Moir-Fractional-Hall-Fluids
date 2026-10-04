"""Crude band-mixing indicator: interaction-weighted interband vs intraband form-factor weight on the N_e=6 cluster."""
import numpy as np,pickle,sys,json
sys.path.insert(0,'/home/claude/sim')
from ana import Params,Band,get_cluster,System
from ed import KE2
o=pickle.load(open('/home/claude/sim/analysis2.pkl','rb'))['rows']
def spec(n): return pickle.load(open(f'/home/claude/sim/res2/{n}.pkl','rb'))['spec'].get('p',{})
out={}
for n in ['A','P3','P2','psi97.7','w0.8','sA0.1','th3.5']:
    b=Band(Params(**{**dict(Gcut=3.1,smap='exp'),**spec(n)})); cl=get_cluster('A',6)
    S=System(b,cl)
    k0=b.frac2cart(np.stack([cl.X1/cl.D,cl.X2/cl.D],1))
    e,V=b.top(k0,2)
    num=0;den=0
    for iq,(u,_,m1,m2,nq,qx,qy) in enumerate(S.Qs):
        if nq<1e-9: continue
        Vq=np.tanh(nq*300)/nq
        kq=k0+np.array([qx,qy])
        e2,V2=b.top(kq,2)
        F11=np.abs(np.einsum('kg,kg->k',np.conj(V2[:,:,0]),V[:,:,0])); F21=np.abs(np.einsum('kg,kg->k',np.conj(V2[:,:,1]),V[:,:,0]))
        if iq<40 and n=='A' and abs(np.mean(F11)-S.meanF_band[iq])>1e-6: print('mismatch',iq,np.mean(F11),S.meanF_band[iq])
        num+=Vq*np.mean(F21**2); den+=Vq*np.mean(F11**2)
    out[n]=dict(ratio=float(num/den),iso=float(o[n]['iso']),gap6=float(o[n]['gap6']))
    print(n,out[n],flush=True)
json.dump(out,open('/home/claude/sim/res4/mixest.json','w'),indent=1)
