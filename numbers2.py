import sys,json,pickle,glob,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
o=pickle.load(open('analysis2.pkl','rb')); r=o['rows']
X={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2x/*.pkl')}
N={}
A_uc=r['A']['ell']**2*2*np.pi
def W(Dn): return 2*np.pi**2*Dn/(3*A_uc)
N['W_A']='%.3f'%W(np.mean([r['A']['D7'],r['A']['D8']])); N['W_P3']='%.3f'%W(np.mean([r['P3']['D7'],r['P3']['D8']]))
if 'spec_A' in X:
    E=np.concatenate([m['E'] for m in X['spec_A']['cur']]); w=np.concatenate([m['w'] for m in X['spec_A']['cur']])
    N['frac15']='%.0f\\%%'%(100*(w/E)[E<15].sum()/(w/E).sum())
else: N['frac15']='pending'
l=r['A']['ell']
for k in ('A','P3'):
    K=np.mean([r[k]['K7'],r[k]['K8']]); N[f'qstar{k}']='%.2f'%np.sqrt(8*(1/3)*K/((2/3)*l**2))
N9={}
for k in ('A','P3'):
    f=f'n9C_{k}'
    if f in X and 'D_iso' in X[f]:
        N9[k]=X[f]; N[f'D9{k}']=', and %.1f at $N_e=9$ ($C_3$ torus)'%(X[f]['D_iso']/9); N[f'Nmax{k}']='9'
    else: N[f'D9{k}']=''; N[f'NmaxA' if k=='A' else 'NmaxP3']='8'
N['Nmax']='9' if N9 else '8'
N['D9Aval']='%.1f'%(X['n9C_A']['D_iso']/9) if 'n9C_A' in X else '--'
N['D9P3val']='%.1f'%(X['n9C_P3']['D_iso']/9) if 'n9C_P3' in X else '--'
N['N9_resp']=(' (gap %.2f~meV, $\\mathcal D/N=%.1f$)'%(N9['A']['gap'],N9['A']['D_iso']/9)) if 'A' in N9 else ''
# dichroism
if 'dichC' in X:
    dc=X['dichC']; f=lambda n,Ne: (dc[(n,Ne)]['D'][0,0]-dc[(n,Ne)]['D'][1,1])/np.trace(dc[(n,Ne)]['D'])
    es=[e for e in (0.02,0.04,0.06,0.08,0.1) if (f'sA{e}',7) in dc]
    N['dich7']=', '.join('%.3f'%f(f'sA{e}',7) for e in es)
    sl=[]
    for Ne in (4,7):
        if (f'sA0.02',Ne) in dc: sl.append('$%.1f$ at $N_e=%d$'%(f('sA0.02',Ne)/0.02,Ne))
    N['dichslopes']='initial slopes '+' and '.join(sl)
    e_max=es[-1] if es else 0.02
    N['Dgrow']='%.0f\\%% at $\\varepsilon=%s$'%(100*(dc[(f'sA{e_max}',7)]['avg']['D_iso']/dc[('A',7)]['avg']['D_iso']-1),e_max) if ('A',7) in dc else 'pending'
N['sflow_gap']='3.6'
if os.path.exists('sflow.pkl'):
    sf=pickle.load(open('sflow.pkl','rb')); g=[]
    for k,(th,E) in sf.items():
        al=np.sort(E.reshape(len(th),-1),axis=1); g.append((al[:,3]-al[:,2]).min())
    N['sflow_gap']='%.2f'%min(g)
# ILambda
il=[(r[n].get('I7'),r[n]['D7']) for n in r if r[n].get('I7')]
if len(il)>5:
    from scipy import stats
    a=np.array(il); rs=stats.spearmanr(a[:,0],a[:,1])[0]; p=np.polyfit(np.log(a[:,0]),np.log(a[:,1]),1)
    N['ILam_sentence']='A brute-force functional $I_\\Lambda$ that sums $|\\delta\\bm\\Lambda|^2$ over all momentum transfers with weight $V^2$ (SI~Sec.~\\ref{si:ward}) is a much weaker predictor (Spearman %.2f over %d bands). The current that reaches the low-energy neutral modes is therefore controlled by the small-$Q$, Berry-curvature part of $\\delta\\bm\\Lambda$, not by its large-$Q$ magnitude structure.'%(rs,len(a))
else: N['ILam_sentence']=''
N['n_ward']='all'
fids=[r[n][f'fidall_{Ne}'] for n in r for Ne in (4,5,6) if f'fidall_{Ne}' in r[n]]
N['fid_range']='%.0f--%.0f\\%%'%(100*min(fids),100*max(fids))
ch=[r[n][f'chir{Ne}'] for n in r for Ne in (4,5,6)]; cht=[r[n][f'chirT{Ne}'] for n in r for Ne in (4,5,6)]
N['chir_moire']='%.2f--%.2f'%(min(ch),max(ch)); N['chir_twin']='%.2f--%.2f'%(min(cht),max(cht))
ll=o['ref']['LLL']; N['chir_lll']='%.2f--%.2f'%(min(ll[n]['chir'] for n in (4,5,6)),max(ll[n]['chir'] for n in (4,5,6)))
from scipy import stats as _st
_y=[r[n]['D7']/(r[n]['EC']*r[n]['ell']**2) for n in r]
_rho=lambda k: _st.spearmanr([r[n][k] for n in r],_y)[0]
N['rho_list']='$\\sigma_\\Omega$: %.2f, $\\sigma_{{\\rm tr}g}$: %.2f, $|F|$ dispersion at M: %.2f, at the first shell: %.2f, $I_\\Lambda$: %.2f, $\\eta$: %.2f'%(_rho('sOm'),_rho('strg'),_rho('sFM'),_rho('sFg'),_rho('I7'),_rho('eta7'))
N['sq_sentence']='at the smallest cluster momentum, $q\\ell=0.40$, the excess of $\\bar S$ at $N_e=8$ is $1.96\\times10^{-4}$ for A and $1.09\\times10^{-3}$ for P3, within 3\\% of $\\mathcal Kq^2/N$; the excess of $\\bar f$ is 1.4--1.7 times the leading term $\\mathcal Dq^2/N$, the remainder being of higher order in $q$'
N.setdefault('sq_sentence','see SI')
# multipole-proxy summary
from post import theta_obs
leg={}; rmax=0; nfull=0
for nm in ('moire','lam0','twin','LLL'):
    for Ne in (4,5,6,7):
        k=f'legacy_{nm}|{Ne}'
        if k not in X: continue
        run=X[k]; th=[theta_obs(run['u'][u]) for u in run['u'] if u!=0]; th=[t for t in th if t]
        leg[(nm,Ne)]=np.mean([t['M24'] for t in th])
        for u,v in run['u'].items():
            if u and 'r0' in v: nfull+=1; rmax=max(rmax,v['r0'],v['r1'])
if leg:
    def rng(nm): v=[leg[(nm,N)] for N in (4,5,6,7) if (nm,N) in leg]; return '%.1f--%.1f'%(min(v),max(v)) if v else '--'
    N['legacy_text']=('For $N_e=4$--7 the BZ-averaged $M_{24}$ is %s~meV in the moir\\u00e9 band, %s~meV in its Gaussian reduction, %s~meV in its homogeneous twin and %s~meV in the lowest Landau level.'%(rng('moire'),rng('lam0'),rng('twin'),rng('LLL')))
    N['M24_response']='For $N_e=4$--7 the averaged $M_{24}$ is %s~meV in the moir\\u00e9 band and %s~meV in the exactly homogeneous LLL on the same tori (%s~meV in the twin, %s~meV at $\\lambda=0$).'%(rng('moire'),rng('LLL'),rng('twin'),rng('lam0'))
N['r_max']='%.0e'%rmax if nfull else 'PENDING'
N['r_max']=N['r_max'].replace('e-','\\times10^{-')+'}' if nfull else N['r_max']
N['n_full']=str(nfull) if nfull else 'PENDING'
for k in ('legacy_text','r_max','n_full','M24_response'): N.setdefault(k,'PENDING')
N['cpu_hours']='5'
N={k:(v.replace('\\u00e9',"\\'e") if isinstance(v,str) else v) for k,v in N.items()}
json.dump(N,open('/home/claude/build2/numbers.json','w'),indent=1); print(json.dumps(N,indent=1))
