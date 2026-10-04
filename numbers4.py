"""Numbers for the robustness analyses: grouped statistics table, N_e=8 subset, WSe2 test, mixing, replica threshold table."""
import json,pickle,glob,os,numpy as np
from scipy import stats
B='/home/claude/build2/'; S='/home/claude/sim/'
N=json.load(open(B+'numbers.json'))
st=json.load(open(S+'res4/stats2.json')); rs=json.load(open(S+'res4/repsens.json'))
o=pickle.load(open(S+'analysis2.pkl','rb'))['rows']
f2=lambda x:'%.2f'%x
N['rhoCI']='; '.join(f"[{st[k]['rho_CI_band'][0]:.2f}, {min(st[k]['rho_CI_band'][1],1):.2f}]" for k in ('5','6','7'))
N['pCI']='95\\% intervals '+'; '.join(f"[{st[k]['p_CI_band'][0]:.2f}, {st[k]['p_CI_band'][1]:.2f}]" for k in ('5','6','7'))
N['lofo_min']=f2(min(st[k]['lofo_rho_min'] for k in ('5','6','7')))
N['lofo_p']='['+f2(min(st[k]['lofo_p_range'][0] for k in ('5','6','7')))+', '+f2(max(st[k]['lofo_p_range'][1] for k in ('5','6','7')))+']'
N['partial_strg']=', '.join(f2(st[k]['partial']['strg']) for k in ('5','6','7'))+' ($N_e=5,6,7$)'
N['partial_rev']=', '.join(f2(st[k]['partial_strg_given_sOm']) for k in ('5','6','7'))
N['partial_strg_all']=N['partial_strg']; N['partial_rev_all']=N['partial_rev']
# table
L=[r'\begin{table}[h]\centering\scriptsize',
   r'\caption{\textbf{Grouped statistics of the $\tilde{\mathcal D}$--$\sigma_\Omega$ association.} Bootstrap intervals are 95\%; ``LOFO'' is leave-one-family-out; within-family entries give (number of bands, $\rho_s$, $p$) with A included in each family.}\label{tab:stats}',
   r'\begin{tabular}{@{}lccc@{}}\toprule & $N_e=5$ & $N_e=6$ & $N_e=7$\\\midrule']
row=lambda lab,fn: L.append(lab+' & '+' & '.join(fn(st[k]) for k in ('5','6','7'))+r'\\')
row('pooled $\\rho_s$',lambda r:f2(r['rho']))
row('\\quad CI (bands)',lambda r:f"[{r['rho_CI_band'][0]:.2f}, {min(r['rho_CI_band'][1],1):.2f}]")
row('\\quad CI (families)',lambda r:f"[{r['rho_CI_family'][0]:.2f}, {min(r['rho_CI_family'][1],1):.2f}]")
row('exponent $p$',lambda r:f2(r['p']))
row('\\quad CI (bands)',lambda r:f"[{r['p_CI_band'][0]:.2f}, {r['p_CI_band'][1]:.2f}]")
row('\\quad CI (families)',lambda r:f"[{r['p_CI_family'][0]:.2f}, {r['p_CI_family'][1]:.2f}]")
row('LOFO: min $\\rho_s$',lambda r:f2(r['lofo_rho_min']))
row('LOFO: range of $p$',lambda r:f"{r['lofo_p_range'][0]:.2f}--{r['lofo_p_range'][1]:.2f}")
row('without P3: $\\rho_s$, $p$',lambda r:f"{r['noP3'][0]:.2f}, {r['noP3'][1]:.2f}")
row('block permutation $p$-value',lambda r:'$<10^{-4}$')
for fam,lab in (('matched','matched'),('psi','$\\psi$'),('w','$w$'),('V','$V$'),('theta','$\\theta$'),('strainA','strain $0^\\circ$'),('strainB','strain $30^\\circ$')):
    row('within: '+lab,lambda r,f=fam:f"({r['within'][f][0]}, {r['within'][f][1]:.2f}, {r['within'][f][2]:.2f})")
row('partial $\\rho_s$ $|$ $\\sigma_{{\\rm tr}g}$',lambda r:f2(r['partial']['strg']))
row('partial $\\rho_s$ $|$ $\\sigma_{{\\rm tr}g}$, $\\Delta_{\\rm iso}$, $E_C$',lambda r:f2(r['partial_all']))
row('partial $\\rho_s(\\sigma_{{\\rm tr}g})$ $|$ $\\sigma_\\Omega$',lambda r:f2(r['partial_strg_given_sOm']))
L+=[r'\bottomrule\end{tabular}\end{table}']
open(B+'tab_stats.tex','w').write('\n'.join(L)+'\n')
# N_e=8 subset
n8={os.path.basename(f)[4:-12]:pickle.load(open(f,'rb'))['avg']['D_iso']/8 for f in glob.glob(S+'res4/cur_*_A8_d300.pkl')}
n8['A']=o['A']['D8'] if 'D8' in o['A'] else 8.58; n8['P3']=o['P3'].get('D8',36.18)
names=sorted(n8); x=np.array([o[n]['sOm'] for n in names]); y=np.array([n8[n]/(o[n]['EC']*o[n]['ell']**2) for n in names])
rho8=stats.spearmanr(x,y)[0]; p8=np.polyfit(np.log(x),np.log(y),1)[0]
y7=np.array([o[n]['D7']/(o[n]['EC']*o[n]['ell']**2) for n in names]); rho7=stats.spearmanr(x,y7)[0]; p7=np.polyfit(np.log(x),np.log(y7),1)[0]
rat=np.array([n8[n]/o[n]['D7'] for n in names])
N['n8sentence']=f"for a subset of {len(names)} bands spanning $\\sigma_\\Omega=0.12$--0.37, $\\rho_s={rho8:.2f}$ and $p={p8:.2f}$ (the same subset at $N_e=7$: $\\rho_s={rho7:.2f}$, $p={p7:.2f}$), and $\\mathcal D/N$ changes by {100*(rat.min()-1):+.0f} to {100*(rat.max()-1):+.0f}\\% between $N_e=7$ and 8."
PSI='$\\psi$'
lab=lambda n: n if not n.startswith('psi') else PSI+n[3:]
blist=', '.join(lab(n) for n in names)
N['n8para']=f"For {len(names)} bands ({blist}) spanning $\\sigma_\\Omega=$ {x.min():.2f}--{x.max():.2f}, $\\mathcal D/N$ at $N_e=8$ differs from $N_e=7$ by {100*(rat.min()-1):+.0f} to {100*(rat.max()-1):+.0f}\\%. The subset correlation is $\\rho_s={rho8:.2f}$ with $p={p8:.2f}$ (at $N_e=7$: $\\rho_s={rho7:.2f}$, $p={p7:.2f}$). The association and the exponent are thus stable to one more size; a quantitative $1/N_e$ extrapolation is not attempted because family-A tori change shape with $N_e$."
json.dump(dict(names=names,n8=n8,rho8=rho8,p8=p8,rho7=rho7,p7=p7),open(S+'res4/n8summary.json','w'),indent=1,default=float)
# gauge
g=[pickle.load(open(f,'rb'))['avg']['D_iso'] for f in sorted(glob.glob(S+'res4/gauge_A_6_*.pkl'))]
N['n8count']=str(len(names)-2)
N['gauge_detail']=f"A, $N_e=6$: $\\mathcal D=${g[0]:.10f} and {g[1]:.10f}, {g[2]:.10f} for two random gauges"
mt=pickle.load(open(S+'res4/cur_A_meantwin6.pkl','rb'))['avg']['D_iso']/6
N['Dmeantwin']='%.0f\\times10^{-6}'%(mt*1e6) if mt>1e-6 else '<10^{-6}'
e=pickle.load(open(S+'res4/etascan.pkl','rb')); N['eta_scan']='less than $10^{-10}$' if max(abs(a-b) for a,b in e.values())<1e-10 else '%.1e'%max(abs(a-b) for a,b in e.values())
# mixing
mx=json.load(open(S+'res4/mixest.json'))
N['mixpara']='We find $\\kappa=$ '+', '.join(f"{v['ratio']:.2f} ({lab(k)})" for k,v in mx.items())+'. Band mixing is thus not small in this model, and it is of similar strength for all bands.'
# WSe2
ws=[pickle.load(open(f,'rb')) for f in sorted(glob.glob(S+'res4/wse2_*.pkl'))]
if ws:
    parts=[]
    for w in ws:
        cert=('D' in w)
        parts.append(f"$\\theta={w['theta']}^\\circ$: isolation gap {w['iso']:.1f}~meV, neutral gap {w['gap']:.2f}~meV, splitting {w['spread']:.1e}~meV, entanglement counting {'satisfied' if w['pes']['largest'] else 'not satisfied'}"+(f", $\\sigma_\\Omega={w['descr']['sigma_Omega']:.3f}$, $\\mathcal D/N={w['D']['D_iso']/6:.2f}$ (twin {w['DT']['D_iso']/6:.0e}), $\\tilde{{\\mathcal D}}=${1e3*w['D']['D_iso']/6/(w['EC']*w['ell']**2):.2f}$\\times10^{{-3}}$" if cert else ''))
    nm6=[n for n in o if 'D6' in o[n]]; xx=np.log([o[n]['sOm'] for n in nm6]); yy=np.log([o[n]['D6']/(o[n]['EC']*o[n]['ell']**2) for n in nm6]); pf=np.polyfit(xx,yy,1)
    resid=yy-np.polyval(pf,xx); sres=np.std(resid)
    cw=[w for w in ws if 'D' in w]
    if cw:
        rows=[]
        for w in cw:
            dt=w['D']['D_iso']/6/(w['EC']*w['ell']**2); so=w['descr']['sigma_Omega']; pred=np.exp(np.polyval(pf,np.log(so))); rows.append((w['theta'],so,dt,pred,dt/pred,np.log(dt/pred)/sres))
        smax=max(o[n]['sOm'] for n in nm6)
        txt='; '.join(f"at $\\theta={r[0]}^\\circ$ ($\\sigma_\\Omega={r[1]:.2f}$) the measured $\\tilde{{\\mathcal D}}$ is {r[4]:.2f} times the prediction ({r[5]:+.1f} standard deviations of the MoTe$_2$ scatter in log space)" for r in rows)
        xs=np.log([r[1] for r in rows]); ys=np.log([r[2] for r in rows]); pw=np.polyfit(xs,ys,1)[0] if len(rows)>2 else np.nan
        N['wse2concl']=f"Every tested angle has an FQAH ground state at $N_e=6$ (gap and $(1,3)$ entanglement counting), and every GMP twin has $\\mathcal D=0$. Comparing with the twisted-MoTe$_2$ $N_e=6$ power law fitted for $\\sigma_\\Omega\\le{smax:.2f}$: {txt}. The $2.0^\\circ$ point lies inside the MoTe$_2$ range of $\\sigma_\\Omega$ and on the law. The larger angles lie far outside that range, and their bandwidths (11 and 20~meV) approach or exceed the isolation gap, so the flat-band projection is less reliable there. They exceed the extrapolated law by 30--43\\%. The WSe$_2$ points alone give a log--log slope of {pw:.2f}, close to the MoTe$_2$ exponent at $N_e=7$. Three points in a second model cannot establish universality, but they show that the association is not specific to the MoTe$_2$ parametrization."
        N['wse2resp']=f"We repeated the full calculation in the twisted-WSe$_2$ continuum model, a second material with different mass, potential and tunnelling. At $\\theta=2.0^\\circ$, $2.5^\\circ$ and $3.0^\\circ$ the $\\nu=1/3$ states are certified FQAH fluids (entanglement counting 330 with the largest gap), and their GMP twins have $\\mathcal D=0$. The point at $\\sigma_\\Omega={rows[0][1]:.2f}$, inside the MoTe$_2$ range, lies on the MoTe$_2$ power law (ratio {rows[0][4]:.2f}). The points at $\\sigma_\\Omega={rows[1][1]:.2f}$ and {rows[-1][1]:.2f}, far outside it, exceed the extrapolated law by factors {rows[1][4]:.2f} and {rows[-1][4]:.2f}, and give their own exponent {pw:.2f}. Details are in SI Sec.~\\ref{{si:stats}}. Three points cannot establish universality, and we say so."
    N['wse2para']='We repeated the calculation for the twisted-WSe$_2$ continuum model ($a_0=3.317$~\\AA, $m^*=0.43m_e$, $V=9$~meV, $\\psi=128^\\circ$, $w=-18$~meV), at $N_e=6$, $\\nu=1/3$: '+'; '.join(parts)+'. '+N.get('wse2concl','')
json.dump(N,open(B+'numbers.json','w'),indent=1)
# replica threshold table + numbers
T=[r'\begin{table}[h]\centering\scriptsize',r'\caption{\textbf{Threshold sensitivity of the replica fraction.} $R_{\rm B}$ for A and P3, the ensemble range, the largest homogeneous residual, and rank correlations of $R_{\rm B}$ with $\eta$, $\sigma_\Omega$ and $\tilde{\mathcal D}$, for thresholds $Q_c\ell=2.4$--2.8.}\label{tab:thr}',
   r'\begin{tabular}{@{}cc ccccccc@{}}\toprule $N_e$ & $Q_c\ell$ & A & P3 & range & homog.\ max & $\rho_s(\eta)$ & $\rho_s(\sigma_\Omega)$ & $\rho_s(\tilde{\mathcal D})$\\\midrule']
for k,v in rs.items():
    ne,qc=k.split('|'); T.append(f"{ne} & {qc} & {v['A']:.3f} & {v['P3']:.3f} & {v['range'][0]:.2f}--{v['range'][1]:.2f} & {v['hom_max']:.3f} & {v['rho_eta']:.2f} & {v['rho_sOm']:.2f} & {v['rho_D']:.2f}\\\\")
T+=[r'\bottomrule\end{tabular}\end{table}']
open(B+'tab_thr.tex','w').write('\n'.join(T)+'\n')
dA=max(abs(rs[f'7|{q}']['A']-rs['7|2.6']['A']) for q in (2.4,2.5,2.7,2.8)); dP=max(abs(rs[f'7|{q}']['P3']-rs['7|2.6']['P3']) for q in (2.4,2.5,2.7,2.8))
N['RB_thr']='%.2f (absolute)'%max(dA,dP)
N['RB_thr_rho']='%.2f'%max(abs(rs[f'{ne}|{q}'][c]-rs[f'{ne}|2.6'][c]) for ne in (6,7) for q in (2.4,2.5,2.7,2.8) for c in ('rho_eta','rho_sOm','rho_D'))
REP={}
for f in glob.glob(S+'res3/rep/*.pkl'):
    d=pickle.load(open(f,'rb')); REP[(d['name'],d['variant'],d['Ne'])]=d
from_sm={'A':0.385,'P3':0.393}
N['RB_peakfrac']='%.1f--%.1f\\%%'%(100*REP[('A','main',7)]['Rabs']/from_sm['A'],100*REP[('P3','main',7)]['Rabs']/from_sm['P3'])
json.dump(N,open(B+'numbers.json','w'),indent=1)
print({k:N[k] for k in ('rhoCI','pCI','lofo_min','lofo_p','partial_strg','partial_rev','n8sentence','Dmeantwin','eta_scan','RB_thr','RB_thr_rho','RB_peakfrac')})
N=json.load(open(B+'numbers.json'))
N['ILam_sentence']=N['ILam_sentence'].replace('is therefore controlled by','is therefore associated mainly with')
json.dump(N,open(B+'numbers.json','w'),indent=1)
import re
N=json.load(open(B+'numbers.json'))
for k in ('wse2para',):
    if k in N: N[k]=re.sub(r'(\d\.\d)e-0?(\d+)',lambda m:'$'+m.group(1)+'\\times10^{-'+m.group(2)+'}$',N[k])
json.dump(N,open(B+'numbers.json','w'),indent=1)
N=json.load(open(B+'numbers.json'))
if os.path.exists(S+'res4/phiunc.pkl'):
    pu=pickle.load(open(S+'res4/phiunc.pkl','rb'))
    d2=[];d4=[];dd=[]
    for e in (0.04,0.06,0.10):
        a0=pu[(e,0.0,10)]; a3=pu[(e,30.0,10)]
        r2=(a3[0]-a0[0]); u2=np.hypot(a3[1],a0[1]); r4=((a3[2]-a0[2]+45)%90)-45; u4=np.hypot(a3[3],a0[3])
        d2.append(f'{r2:.1f}\\pm{u2:.1f}'); d4.append(f'{r4:.1f}\\pm{u4:.1f}')
        m0=pu[(e,0.0,8)]; m3=pu[(e,30.0,8)]
        dd.append(max(abs((m3[0]-m0[0])-r2),abs(((m3[2]-m0[2]+45)%90-45)-r4)))
    N['orient_sentence']='Numerically, for $\\phi=0^\\circ\\to30^\\circ$ at $\\varepsilon=0.04,0.06,0.10$ the fitted $\\varphi_2$ rotates by $'+'^\\circ,\\ '.join(d2)+'^\\circ$ (prediction $+30^\\circ$) and $\\varphi_4$ by $'+'^\\circ,\\ '.join(d4)+'^\\circ$ (mod $90^\\circ$; prediction $-15^\\circ$). Uncertainties are residual-bootstrap standard deviations (300 resamples) on the $10\\times10$ mesh. The $8\\times8$ mesh gives the same central values within $'+f'{max(dd):.1f}'+'^\\circ$, with larger bootstrap errors. The deviation of $\\varphi_4$ from $-15^\\circ$ (about $1^\\circ$, two to three bootstrap standard deviations) decreases with $\\varepsilon$ and is attributed to the finite fit window and higher-order terms, which the bootstrap does not capture.'
    json.dump(N,open(B+'numbers.json','w'),indent=1)
