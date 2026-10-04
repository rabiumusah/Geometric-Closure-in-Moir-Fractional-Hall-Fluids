import sys,json,pickle,glob,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
o=pickle.load(open('analysis2.pkl','rb')); r=o['rows']; ref=o['ref']
R={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2/*.pkl')}
X={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2x/*.pkl')}
P=json.load(open('partners.json'))
B='/home/claude/build2/'
def w(name,s): open(B+name,'w').write(s)
def sci(v,d=1):
    if v is None or v!=v: return '--'
    if v==0: return '$0$'
    e=int(np.floor(np.log10(abs(v)))); m=v/10**e
    return '$%.*f\\times10^{%d}$'%(d,m,e) if d>0 else '$10^{%d}$'%e

# ---- matched-geometry table
L=[r"""\begin{table}[t]\centering\scriptsize
\caption{\textbf{Matched-geometry bands.} Partners of band A solved for $\gb_*$ and $\tau_0$ at fixed $(\psi,m^*)$ (geometry at $N_{\rm mesh}=8$, check at 10). $\sigma_\Omega$, $\sigma_{{\rm tr}g}$: relative $k$-space dispersions of Berry curvature and metric trace; $\mathcal D/N$ in meV\,\AA$^2$; $\Delta_{\rm GS}$: topological splitting. P4/P5 ($m^*=0.70,0.55$) have Bloch states identical to A (exact scaling symmetry) and are not listed.}\label{tab:matched}
\begin{tabular}{@{}lccccccccccc@{}}\toprule
Band & $\psi$ & $V$ & $w$ & $\gb_*$ (\AA$^2$) & $\tau_0$ (8/10) & $\sigma_\Omega$ & $\sigma_{{\rm tr}g}$ & $\mathcal D/N$ ($N_e{=}5,6,7,8$) & $\Delta_E$ ($N_e{=}6$) & $\Delta_{\rm GS}$ ($N_e{=}7$)\\\midrule"""]
for n,lab in (('A','A'),('P1','P1'),('P2','P2'),('P3','P3')):
    x=r[n]; p=P[n if n!='A' else 'base']
    pp=dict(psi_deg=107.7,V=20.8,w=-23.8); pp.update(p['p'])
    D='/'.join('%.1f'%x[f'D{N}'] for N in (5,6,7,8) if f'D{N}' in x)
    L.append(f"{lab} & {pp['psi_deg']:.1f} & {pp['V']:.2f} & {pp['w']:.2f} & {x['gstar']:.2f} & {p['tau0_8']:.4f}/{p['tau0_10']:.4f} & {x['sOm']:.3f} & {x['strg']:.3f} & {D} & {x['gap6']:.3f} & {sci(x['spread7'])}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_matched.tex','\n'.join(L))
# ---- selection-rule table
L=[r"""\begin{table}[t]\centering\scriptsize
\caption{\textbf{Selection rule.} Intraband weight $\mathcal D/N$ (meV\,\AA$^2$, isotropic part, averaged over the manifold) and topological splitting $\Delta_{\rm GS}$ (meV). Homogeneous bands (LLL, 1LL, twins) obey $\mathcal D=0$ and $\Delta_{\rm GS}=0$ to numerical precision; the twins' residual $\mathcal D$ stems from the weak twist dependence of the cluster-sampled $f$.}\label{tab:sel}
\begin{tabular}{@{}l ccccc ccccc@{}}\toprule
 & \multicolumn{5}{c}{$\mathcal D/N$} & \multicolumn{5}{c}{$\Delta_{\rm GS}$}\\\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Band & 4&5&6&7&8 & 4&5&6&7&8\\\midrule"""]
def cell(v,f):
    if v is None or v!=v: return '--'
    if 'e' in f: return sci(v,1 if f!='%.0e' else 0)
    return '$'+f%v+'$'
for n,lab in (('A',r"A (moir\'e)"),('P3',r"P3 (moir\'e)")):
    x=r[n]; L.append(lab+' & '+' & '.join(cell(x.get(f'D{N}'),'%.2f') for N in (4,5,6,7,8))+' & '+' & '.join(cell(x.get(f'spread{N}'),'%.1e') for N in (4,5,6,7,8))+r'\\')
for n,lab in (('A','A twin'),('P3','P3 twin')):
    x=r[n]; L.append(lab+' & '+' & '.join(cell(x.get(f'DT{N}'),'%.0e') for N in (4,5,6,7,8))+' & '+' & '.join(cell(x.get(f'spreadT{N}'),'%.0e') for N in (4,5,6,7,8))+r'\\')
for n in ('LLL','1LL'):
    x=ref[n]; L.append(n+' & '+' & '.join(cell(x[N]['D'] if N in x else None,'%.0e') for N in (4,5,6,7,8))+' & '+' & '.join(cell(x[N]['spread'] if N in x else None,'%.0e') for N in (4,5,6,7,8))+r'\\')
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_sel.tex','\n'.join(L))
# ---- topology / spectra table for all moire bands
fam=lambda n: 'matched' if n in ('A','P1','P2','P3') else ('strain' if n.startswith('s') else ('$\\theta$' if n.startswith('th') else ('$\\psi$' if n.startswith('psi') else ('$w$' if n.startswith('w') else '$V$'))))
order=['A','P1','P2','P3']+sorted([n for n in r if n.startswith('psi')])+sorted([n for n in r if n.startswith('w')])+sorted([n for n in r if n.startswith('V')])+sorted([n for n in r if n.startswith('th')])+sorted([n for n in r if n.startswith('sA')],key=lambda s: float(s[2:]))+sorted([n for n in r if n.startswith('sB')],key=lambda s: float(s[2:]))
lab={'A':'A','P1':'P1','P2':'P2','P3':'P3'}
def nice(n):
    if n in lab: return lab[n]
    if n.startswith('psi'): return '$\\psi=%s^\\circ$'%n[3:]
    if n.startswith('w'): return '$w=%s\\,w_{\\rm A}$'%n[1:]
    if n.startswith('V'): return '$V=%s\\,V_{\\rm A}$'%n[1:]
    if n.startswith('th'): return '$\\theta=%s^\\circ$'%n[2:]
    if n.startswith('sA'): return '$\\varepsilon=%s,\\ 0^\\circ$'%n[2:]
    if n.startswith('sB'): return '$\\varepsilon=%s,\\ 30^\\circ$'%n[2:]
L=[r"""\begin{table}[p]\centering\scriptsize
\caption{\textbf{Band ensemble: geometry, topological certification and intraband weight.} $C_{\rm MB}$: many-body Chern number of the manifold ($N_e=5$); PES: number of entanglement levels below the largest low-lying gap (equal to the $(1,3)$ counting in every case) and the entanglement gap ($N_e=6$, $N_A=3$); $\Delta_E$: neutral gap ($N_e=6$, meV); $\tilde{\mathcal D}=\mathcal D/(NE_C\ell^2)$ at $N_e=7$. All bands: $C=1$ single-particle band.}\label{tab:ens}
\begin{tabular}{@{}lccccccccc@{}}\toprule
Band & $\gb_*$ (\AA$^2$) & $\tau_0$ & $\tau_2$ & $\tau_4$ & $\sigma_\Omega$ & $C_{\rm MB}$ & PES & $\Delta_E$ & $10^3\tilde{\mathcal D}$\\\midrule"""]
for n in order:
    x=r[n]; p=x['pes6']
    L.append(f"{nice(n)} & {x['gstar']:.1f} & {x['tau0']:.3f} & {x['tau2']:.3f} & {x['tau4']:.3f} & {x['sOm']:.3f} & ${x['C5']:.4f}$ & {p[0] if p[2] else 'fail'}, {p[1]:.2f} & {x['gap6']:.2f} & {x[f'D7']/(x['EC']*x['ell']**2)*1e3:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_ens.tex','\n'.join(L))
# ---- stress table
def gv(n,N,k): return r[n].get(f'{k}{N}',np.nan)
L=[r"""\begin{table}[t]\centering\scriptsize
\caption{\textbf{Physical shear vertex} (manifold averages). Contact term $\langle\partial^2_\varepsilon H\rangle/N$ and paramagnetic part $2\sum|\Pi_{n0}|^2/(E_nN)$ (meV, isotropic average of the two channels); Ward: largest relative deviation of the Kubo kernel from the exact energy curvature; fidelity: fraction of $\Pi_0|0\rangle$ in the span of all five multipole states; $\chi_\Pi$: stress chirality (moir\'e / homogeneous twin).}\label{tab:stress}
\begin{tabular}{@{}llcccccc@{}}\toprule
Band & $N_e$ & contact$/N$ & param.$/N$ & Ward & fidelity & $\chi_\Pi$ (band) & $\chi_\Pi$ (twin)\\\midrule"""]
for n in ('A','P3'):
    for N in (4,5,6,7):
        if N==7:
            v=X.get(f'vert7_{n}')
            if v is None: continue
            a=v['avg']; L.append(f"{n} & 7 & {a['contact_iso']:.2f} & {a['chi_iso']:.2f} & {sci(a['ward'])} & -- & {a['chir']:.3f} & {v['twin']['avg']['chir']:.3f}\\\\")
        else:
            L.append(f"{n} & {N} & {gv(n,N,'contact'):.2f} & {gv(n,N,'chi'):.2f} & {sci(gv(n,N,'wardS'))} & {r[n][f'fidall_{N}']:.3f} & {gv(n,N,'chir'):.3f} & {gv(n,N,'chirT'):.3f}\\\\")
for N in (4,5,6):
    x=ref['LLL'][N]; L.append(f"LLL & {N} & {x['contact']:.2f} & {x['chi']:.2f} & {sci(x['ward'])} & -- & {x['chir']:.3f} & --\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_stress.tex','\n'.join(L))
print('tables ok')
# ---- clusters
from ana import FAM, Cluster
L=[r"""\begin{table}[t]\centering\scriptsize
\caption{\textbf{Clusters.} Superlattice matrices $S$ (rows $\bm L_i$ in the real-space basis dual to $\bm b_{1,2}$), sector dimension, and baseline (band A) neutral gap $\Delta_E$, manifold splitting $\Delta_{\rm GS}$ and $\mathcal D/N$. Family A: compact tori used for the ensemble; family C: $C_3$-symmetric tori; family B: elongated $3\times N_e$ tori (cluster-shape systematics; the $N_e=6$ family-B torus favours a charge-density wave). Dimensions are those of one momentum sector, $\simeq\binom{N_k}{N_e}/N_k$.}\label{tab:clusters}
\begin{tabular}{@{}cclcccc@{}}\toprule
Family & $N_e$ & $S$ & dim. & $\Delta_E$ (meV) & $\Delta_{\rm GS}$ (meV) & $\mathcal D/N$\\\midrule"""]
def mat(S): return '$\\begin{psmallmatrix}%d&%d\\\\%d&%d\\end{psmallmatrix}$'%(S[0][0],S[0][1],S[1][0],S[1][1])
from math import comb
for Ne in (4,5,6,7,8):
    S=FAM['A'][Ne]; dim=int(np.ceil(comb(3*Ne,Ne)/(3*Ne)))
    x=r['A']; L.append(f"A & {Ne} & {mat(S)} & {dim} & {x[f'gap{Ne}']:.2f} & {sci(x[f'spread{Ne}'])} & {x[f'D{Ne}']:.2f}\\\\")
dc=X.get('dichC',{})
for Ne in (4,7,9):
    S=FAM['C'][Ne]; dim=int(np.ceil(comb(3*Ne,Ne)/(3*Ne)))
    if Ne==9 and 'n9C_A' in X: g=X['n9C_A']; L.append(f"C & 9 & {mat(S)} & {dim} & {g['gap']:.2f} & {sci(g['spread'])} & {g['D_iso']/9:.2f}\\\\")
    elif ('A',Ne) in dc: L.append(f"C & {Ne} & {mat(S)} & {dim} & -- & -- & {dc[('A',Ne)]['avg']['D_iso']/Ne:.2f}\\\\")
for Ne in (5,6,7):
    k=f'sys_A|famB|{Ne}'
    if k in X: L.append(f"B & {Ne} & {mat(FAM['B'][Ne])} & {int(np.ceil(comb(3*Ne,Ne)/(3*Ne)))} & -- & -- & {X[k]['D_iso']/Ne:.2f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_clusters.tex','\n'.join(L))
# ---- uncertainty table for D (A and P3, N_e=5,6)
L=[r"""\begin{table}[t]\centering\scriptsize
\caption{\textbf{Uncertainty of the intraband weight}, by class (not combined). Entries: $\mathcal D/N$ (meV\,\AA$^2$) under each variation, with the baseline in the first row. Numerical: finite-difference step $h$ (range over $h=0.5$--$4\times10^{-3}|\bm g|$, one manifold member). Model form: boundary twists, cluster family. Parameters: dielectric constant and gate distance ($\mathcal D\propto1/\epsilon_r$ exactly; $d$ has no effect because $Qd\gg1$ on these clusters). Last row: ratio P3/A.}\label{tab:unc}
\begin{tabular}{@{}llcccc@{}}\toprule
Class & Variation & A ($N_e{=}5$) & A ($N_e{=}6$) & P3 ($N_e{=}5$) & P3 ($N_e{=}6$)\\\midrule"""]
def gd(n,what,Ne):
    k=f'sys_{n}|{what}|{Ne}'
    return X[k]['D_iso']/Ne if k in X and 'D_iso' in X[k] else np.nan
L.append('-- & baseline & '+' & '.join('%.2f'%r[n][f'D{Ne}'] for n in ('A','P3') for Ne in (5,6))+r'\\')
def hrange(n,Ne):
    k=f'sys_{n}|h|{Ne}'
    if k not in X: return '--'
    v=[a[1] for a in X[k]['vals']]; return sci((max(v)-min(v))/np.mean(v))
L.append('numerical & $h$ (rel.\\ spread) & '+' & '.join(hrange(n,Ne) for n in ('A','P3') for Ne in (5,6))+r'\\')
for wt,lab in (('twpi0','twist $(\\pi,0)$'),('tw0pi','twist $(0,\\pi)$'),('twpipi','twist $(\\pi,\\pi)$'),('famB','family B')):
    L.append(f'model form & {lab} & '+' & '.join('%.2f'%gd(n,wt,Ne) for n in ('A','P3') for Ne in (5,6))+r'\\')
for wt,lab in (('epsr7','$\\epsilon_r=7$'),('epsr14','$\\epsilon_r=14$'),('d200','$d=200$~\\AA'),('d500','$d=500$~\\AA')):
    L.append(f'parameters & {lab} & '+' & '.join('%.2f'%gd(n,wt,Ne) for n in ('A','P3') for Ne in (5,6))+r'\\')
L.append(r'\midrule')
rows_=[('baseline',None)]+[(wt,wt) for wt in ('twpi0','tw0pi','twpipi','famB','epsr7','epsr14','d200','d500')]
rat=[]
for Ne in (5,6):
    for _,wt in rows_:
        a=r['A'][f'D{Ne}'] if wt is None else gd('A',wt,Ne); b=r['P3'][f'D{Ne}'] if wt is None else gd('P3',wt,Ne)
        if a==a and b==b: rat.append(b/a)
L.append('ratio P3/A & all rows above & \\multicolumn{4}{c}{%.2f--%.2f}\\\\'%(min(rat),max(rat)) if rat else '')
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_unc.tex','\n'.join(L))
# ---- multipole-proxy table (SM Sec. S16)
from post import theta_obs
def bzM(run):
    th=[theta_obs(run['u'][u]) for u in run['u'] if u!=0]; th=[t for t in th if t]
    return np.mean([t['M24'] for t in th]),np.mean([t['f4'] for t in th]),np.mean([t['E2'] for t in th]),np.mean([t['E4'] for t in th])
L=[r"""\begin{table}[h]\centering\scriptsize
\caption{\textbf{Legacy multipole observables on identical clusters} (BZ averages over target momenta; hierarchical operators, $w_s$ = cluster-averaged $|F|$). The ``spin-2--spin-4 mixing'' $M_{24}$ is of the same size in exactly homogeneous references (twin, LLL) as in the moir\'e band.}\label{tab:legacy}
\begin{tabular}{@{}l cccc cccc@{}}\toprule
 & \multicolumn{4}{c}{$M_{24}$ (meV)} & \multicolumn{4}{c}{$f_4$}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Hamiltonian & 4&5&6&7 & 4&5&6&7\\\midrule"""]
leg={}
for nm,lab in (('moire','moir\\\'e A ($\\lambda=1$)'),('lam0','Gaussian reduction ($\\lambda=0$)'),('twin','homogeneous twin'),('LLL','LLL')):
    vals=[];fv=[]
    for Ne in (4,5,6,7):
        k=f'legacy_{nm}|{Ne}'
        if k in X: m,f4,_,_=bzM(X[k]); vals.append('%.2f'%m); fv.append('%.3f'%f4); leg[(nm,Ne)]=(m,f4)
        else: vals.append('--'); fv.append('--')
    L.append(lab+' & '+' & '.join(vals)+' & '+' & '.join(fv)+r'\\')
L.append(r"\bottomrule\end{tabular}\end{table}"); w('tab_legacy.tex','\n'.join(L))
pickle.dump(leg,open('legacy_summary.pkl','wb'))
print('tables2 all ok')
