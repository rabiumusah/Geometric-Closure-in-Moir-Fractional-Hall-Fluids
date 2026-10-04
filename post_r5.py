"""Non-Abelian Chern numbers into numbers.json and Table 1 (tab_ens) C_MB column."""
import pickle,glob,json,os,re
B='/home/claude/build2/'; S='/home/claude/sim/'
N=json.load(open(B+'numbers.json'))
ens={os.path.basename(f).split('_')[1]:pickle.load(open(f,'rb')) for f in glob.glob(S+'res4/chernNA_*_5_6.pkl')}
assert len(ens)==23, len(ens)
assert all(abs(r['C']+1)<1e-9 for r in ens.values())
N['NA_Fmax']='%.2f'%max(r['Fmax'] for r in ens.values()); N['NA_smin']='%.2f'%min(r['smin'] for r in ens.values())
conv=[]
for n in ('A','P3'):
    for M in (5,8,10):
        r=pickle.load(open(S+f'res4/chernNA_{n}_5_{M}.pkl','rb')); conv.append((n,M,r['C'],r['Fmax'],r['smin']))
N['NA_conv']='; '.join(f"{n}, $M={M}$: $C_{{\\rm MB}}={C:.0f}$, $\\max|F|={F:.2f}$, $\\sigma_{{\\min}}={s:.2f}$" for n,M,C,F,s in conv)
json.dump(N,open(B+'numbers.json','w'),indent=1)
t=open(B+'tab_ens.tex').read()
t=re.sub(r'\$-0\.99\d\d\$','$-1$',t)
open(B+'tab_ens.tex','w').write(t)
print(N['NA_Fmax'],N['NA_smin']); print(N['NA_conv'])
# --- caption edits applied to regenerated tables (idempotent)
def rep(fn,a,b):
    t=open(B+fn).read()
    if b not in t:
        assert a in t,(fn,a[:60]); t=t.replace(a,b); open(B+fn,'w').write(t)
rep('tab_ens.tex',"\\caption{\\textbf{Band ensemble: geometry, topological certification and intraband weight.} $C_{\\rm MB}$: many-body Chern number of the manifold ($N_e=5$);",
    "\\caption{\\textbf{Band ensemble: geometry, topological certification and intraband weight.} $\\gb_*=\\sqrt{\\det\\gb}$: averaged metric; $\\tau_{0,2,4}$: isotropic and anisotropic quartic invariants (Eq.~\\eqref{eq:cum}; $\\tau_{2,4}$ vanish by $C_3$ symmetry up to the estimator floor for unstrained bands); $\\sigma_\\Omega$: relative Berry-curvature dispersion; $C_{\\rm MB}$: non-Abelian many-body Chern number of the three-fold manifold ($N_e=5$, $6\\times6$ sewn twist grid; exact integer, branch-safe in every case);")
rep('tab_legacy.tex',"\\textbf{Legacy multipole observables on identical clusters}","\\textbf{Multipole observables on identical clusters}")
rep('tab_unc.tex',"$d$ has no effect because $Qd\\gg1$ on these clusters)","$d\\ge200$~\\AA\\ has no effect because $Qd\\gg1$ on these clusters, while strong screening, $d=50$~\\AA, reweights the momentum transfers and changes $\\mathcal D$ by $\\le6\\%$)")
u=open(B+'tab_unc.tex').read()
if '(strong screening)' not in u:
    a=pickle.load(open(S+'res4/cur_A_A6_d50.pkl','rb'))['avg']['D_iso']/6; p=pickle.load(open(S+'res4/cur_P3_A6_d50.pkl','rb'))['avg']['D_iso']/6
    row="parameters & $d=500$~\\AA & 11.35 & 11.11 & 35.59 & 34.80\\\\"
    assert row in u
    u=u.replace(row,row+f"\nparameters & $d=50$~\\AA\\ (strong screening) & -- & {a:.2f} & -- & {p:.2f}\\\\"); open(B+'tab_unc.tex','w').write(u)
print('post_r5 tables done')
rep('tab_sel.tex',"Band & 4&5&6&7&8 & 4&5&6&7&8\\\\","Band\\quad $N_e=$ & 4 & 5 & 6 & 7 & 8 & 4 & 5 & 6 & 7 & 8\\\\")
print('post_r5 tab_sel done')
