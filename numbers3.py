"""Fill numbers for the neutral-spectrum / replica section and write tab_dyn.tex."""
import json,pickle,glob,numpy as np,sys
sys.path.insert(0,'/home/claude/sim')
from dynana import load, dispersion, binned
NJ='/home/claude/build2/numbers.json'; N=json.load(open(NJ))
RN=json.load(open('/home/claude/sim/res3/rep_numbers.json'))
REP={}
for f in glob.glob('/home/claude/sim/res3/rep/*.pkl'):
    d=pickle.load(open(f,'rb')); REP[(d['name'],d['variant'],d['Ne'])]=d['R']
def summ(n,v,Ne):
    d=load(n,v,Ne)
    if d is None: return None
    D=dispersion(d); q,S=binned(d,'S',0.02); return float(D[:,1].min()),float(S.max())
s={k:summ(*k,7) for k in [('A','main'),('A','twin'),('P3','main'),('P3','twin'),('LLL','main')]}
N['rot_A']='%.2f'%s[('A','main')][0]; N['rot_At']='%.2f'%s[('A','twin')][0]; N['rot_P']='%.2f'%s[('P3','main')][0]; N['rot_Pt']='%.2f'%s[('P3','twin')][0]
N['Sm_A']='%.3f'%s[('A','main')][1]; N['Sm_At']='%.3f'%s[('A','twin')][1]; N['Sm_P']='%.3f'%s[('P3','main')][1]; N['Sm_Pt']='%.3f'%s[('P3','twin')][1]; N['Sm_L']='%.3f'%s[('LLL','main')][1]
NE=7 if RN.get('nb7',0)>=20 else 6
names=sorted(set(n for (n,v,Ne) in REP if v=='main' and n!='LLL'))
lo,hi=RN[f'Rrange{NE}']; N['RB_range7']='%.2f--%.2f'%(lo,hi)
N['RB_A']='%.3f'%REP[('A','main',7)]; N['RB_P']='%.3f'%REP[('P3','main',7)]
dN=[abs(REP[(n,'main',7)]/REP[(n,'main',6)]-1) for n in names if (n,'main',7) in REP and (n,'main',6) in REP]
N['RB_dN']='%d\\%%'%int(np.ceil(100*max(dN)))
N['RBt6']='%.3f'%RN['Rtwin6'][1]; N['RBt7']='%.4f'%RN['Rtwin7'][1] if 'Rtwin7' in RN else 'n/a'
N['RBL6']='%.4f'%RN['RLLL6']; N['RBL7']='%.4f'%RN.get('RLLL7',float('nan'))
N['RB_NE']=str(NE); N['RB_nb']=str(RN[f'nb{NE}']); N['RB_exp']='%.2f'%RN[f'Rexp{NE}']; N['RB_R2']='%.2f'%RN[f'R2{NE}']
N['RB_rho_eta']='%.2f'%RN[f'rhoR{NE}|eta{NE}']; N['RB_rho_sOm']='%.2f'%RN[f'rhoR{NE}|sOm']; N['RB_rho_D']='%.2f'%RN[f'rhoRD{NE}']
N['RB_rho_I']='%.2f'%RN[f'rhoR{NE}|I{NE}']; N['RB_rho_tau']='%.2f'%RN[f'rhoR{NE}|tau0']; N['RB_rho_g']='%.2f'%RN[f'rhoR{NE}|gstar']
N['RB_rho_sFg']='%.2f'%RN[f'rhoR{NE}|sFg']
N['RB_A_seq']=', '.join('%.3f'%REP[('A','main',x)] for x in (6,7,8) if ('A','main',x) in REP)
N['RB_s0']='%.3f'%REP[('A','main',7)]; N['RB_s10']='%.3f'%REP[('sA0.1','main',7)]
N['RB_P_seq']=', '.join('%.3f'%REP[('P3','main',x)] for x in (6,7,8) if ('P3','main',x) in REP)
hv=[REP[k] for k in REP if k[1]=='twin' or k[0]=='LLL']; hv7=[REP[k] for k in REP if (k[1]=='twin' or k[0]=='LLL') and k[2]>=7]
N['RBhom']='%.3f--%.3f at $N_e=6$ (all twins and the LLL) and %.4f--%.4f at $N_e=7,8$'%(min(REP[k] for k in REP if (k[1]=='twin' or k[0]=='LLL') and k[2]==6),max(REP[k] for k in REP if (k[1]=='twin' or k[0]=='LLL') and k[2]==6),min(hv7),max(hv7))
N['RB_ratio']='%.1f'%(REP[('P3','main',7)]/REP[('A','main',7)])
json.dump(N,open(NJ,'w'),indent=1)
# table
rows=[('A','main','A'),('A','twin','A twin'),('P3','main','P3'),('P3','twin','P3 twin'),('LLL','main','LLL')]
L=[r'\begin{table}[h]\centering\small',r'\caption{\textbf{Neutral spectrum and Bragg replicas.} Lowest neutral excitation $\Delta$ (meV), peak of $\bar S$, and replica fraction $R_{\rm B}$ for $N_e=6,7,8$. Homogeneous bands (twins, LLL) carry only a cluster-dependent residual; the moir\'e values converge with size.}\label{tab:dyn}',
   r'\begin{tabular}{@{}l ccc ccc@{}}\toprule',r'band & $\Delta$ ($N_e=6/7$) & $\bar S_{\max}$ ($N_e=6/7$) & $R_{\rm B}$ ($N_e=6$) & $R_{\rm B}$ ($N_e=7$) & $R_{\rm B}$ ($N_e=8$)\\\midrule']
for n,v,lab in rows:
    a=summ(n,v,6); b=summ(n,v,7)
    f=lambda Ne: ('%.4f'%REP[(n,v,Ne)]) if (n,v,Ne) in REP else '--'
    L.append(f'{lab} & {a[0]:.2f} / {b[0]:.2f} & {a[1]:.3f} / {b[1]:.3f} & {f(6)} & {f(7)} & {f(8)}\\\\')
L+=[r'\bottomrule\end{tabular}\end{table}']
open('/home/claude/build2/tab_dyn.tex','w').write('\n'.join(L)+'\n')
for k in [k for k in N if k.startswith(('rot_','Sm_','RB'))]: print(k,N[k])
