"""Main-text figure: blind magnetoroton, bright Bragg replicas; replica law; two independent faces of inhomogeneity."""
import sys,json,pickle,glob,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
from scipy import stats
from dynana import load
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm","lines.linewidth":1.2,
                     "xtick.major.width":0.5,"ytick.major.width":0.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]; INK="#3a3a38"; MUTED="#8a8984"
O='/home/claude/build2/fig/'
rows=pickle.load(open('/home/claude/sim/analysis2.pkl','rb'))['rows']
REP={}
for f in glob.glob('/home/claude/sim/res3/rep/*.pkl'):
    d=pickle.load(open(f,'rb')); REP[(d['name'],d['variant'],d['Ne'])]=d
names=sorted(rows)
def Rv(n,v,Ne): d=REP.get((n,v,Ne)); return None if d is None else d['R']
N={}
# ---------- numbers
for Ne in (6,7,8):
    m=[n for n in names if Rv(n,'main',Ne) is not None]
    if not m: continue
    R=np.array([Rv(n,'main',Ne) for n in m]); N[f'nb{Ne}']=len(m); N[f'Rrange{Ne}']=(float(R.min()),float(R.max()))
    tw=[Rv(n,'twin',Ne) for n in names if Rv(n,'twin',Ne) is not None]
    if tw: N[f'Rtwin{Ne}']=(float(min(tw)),float(max(tw)))
    if Rv('LLL','main',Ne) is not None: N[f'RLLL{Ne}']=Rv('LLL','main',Ne)
    if len(m)>5:
        Dt=np.array([rows[n][f'D{min(Ne,7)}']/(rows[n]['EC']*rows[n]['ell']**2) for n in m])
        for k in ['sOm','strg','tau0','gstar','TV','sFg','sFM',f'eta{min(Ne,7)}',f'I{min(Ne,7)}']:
            x=np.array([rows[n].get(k,np.nan) for n in m])
            N[f'rhoR{Ne}|{k}']=float(stats.spearmanr(x,R)[0]); N[f'rhoD{Ne}|{k}']=float(stats.spearmanr(x,Dt)[0])
        N[f'rhoRD{Ne}']=float(stats.spearmanr(R,Dt)[0])
        e=np.array([rows[n][f'eta{min(Ne,7)}'] for n in m]); p=np.polyfit(np.log(e),np.log(R),1)
        N[f'Rexp{Ne}']=float(p[0]); N[f'R2{Ne}']=float(stats.pearsonr(np.log(e),np.log(R))[0]**2)
for k in ('A','P3'):
    for Ne in (6,7,8):
        if Rv(k,'main',Ne) is not None: N[f'R_{k}_{Ne}']=Rv(k,'main',Ne)
        if Rv(k,'twin',Ne) is not None: N[f'RT_{k}_{Ne}']=Rv(k,'twin',Ne)
json.dump(N,open('/home/claude/sim/res3/rep_numbers.json','w'),indent=1)
for k,v in N.items(): print(k,v)
NE=max(Ne for Ne in (6,7) if f'nb{Ne}' in N and N[f'nb{Ne}']>=20) if any(f'nb{x}' in N and N[f'nb{x}']>=20 for x in (6,7)) else 6
print('ensemble Ne',NE)
# ---------- figure
_PC={}
def poles(r):
    if id(r) in _PC: return _PC[id(r)]
    al,be=r['al'],r['be']; T=np.diag(al)+np.diag(be,1)+np.diag(be,-1); e,V=np.linalg.eigh(T); _PC[id(r)]=(e,np.abs(V[0])**2*r['n2']); return _PC[id(r)]
def sqw_map(d,qgrid,w,eta=0.3):
    M=np.zeros((len(qgrid)-1,len(w))); cnt=np.zeros(len(qgrid)-1)
    for r in d['res']:
        if len(r['al'])==0: continue
        i=np.searchsorted(qgrid,r['q'])-1
        if i<0 or i>=len(qgrid)-1: continue
        e,wt=poles(r); m=e>0.5
        M[i]+=np.sum(wt[m][:,None]*eta/np.pi/((w[None,:]-e[m][:,None])**2+eta**2),0); cnt[i]+=1
    M[cnt>0]/=cnt[cnt>0][:,None]; M[cnt==0]=np.nan; return M
fig=plt.figure(figsize=(7.1,4.6)); gs=fig.add_gridspec(2,3,width_ratios=[1,1,0.04],height_ratios=[1,1],wspace=0.18,hspace=0.42)
w=np.linspace(0,14,420); qg=np.arange(0.3,4.65,0.12); qc=0.5*(qg[1:]+qg[:-1]); vm=None
for i,(key,lab) in enumerate(((('A','main'),'(a) moiré band A'),(('A','twin'),'(b) homogeneous twin'))):
    ax=fig.add_subplot(gs[0,i]); M=sqw_map(load(*key,7),qg,w)
    if vm is None: vm=np.nanpercentile(M,99.3)
    from matplotlib.colors import PowerNorm
    im=ax.pcolormesh(qc,w,M.T,cmap='magma_r',norm=PowerNorm(0.35,vmin=0,vmax=vm),shading='nearest',rasterized=True)
    ax.axvline(2.6,color=C[2],lw=0.6,ls=':'); ax.set_xlim(0.3,4.6); ax.set_ylim(0,14)
    ax.set_title(lab,fontsize=7.5,color=INK,loc='left'); ax.set_xlabel('$|\\mathbf{Q}|\\ell$')
    if i==0:
        ax.set_ylabel('$\\hbar\\omega$ (meV)'); ax.annotate('Bragg replicas',xy=(3.6,4.9),xytext=(2.75,9.5),fontsize=6.6,color=INK,arrowprops=dict(arrowstyle='->',lw=0.5,color=INK))
        ax.annotate('roton',xy=(1.45,4.8),xytext=(0.45,1.6),fontsize=6.6,color=INK,arrowprops=dict(arrowstyle='->',lw=0.5,color=INK))
    else: ax.set_yticklabels([])
cax=fig.add_subplot(gs[0,2]); cb=fig.colorbar(im,cax=cax); cb.set_label('$S(\\mathbf{Q},\\omega)$ (meV$^{-1}$)',fontsize=6.5); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.5)
gb_=gs[1,:].subgridspec(1,2,wspace=0.32,width_ratios=[1,1.1])
ax=fig.add_subplot(gb_[0,0])
for Ne,mk,al in ((6,'o',0.35),(7,'o',1.0)):
    m=[n for n in names if Rv(n,'main',Ne) is not None]
    if not m: continue
    e=np.array([rows[n][f'eta{Ne}'] for n in m]); R=np.array([Rv(n,'main',Ne) for n in m])
    ax.plot(e,R,mk,ms=3 if Ne==7 else 2.4,color=C[0],alpha=al,mew=0,label=f'moiré, $N_e={Ne}$')
    for k,c in (('A',C[0]),('P3',C[1])):
        if k in m: ax.plot(rows[k][f'eta{Ne}'],Rv(k,'main',Ne),'o',ms=4.5 if Ne==7 else 3,mfc='none',mec=c,mew=0.9)
    if Ne==NE:
        p=np.polyfit(np.log(e),np.log(R),1); xx=np.linspace(e.min(),e.max(),20); ax.plot(xx,np.exp(np.polyval(p,np.log(xx))),color=INK,lw=0.7,ls='--',label=f'$\\propto\\eta^{{{p[0]:.2f}}}$')
for Ne,al in ((6,0.4),(7,1.0)):
    tw=[(0.0,Rv(n,'twin',Ne)) for n in names if Rv(n,'twin',Ne) is not None]+([(0.0,Rv('LLL','main',Ne))] if Rv('LLL','main',Ne) is not None else [])
    if tw:
        tw=np.array(tw); ax.plot(tw[:,0]+(0.004 if Ne==7 else 0),tw[:,1],'s',ms=2.2,color=MUTED,alpha=al,mew=0,label=f'twins + LLL, $N_e={Ne}$')
ax.text(0.97,0.04,'rings: A (blue), P3 (orange)',transform=ax.transAxes,ha='right',fontsize=6.6,color=INK); ax.set_xlim(-0.015,0.39); ax.set_ylim(0,0.24)
ax.set_xlabel('distance to GMP class $\\eta$'); ax.set_ylabel('replica fraction $R_{\\rm B}$'); ax.legend(fontsize=6.6,loc='center left',bbox_to_anchor=(0.02,0.6),handletextpad=0.2)
ax.set_title('(c)',fontsize=7.5,color=INK,loc='left')
ax=fig.add_subplot(gb_[0,1]); keys=[('sOm','$\\sigma_\\Omega$'),('strg','$\\sigma_{{\\rm tr}g}$'),('tau0','$\\tau_0$'),('gstar','$\\bar g_*$'),('sFg','$\\sigma_{|F|}(G)$'),(f'eta{min(NE,7)}','$\\eta$'),(f'I{min(NE,7)}','$I_\\Lambda$')]
x=np.arange(len(keys)); a=[N.get(f'rhoD{NE}|{k}',np.nan) for k,_ in keys]; b=[N.get(f'rhoR{NE}|{k}',np.nan) for k,_ in keys]
ax.bar(x-0.19,a,0.36,color=C[1],label='$\\tilde{\\mathcal{D}}$ (terahertz)'); ax.bar(x+0.19,b,0.36,color=C[0],label='$R_{\\rm B}$ (Bragg replicas)')
ax.axhline(0,color=INK,lw=0.5); ax.set_xticks(x); ax.set_xticklabels([l for _,l in keys]); ax.set_ylim(-1,1.05); ax.set_ylabel(f'Spearman $\\rho_s$ ($N_e={NE}$, {N[f"nb{NE}"]} bands)')
ax.legend(fontsize=6.6,loc='lower left'); ax.set_title('(d)',fontsize=7.5,color=INK,loc='left'); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x',length=0)
fig.savefig(O+'replica.pdf',bbox_inches='tight',dpi=300); plt.close(); print('fig ok')
