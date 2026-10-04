import sys,json,pickle,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
from dynana import load, dispersion, binned
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7.5,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm","lines.linewidth":1.2,
                     "xtick.major.width":0.5,"ytick.major.width":0.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]; INK="#3a3a38"; MUTED="#8a8984"
O='/home/claude/build2/fig/'
NE=int(sys.argv[1]) if len(sys.argv)>1 else 8
BANDS=[('A','main','A',C[0],'-'),('A','twin','A twin',C[0],':'),('P3','main','P3',C[1],'-'),('P3','twin','P3 twin',C[1],':'),('LLL','main','LLL',MUTED,'--')]
D={(n,v):load(n,v,NE) for n,v,*_ in BANDS}
_PC={}
def poles(r):
    if id(r) in _PC: return _PC[id(r)]
    al,be=r['al'],r['be']; T=np.diag(al)+np.diag(be,1)+np.diag(be,-1); e,V=np.linalg.eigh(T); w=np.abs(V[0])**2*r['n2']
    _PC[id(r)]=(e,w); return e,w
def lowest_active(d,frac=0.05,bw=0.12):
    """per |Q| bin: lowest pole carrying >= frac of S(Q) (density-active neutral mode)"""
    rows=[]
    for r in d['res']:
        if r['S']<1e-10: continue
        e,w=poles(r); m=(w>=frac*r['S'])&(e>0.5)
        if m.any(): rows.append((r['q'],e[m][0],r['f']/r['S']))
    rows=np.array(rows); qb=np.round(rows[:,0]/bw)*bw; out=[]
    for x in np.unique(qb):
        m=qb==x; out.append((rows[m,0].mean(),rows[m,1].min(),rows[m,2].min()))
    return np.array(out)
def sqw_map(d,qgrid,w,eta=0.35):
    M=np.zeros((len(qgrid)-1,len(w))); cnt=np.zeros(len(qgrid)-1)
    for r in d['res']:
        if len(r['al'])==0: continue
        i=np.searchsorted(qgrid,r['q'])-1
        if i<0 or i>=len(qgrid)-1: continue
        e,wt=poles(r); m=e>0.5
        M[i]+=np.sum(wt[m][:,None]*eta/np.pi/((w[None,:]-e[m][:,None])**2+eta**2),0); cnt[i]+=1
    M[cnt>0]/=cnt[cnt>0][:,None]; M[cnt==0]=np.nan; return M
summ={}
for (n,v),d in D.items():
    if d is None: continue
    la=lowest_active(d); lm=la[la[:,0]<2.5]; k=np.argmin(lm[:,1]); la=lm; Dd=dispersion(d)
    qS,S=binned(d,'S'); 
    summ[f'{n}_{v}']=dict(roton_gap=float(la[k,1]),roton_q=float(la[k,0]),ed_gap=float(Dd[:,1].min()),Smax=float(S.max()),qSmax=float(qS[S.argmax()]),
                          sma_min=float(la[:,2].min()))
json.dump(summ,open(f'/home/claude/sim/res3/summary_{NE}.json','w'),indent=1)
for k,v in summ.items(): print(k,{a:round(b,3) for a,b in v.items()})

# ---------- Fig. S2: microscopic neutral spectrum
w=np.linspace(0,16,500); qg=np.arange(0.3,4.65,0.12)
fig=plt.figure(figsize=(7.1,2.35)); gs=fig.add_gridspec(1,5,width_ratios=[1,1,1,0.12,1.25],wspace=0.2)
vm=None
for i,(key,lab) in enumerate(((('A','main'),'band A'),(('A','twin'),'homogeneous twin of A'),(('LLL','main'),'lowest Landau level'))):
    ax=fig.add_subplot(gs[0,i]); M=sqw_map(D[key],qg,w)
    if vm is None: vm=np.nanpercentile(M,99.5)
    from matplotlib.colors import PowerNorm
    ax.pcolormesh(0.5*(qg[1:]+qg[:-1]),w,M.T,cmap='magma_r',norm=PowerNorm(0.35,vmin=0,vmax=vm),shading='nearest',rasterized=True)
    la=lowest_active(D[key]); ax.plot(la[:,0],la[:,2],color=C[2],lw=0.9,label='SMA $\\bar f/\\bar S$')
    ax.set_title(lab,fontsize=7.5,color=INK); ax.set_xlabel('$q\\ell$'); ax.set_ylim(0,16); ax.set_xlim(0.3,4.6)
    if i==0: ax.set_ylabel('$\\hbar\\omega$ (meV)'); ax.legend(fontsize=6,loc='upper right')
    else: ax.set_yticklabels([])
ax=fig.add_subplot(gs[0,4])
for n,v,lab,c,ls in BANDS:
    if D[(n,v)] is None: continue
    la=lowest_active(D[(n,v)]); ax.plot(la[:,0],la[:,1],color=c,ls=ls,marker='o' if v=='main' else None,ms=2,label=lab)
ax.set_xlabel('$q\\ell$'); ax.set_ylabel('lowest density-active mode (meV)'); ax.set_xlim(0.3,4.6); ax.legend(fontsize=6,loc='upper right',ncol=1)
fig.savefig(O+'dyn_disp.pdf',bbox_inches='tight'); plt.close()

# ---------- Fig S-dyn2: static structure factor and oscillator strength
fig,axs=plt.subplots(1,3,figsize=(7.1,2.2),gridspec_kw=dict(wspace=0.36))
for n,v,lab,c,ls in BANDS:
    d=D[(n,v)]
    if d is None: continue
    q,S=binned(d,'S',0.02); q2,f=binned(d,'f',0.02)
    axs[0].plot(q,S,color=c,ls=ls,lw=0.9,label=lab); axs[1].plot(q2,f,color=c,ls=ls,lw=0.9)
x=np.linspace(0.3,0.95,50); axs[0].plot(x,(1-1/3)/(8/3)*x**4,color=INK,lw=0.6,ls='-.',label='$\\frac{1-\\nu}{8\\nu}(q\\ell)^4$')
axs[0].set_xlabel('$q\\ell$'); axs[0].set_ylabel('$\\bar S(q)$'); axs[0].legend(fontsize=5.8,loc='lower right')
axs[1].set_xlabel('$q\\ell$'); axs[1].set_ylabel('$\\bar f(q)$ (meV)')
ax=axs[2]
an=pickle.load(open('/home/claude/sim/analysis2.pkl','rb'))
R2={k:pickle.load(open(f'/home/claude/sim/res2/{k}.pkl','rb')) for k in ('A','P3')}
for i,k in enumerate(('A','P3')):
    dm=D[(k,'main')]; dt=D[(k,'twin')]
    qm,Sm=binned(dm,'S',0.02); qt,St=binned(dt,'S',0.02); n=min(len(qm),len(qt)); sel=qm[:n]<1.3
    ax.loglog(qm[:n][sel],np.abs(Sm[:n]-St[:n])[sel],'o',color=C[i],ms=3,label=f'{k}: $|\\bar S-\\bar S_{{\\rm twin}}|$')
    K=R2[k]['sizes'][NE]['current']['avg']['K_iso']/NE; ell=dm['ell']; xx=np.linspace(0.3,1.3,20)
    ax.loglog(xx,K*(xx/ell)**2,color=C[i],lw=0.8,label=f'{k}: $\\mathcal{{K}} q^2/N$')
ax.loglog(xx,(1-1/3)/(8/3)*xx**4,color=INK,lw=0.6,ls='-.',label='Laughlin $q^4$')
ax.set_xlabel('$q\\ell$'); ax.set_ylabel('excess over twin'); ax.legend(fontsize=5.5,loc='upper left')
ax.set_xticks([0.4,0.6,1.0]); ax.set_xticklabels(['0.4','0.6','1.0']); ax.minorticks_off()
for a_,t_ in zip(axs,'abc'): a_.set_title(f'({t_})',fontsize=7.5,loc='left',color=INK)
fig.savefig(O+'dyn_sq.pdf',bbox_inches='tight'); plt.close()
print('figs done')
