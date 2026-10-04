import sys,json,pickle,glob,os,numpy as np
sys.path.insert(0,'/home/claude/sim')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.linewidth":0.6,"font.family":"serif","mathtext.fontset":"cm","lines.linewidth":1.2,
                     "xtick.major.width":0.5,"ytick.major.width":0.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]
INK="#3a3a38"; MUTED="#8a8984"
O='/home/claude/build2/fig/'
o=pickle.load(open('analysis2.pkl','rb')); r=o['rows']; ref=o['ref']
X={os.path.basename(f)[:-4]:pickle.load(open(f,'rb')) for f in glob.glob('res2x/*.pkl')}
R={n:pickle.load(open(f'res2/{n}.pkl','rb')) for n in ('A','P3','LLL')}

# ---------------- Fig 1: geometry of the matched pair
g=pickle.load(open('figgeo2.pkl','rb'))
fig=plt.figure(figsize=(7.1,2.2)); gs=fig.add_gridspec(1,5,width_ratios=[1,1,0.06,0.32,1.5],wspace=0.25)
vmax=max(np.abs(g[k]['Om']/g[k]['Omean']-1).max() for k in g)
for i,k in enumerate(('A','P3')):
    ax=fig.add_subplot(gs[0,i]); M=g[k]['Om']/g[k]['Omean']
    im=ax.imshow(M.T,origin='lower',cmap='RdBu_r',vmin=1-vmax,vmax=1+vmax,extent=[0,1,0,1],aspect='equal')
    ax.set_title(f'band {k}: $\\sigma_\\Omega$={M.std():.3f}',fontsize=7.5,color=INK); ax.set_xlabel('$k_1$ (units of $b_1$)'); 
    if i==0: ax.set_ylabel('$k_2$ (units of $b_2$)')
    else: ax.set_yticklabels([])
cax=fig.add_subplot(gs[0,2]); cb=fig.colorbar(im,cax=cax); cb.set_label('$\\Omega_{\\mathbf{k}}/\\bar\\Omega$',fontsize=7); cb.outline.set_linewidth(0.4)
ax=fig.add_subplot(gs[0,4])
for i,k in enumerate(('A','P3')):
    x=g[k]['qs']*g[k]['g0']*np.sqrt(g[k]['l2'])
    ax.plot(x,g[k]['mF'],color=C[i],label=f'{k}: $\\langle|F|\\rangle$',marker='os'[i],ms=2.5,markevery=4)
    ax.fill_between(x,g[k]['mF']-g[k]['sF'],g[k]['mF']+g[k]['sF'],color=C[i],alpha=0.18,lw=0)
ax.plot(x,np.exp(-x**2/4),color=MUTED,ls='--',lw=0.9,label='LLL $e^{-q^2\\ell^2/4}$')
ax.set_xlabel('$q\\ell$ (along $\\mathbf{b}_1$)'); ax.set_ylabel('$|F_{\\mathbf{k},\\mathbf{q}}|$, mean $\\pm$ s.d.'); ax.legend(fontsize=6.5,loc='upper right')
ax.set_ylim(0,1.02)
fig.savefig(O+'geo.pdf',bbox_inches='tight'); plt.close()

# ---------------- Fig 2: topology (PES + spectral flow)
fig,axs=plt.subplots(1,3,figsize=(7.1,2.2),gridspec_kw=dict(wspace=0.38))
for i,k in enumerate(('A','P3')):
    ax=axs[i]; p=R[k]['sizes'][7]['pes']; xi=p['xi']; cnt=p['count']
    ax.plot(np.arange(len(xi)),xi,'o',ms=1.6,color=C[i],mew=0)
    ax.axvline(cnt-0.5,color=MUTED,lw=0.7,ls=':'); ax.text(cnt+5,xi[cnt-1]-1.5,f'{cnt} levels',fontsize=6.5,color=INK)
    ax.set_xlabel('level index'); ax.set_ylabel('$\\xi$'); ax.set_title(f'PES, band {k}, $N_e=7$, $N_A=3$',fontsize=7.5,color=INK)
if os.path.exists('sflow.pkl'):
    sf=pickle.load(open('sflow.pkl','rb')); ths,E=sf['A']; ax=axs[2]
    E0=E[:,:,0].min(axis=1,keepdims=True)
    for K in range(E.shape[1]):
        for j in range(E.shape[2]): ax.plot(ths/(2*np.pi),E[:,K,j]-E0[:,0],color=MUTED if j>0 else C[0],lw=0.5 if j>0 else 0.9)
    ax.set_ylim(-0.2,6); ax.set_xlabel('$\\theta_1/2\\pi$'); ax.set_ylabel('$E-E_0$ (meV)'); ax.set_title('spectral flow, band A, $N_e=6$',fontsize=7.5,color=INK)
fig.savefig(O+'topo.pdf',bbox_inches='tight'); plt.close()

# ---------------- Fig 3: selection rule
fig,axs=plt.subplots(1,2,figsize=(7.1,2.4),gridspec_kw=dict(wspace=0.32))
ax=axs[0]; Ns=[4,5,6,7,8]
series=[(r"A (moir\'e)",[r['A'].get(f'D{N}') for N in Ns],C[0],'o'),('P3 (matched)',[r['P3'].get(f'D{N}') for N in Ns],C[1],'s'),
        ('P1 (matched)',[r['P1'].get(f'D{N}') for N in Ns],C[2],'^'),('A twin',[r['A'].get(f'DT{N}') for N in Ns],C[0],'o'),
        ('P3 twin',[r['P3'].get(f'DT{N}') for N in Ns],C[1],'s'),('LLL',[ref['LLL'][N]['D'] if N in ref['LLL'] else None for N in Ns],C[6],'D')]
for i,(lab,y,c,m) in enumerate(series):
    y=np.array([v if v is not None else np.nan for v in y],float)
    ax.semilogy(Ns,y,marker=m,color=c,ms=4,ls='-' if 'twin' not in lab and lab!='LLL' else '--',mfc=c if 'twin' not in lab else 'white',label=lab)
for n9,c,m in (('n9C_A',C[0],'o'),('n9C_P3',C[1],'s')):
    if n9 in X: ax.semilogy([9],[X[n9]['D_iso']/9],marker=m,color=c,ms=5,mfc=c,mec='k',mew=0.5)
ax.axhspan(1e-25,1e-4,color='#f1f0ec',zorder=0); ax.text(4.05,1e-19,'homogeneous class: $\\mathcal{D}=0$',fontsize=6.5,color=INK)
ax.set_xticks([4,5,6,7,8,9]); ax.set_xticklabels(['4','5','6','7','8','9\n($C_3$)']); ax.set_xlabel('$N_e$'); ax.set_ylabel('$\\mathcal{D}/N$ (meV Å$^2$)'); ax.set_ylim(1e-24,1e4); ax.legend(fontsize=6.6,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=0.8)
ax=axs[1]
for i,(n,c,m) in enumerate((('A',C[0],'o'),('P3',C[1],'s'),('P1',C[2],'^'))):
    ax.semilogy(Ns,[r[n].get(f'spread{N}',np.nan) for N in Ns],marker=m,color=c,ms=4,label=n)
    ax.semilogy(Ns,[r[n].get(f'spreadT{N}',np.nan) for N in Ns],marker=m,color=c,ms=4,ls='--',mfc='white',label=n+' twin')
ax.set_xlabel('$N_e$'); ax.set_ylabel('$\\Delta_{\\rm GS}$ (meV)'); ax.legend(fontsize=6.6,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=0.8)
fig.savefig(O+'sel.pdf',bbox_inches='tight'); plt.close()

# ---------------- Fig 4: dose-response law
fam=lambda n: 'matched' if n in ('A','P1','P2','P3') else ('strain' if n.startswith('s') else ('twist angle' if n.startswith('th') else ('$\\psi$' if n.startswith('psi') else ('$w$' if n.startswith('w') else '$V$'))))
fams=['matched','$\\psi$','$w$','$V$','twist angle','strain']; mk=dict(zip(fams,'osD^vP'))
fig,axs=plt.subplots(1,3,figsize=(7.1,2.35),gridspec_kw=dict(wspace=0.38))
Ne=7
for j,(key,lab,logx) in enumerate((('sOm','$\\sigma_\\Omega$ (Berry-curvature dispersion)',True),('tau0','$\\tau_0$ (quartic cumulant)',False),('gstar','$\\bar g_*$ (Å$^2$)',False))):
    ax=axs[j]
    for f_i,f in enumerate(fams):
        ns=[n for n in r if fam(n)==f]
        x=[r[n][key] for n in ns]; y=[r[n][f'D{Ne}']/(r[n]['EC']*r[n]['ell']**2)*1e3 for n in ns]
        ax.scatter(x,y,s=14,marker=mk[f],color=C[f_i],edgecolors='white',linewidths=0.4,label=f,zorder=3)
    ax.set_yscale('log')
    if logx: ax.set_xscale('log')
    from matplotlib.ticker import LogLocator,NullFormatter,ScalarFormatter
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(lab)
    ax.set_yticks([0.2,0.5,1,2]); ax.set_yticklabels(['0.2','0.5','1','2'])
    if logx: ax.set_xticks([0.12,0.2,0.3]); ax.set_xticklabels(['0.12','0.2','0.3'])
    if j==0:
        ax.set_ylabel('$10^3\\,\\tilde{\\mathcal{D}}=10^3\\mathcal{D}/(NE_C\\ell^2)$')
        xs=np.array([r[n]['sOm'] for n in r]); ys=np.array([r[n][f'D{Ne}']/(r[n]['EC']*r[n]['ell']**2)*1e3 for n in r]); p=np.polyfit(np.log(xs),np.log(ys),1)
        xx=np.linspace(xs.min()*0.9,xs.max()*1.1,10); ax.plot(xx,np.exp(np.polyval(p,np.log(xx))),color=MUTED,lw=0.8,ls='--',zorder=1)
        ax.text(0.04,0.9,f'slope {p[0]:.2f}',transform=ax.transAxes,fontsize=6.5,color=INK)
        for n in ('A','P3'): ax.annotate(n,(r[n]['sOm'],r[n][f'D{Ne}']/(r[n]['EC']*r[n]['ell']**2)*1e3),xytext=(4,-8),textcoords='offset points',fontsize=6.5,color=INK)
    if j==1:
        ax.axvline(0.665,color=MUTED,lw=0.5,ls=':'); ax.text(0.68,0.97,'A, P1, P2, P3: same $\\bar g,\\bar T$',transform=ax.get_xaxis_transform(),fontsize=6.6,color=INK,va='top')
axs[2].legend(fontsize=6.6,loc='upper right',handletextpad=0.2)
fig.savefig(O+'law.pdf',bbox_inches='tight'); plt.close()

# ---------------- Fig 5: experimental consequences
fig,axs=plt.subplots(1,3,figsize=(7.1,2.3),gridspec_kw=dict(wspace=0.38))
ax=axs[0]; eta=0.25
for i,k in enumerate(('A','P3','A_h')):
    key=f'spec_{k}'
    if key not in X: continue
    w=np.linspace(0,42,1000); s=np.zeros_like(w)
    for m in X[key]['cur']:
        E=np.asarray(m['E']); wt=np.asarray(m['w'])
        s+=np.sum(wt[None,:]/E[None,:]*eta/np.pi/((w[:,None]-E[None,:])**2+eta**2),axis=1)/3
    s/= (3*6)   # per particle (N_e=6)
    ax.plot(w,s,color=[C[0],C[1],C[0]][i],ls=['-','-','--'][i],label={'A':'A','P3':'P3','A_h':'A twin ($\\equiv0$)'}[k])
ax.set_xlabel('$\\hbar\\omega$ (meV)'); ax.set_ylabel('projected intraband absorption (arb. u.)'); ax.legend(fontsize=6.5); ax.set_xlim(0,42); ax.axvline(15,color=MUTED,lw=0.6,ls=':'); ax.text(15.6,ax.get_ylim()[1]*0.55,'isolation\ngap',fontsize=6.6,color=INK)
ax=axs[1]
for i,k in enumerate(('A','P3')):
    for Ne,mk_ in ((6,'o'),(8,'s')):
        key=f'sq_{k}|{Ne}'
        if key not in X: continue
        d=X[key]; qm,qt=d['main'],d['twin']; l=np.sqrt(R['A']['geo']['Auc']/(2*np.pi))
        ql=qm[:,0]*l; ex=qm[:,3]-qt[:,3]; pos=ex>0
        ax.scatter(ql[pos],ex[pos],s=10,marker=mk_,color=C[i],edgecolors='white',linewidths=0.3,label=f'{k}, $N_e$={Ne}')
        ax.scatter(ql[~pos],-ex[~pos],s=10,marker=mk_,facecolors='none',edgecolors=C[i],linewidths=0.5)
    K=r[k].get('K8',r[k].get('K7')); qq=np.linspace(0.2,3,50); l=np.sqrt(R['A']['geo']['Auc']/(2*np.pi))
    ax.plot(qq,K*(qq/l)**2,color=C[i],lw=0.8,ls='--')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-5,0.1); ax.set_xlabel('$q\\ell$'); ax.set_ylabel('$|\\bar S_{\\rm moir\\acute e}-\\bar S_{\\rm twin}|$'); ax.legend(fontsize=6.6,loc='upper left')
ax=axs[2]
if 'dichC' in X:
    dc=X['dichC']
    for i,(pre,phi) in enumerate((('sA',0),('sB',30))):
        es=[0.0]+[e for e in (0.02,0.04,0.06,0.08,0.1) if (f'{pre}{e}',7) in dc]
        names=['A']+[f'{pre}{e}' for e in es[1:]]
        if ('A',7) not in dc: continue
        y=[(dc[(n,7)]['D'][0,0]-dc[(n,7)]['D'][1,1])/(dc[(n,7)]['D'][0,0]+dc[(n,7)]['D'][1,1]) for n in names]
        y2=[2*dc[(n,7)]['D'][0,1]/(dc[(n,7)]['D'][0,0]+dc[(n,7)]['D'][1,1]) for n in names]
        ax.plot(es,y,marker='os'[i],color=C[i],ms=3.5,label=f'$\\varphi={phi}^\\circ$: $(\\mathcal{{D}}_{{xx}}-\\mathcal{{D}}_{{yy}})/{{\\rm tr}}$')
        ax.plot(es,y2,marker='os'[i],color=C[i],ms=3.5,ls='--',mfc='white',label=f'$\\varphi={phi}^\\circ$: $2\\mathcal{{D}}_{{xy}}/{{\\rm tr}}$')
    ax.axhline(0,color=MUTED,lw=0.5); ax.set_xlabel('moir\\\'e strain $\\varepsilon$'); ax.set_ylabel('THz linear dichroism ($N_e=7$, $C_3$ torus)'); ax.legend(fontsize=6.6)
fig.savefig(O+'exp.pdf',bbox_inches='tight'); plt.close()
print('figs done', sorted(os.listdir(O)))
