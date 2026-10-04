"""Additions to generated tables (idempotent): C3-family spectra in tab_clusters, strong-screening row in tab_unc."""
import pickle
B='/home/claude/build2/'
cs=pickle.load(open('/home/claude/sim/res4/cspec.pkl','rb'))
def sci(x):
    import math
    e=int(math.floor(math.log10(abs(x)))); return f'${x/10**e:.1f}\\times10^{{{e}}}$'
s=open(B+'tab_clusters.tex').read()
for Ne,mat in ((4,'4&2\\\\2&4'),(7,'5&1\\\\4&5')):
    r=cs[('A',Ne)]
    old_start=f'C & {Ne} & $\\begin{{psmallmatrix}}{mat}\\end{{psmallmatrix}}$ &'
    i=s.index(old_start); j=s.index('\\\\',s.index('\\end{psmallmatrix}',i)+20)
    row=s[i:j]; parts=row.split('&')
    D=parts[-1].strip()
    new=f"{old_start} {r['dim']} & {r['gap']:.2f} & {sci(r['spread'])} & {D}"
    s=s[:i]+new+s[j:]
open(B+'tab_clusters.tex','w').write(s)
u=open(B+'tab_unc.tex').read()
if 'd=50' not in u:
    a=pickle.load(open('/home/claude/sim/res4/cur_A_A6_d50.pkl','rb'))['avg']['D_iso']/6; p=pickle.load(open('/home/claude/sim/res4/cur_P3_A6_d50.pkl','rb'))['avg']['D_iso']/6
    u=u.replace("parameters & $d=500$~\\AA & 11.35 & 11.11 & 35.59 & 34.80\\\\",f"parameters & $d=500$~\\AA & 11.35 & 11.11 & 35.59 & 34.80\\\\\nparameters & $d=50$~\\AA\\ (strong screening) & -- & {a:.2f} & -- & {p:.2f}\\\\")
    open(B+'tab_unc.tex','w').write(u)
print('post_r4 done')
