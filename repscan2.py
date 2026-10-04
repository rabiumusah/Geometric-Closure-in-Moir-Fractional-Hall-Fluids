import sys; sys.path.insert(0,'/home/claude/sim')
from repscan import job
import glob,os
names=sorted(os.path.basename(f)[:-4] for f in glob.glob('/home/claude/sim/res2/*.pkl') if not f.endswith('LL.pkl'))
part=int(sys.argv[1])
allj=[(n,'main',7) for n in names]+[('A','twin',7),('P3','twin',7),('LLL','main',7),('A','main',8),('A','twin',8),('P3','main',8),('P3','twin',8),('LLL','main',8)]
jobs=allj[part::2]
for j in jobs: print(job(*j),flush=True)
