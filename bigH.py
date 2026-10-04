"""memory-lean sector Hamiltonian for large clusters: exact nnz count pass + int32 indices."""
import numpy as np, numba as nb
from scipy.sparse import csr_matrix
from ed import sgn_below

@nb.njit(cache=True)
def _count(states, X1, X2, D, grid, Nk, Ne):
    dim=len(states); cnt=np.zeros(dim,np.int64); occ=np.empty(Ne,np.int64)
    for i in range(dim):
        s=states[i]; m=0
        for k in range(Nk):
            if (s>>k)&1:
                occ[m]=k; m+=1
        n=1
        for x in range(Ne):
            c=occ[x]; s1=s^(1<<c)
            for y in range(x+1,Ne):
                d=occ[y]; s2=s1^(1<<d)
                P1=X1[c]+X1[d]; P2=X2[c]+X2[d]
                for a in range(Nk):
                    b=grid[(P1-X1[a])%D,(P2-X2[a])%D]
                    if a<b:
                        if (s2>>a)&1 or (s2>>b)&1: continue
                        n+=1
        cnt[i]=n
    return cnt

@nb.njit(cache=True)
def _fill(states, Ms, X1, X2, D, grid, Nk, Ne, eps, indptr, indices, vals):
    dim=len(states); occ=np.empty(Ne,np.int64)
    for i in range(dim):
        s=states[i]; m=0
        for k in range(Nk):
            if (s>>k)&1:
                occ[m]=k; m+=1
        p=indptr[i]
        e=0.0
        for t in range(Ne): e+=eps[occ[t]]
        indices[p]=i; vals[p]=e; p+=1
        for x in range(Ne):
            c=occ[x]; s1=s^(1<<c); sg1=sgn_below(s,c)
            for y in range(x+1,Ne):
                d=occ[y]; sg2=sgn_below(s1,d); s2=s1^(1<<d)
                P1=X1[c]+X1[d]; P2=X2[c]+X2[d]
                for a in range(Nk):
                    b=grid[(P1-X1[a])%D,(P2-X2[a])%D]
                    if a<b:
                        if (s2>>a)&1 or (s2>>b)&1: continue
                        sg3=sgn_below(s2,b); s3=s2|(1<<b)
                        sg4=sgn_below(s3,a); s4=s3|(1<<a)
                        lo=0; hi=dim-1; idx=-1
                        while lo<=hi:
                            mid=(lo+hi)//2
                            if states[mid]==s4: idx=mid; break
                            elif states[mid]<s4: lo=mid+1
                            else: hi=mid-1
                        indices[p]=idx; vals[p]=0.5*Ms[a,b,c,d]*sg1*sg2*sg3*sg4; p+=1
    return 0

def sector_H_T(S,K):
    """returns H^T (row i = column i of H) as CSR; since H is Hermitian we return its conjugate transpose rows -> use .conj() """
    cl=S.cl; st=cl.sector(K)
    cnt=_count(st,cl.X1,cl.X2,cl.D,cl.grid,cl.Nk,cl.Ne)
    indptr=np.zeros(len(st)+1,np.int64); indptr[1:]=np.cumsum(cnt)
    nnz=int(indptr[-1]); indices=np.empty(nnz,np.int32); vals=np.empty(nnz,np.complex128)
    _fill(st,S.Ms,cl.X1,cl.X2,cl.D,cl.grid,cl.Nk,cl.Ne,S.eps,indptr,indices,vals)
    # entries stored as (col=i -> row=idx): matrix element H[idx,i]; build CSC with indptr over columns
    from scipy.sparse import csc_matrix
    H=csc_matrix((vals,indices,indptr),shape=(len(st),len(st)))
    H.sum_duplicates()
    return H
