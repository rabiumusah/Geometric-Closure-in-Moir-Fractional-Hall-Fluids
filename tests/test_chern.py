"""Reproducibility tests of the sewn non-Abelian Chern routine (chern_na.py):
 1. integer result and branch safety for band A (N_e=5) on M=4 and M=6 grids;
 2. invariance under random single-particle (Bloch) gauges at every grid point;
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import topo, chern_na
from ana import Params, Band, get_cluster
from ed import System

b = Band(Params(Gcut=3.1, smap='exp')); cl = get_cluster('A', 5)
r4 = chern_na.chern_na(b, cl, 4); r6 = chern_na.chern_na(b, cl, 6)
print('M=4', r4['C'], r4['Fmax'], r4['smin']); print('M=6', r6['C'], r6['Fmax'], r6['smin'])
assert abs(r4['C'] + 1) < 1e-9 and abs(r6['C'] + 1) < 1e-9 and r6['Fmax'] < 1.0 and r6['smin'] > 0.5

rng = np.random.default_rng(7)
class RandomGauge(System):
    def __init__(self, *a, **k):
        k['gauge_rng'] = rng; super().__init__(*a, **k)
topo.System = RandomGauge; chern_na.make_system = lambda band, cl, tw, lam=None, **kw: RandomGauge(band, cl, twist=tuple(tw), lam_fn=lam)
rg = chern_na.chern_na(b, cl, 4)
print('random gauge M=4', rg['C'], rg['Fmax'])
assert abs(rg['C'] + 1) < 1e-9 and abs(rg['Fmax'] - r4['Fmax']) < 1e-8
print('ALL CHERN TESTS PASSED')
