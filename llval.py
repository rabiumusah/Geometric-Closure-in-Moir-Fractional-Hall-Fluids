"""Validation of the cumulant estimator on exact Landau-level form factors (SM Sec. S8; Table S2 and Fig. S1).

The N-th Landau level with a unimodular metric deformation G = diag(e^{2s}, e^{-2s}) has the exact form factor
    |F_N(q)| = |L_N(q.Gq/2)| exp(-q.Gq/4)            (magnetic length l_B = 1),
whose log-expansion gives gbar = (2N+1)/2 G and, for N = 1,
    T_0 = -3(cosh^2 2s + sinh^2 2s / 2),  T_2 = -3 sinh 4s,  T_4 = -(3/2) sinh^2 2s
(signed cos 2phi and cos 4phi coefficients of the quartic form p/q^4 for a deformation along x).

The estimator is the one used for the moire bands (SM Sec. S8): y = log|F| is fitted on a polar grid (40 radii
between 0.1 q_max and q_max, 48 angles), linearly in (gxx, gyy, gxy, a0, a2c, a2s, a4c, a4s), optionally with nuisance terms
q^6 x {m = 0, +-2, +-4, +-6} and q^8 x {m = 0, +-2, +-4}; rows are weighted by x^-2 with x = q sqrt(g*).
Relative noise 2e-7 is added to |F|; quoted values are the mean and standard deviation over 300 noise realizations.

Outputs: ../build/tab_ll.tex, ../build/figV1.pdf and res4/llval.json.
Usage: python3 llval.py
"""
import json, os
import numpy as np
from scipy.special import eval_laguerre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'build')
RNG_SEED = 20261004
NOISE = 2e-7
NQ, NPHI, NBOOT = 40, 48, 300
QMIN_FRAC = 0.1          # radial grid 0.1 q_max ... q_max (the smallest momenta only amplify the noise)


def exact_logF(N, s, qx, qy):
    qG = np.exp(2 * s) * qx**2 + np.exp(-2 * s) * qy**2
    return np.log(np.abs(eval_laguerre(N, qG / 2))) - qG / 4


def exact_cumulants(N, s):
    gxx, gyy = (2 * N + 1) / 2 * np.exp(2 * s), (2 * N + 1) / 2 * np.exp(-2 * s)
    if N == 0:
        return gxx, gyy, 0.0, 0.0, 0.0
    c, sh = np.cosh(2 * s), np.sinh(2 * s)
    return gxx, gyy, -3 * (c**2 + sh**2 / 2), -3 * np.sinh(4 * s), -1.5 * sh**2


def design(q, phi, nuis):
    qx, qy = q * np.cos(phi), q * np.sin(phi)
    cols = [-0.5 * qx**2, -0.5 * qy**2, -qx * qy]
    q4 = q**4 / 24
    cols += [q4, q4 * np.cos(2 * phi), q4 * np.sin(2 * phi), q4 * np.cos(4 * phi), q4 * np.sin(4 * phi)]
    if nuis >= 6:
        for m in (0, 2, 4, 6):
            cols += [q**6 * np.cos(m * phi)] + ([q**6 * np.sin(m * phi)] if m else [])
    if nuis >= 8:
        for m in (0, 2, 4):
            cols += [q**8 * np.cos(m * phi)] + ([q**8 * np.sin(m * phi)] if m else [])
    return np.array(cols).T


def fit(N, s, qmax, nuis=8, rng=None):
    """Fit the cumulants for NBOOT independent noise realizations; return their mean and standard deviation
    (gxx, gyy, T0, a2c, a4c)."""
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    q1 = np.linspace(QMIN_FRAC * qmax, qmax, NQ)
    p1 = np.arange(NPHI) * 2 * np.pi / NPHI
    q, phi = [a.ravel() for a in np.meshgrid(q1, p1, indexing='ij')]
    F = np.exp(exact_logF(N, s, q * np.cos(phi), q * np.sin(phi)))
    w = 1.0 / (q**2 * (2 * N + 1) / 2)          # x^-2 row weights, x = q sqrt(g*)
    A = design(q, phi, nuis) * w[:, None]
    out = []
    for _ in range(NBOOT):
        y = np.log(F * (1 + NOISE * rng.standard_normal(F.size)))
        c, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        out.append([c[0], c[1], c[3], c[4], c[6]])
    out = np.array(out)
    return out.mean(0), out.std(0)


def fmt(v, e):
    """value(uncertainty) in units of the third decimal"""
    u = max(1, int(round(e * 1000)))
    return ('$-$' if v < -5e-4 else ('$+$' if v > 5e-4 else '')) + f'{abs(v):.3f}({u})'


def main():
    rows, results = [], {}
    for N, s in ((0, 0.0), (1, 0.0), (1, 0.15)):
        ex = exact_cumulants(N, s)
        val, err = fit(N, s, 0.5, nuis=8)
        results[f'N{N}_s{s}'] = {'exact': list(map(float, ex)), 'fit': val.tolist(), 'err': err.tolist()}
        lab = '0 (LLL)' if N == 0 else '1'
        ex_q = ' / '.join(f'{x:.3f}'.replace('-', '$-$') if abs(x) > 5e-4 else '0.000' for x in ex[2:])
        rows.append(f"{lab} & {s:g} & {ex[0]:.4f} / {ex[1]:.4f} & {val[0]:.4f} / {val[1]:.4f} & {ex_q} & "
                    + ' / '.join(fmt(v, e) for v, e in zip(val[2:], err[2:])) + r' \\')
    tab = (r"""\begin{table}[h]\centering\small
\caption{\textbf{Cumulant estimator on exact Landau-level form factors} (window $q_{\max}\ell_B=0.5$, nuisance terms to $q^8$, relative noise $2\times10^{-7}$; mean and standard deviation over 300 noise realizations).}\label{tab:ll}
\begin{tabular}{@{}cc cc cc@{}}\toprule
$N$ & $s$ & $\gb_{xx}/\gb_{yy}$ exact & fit & $T_0/T_2/T_4$ exact & fit\\\midrule
""" + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, 'tab_ll.tex'), 'w').write(tab)

    # Fig. S1
    win = np.linspace(0.2, 0.9, 15)
    T0 = {k: [fit(1, 0.0, w, nuis=k) for w in win] for k in (0, 6, 8)}
    svals = np.linspace(0, 0.3, 11)
    a4 = [fit(1, s, 0.5, nuis=8) for s in svals]
    results['window_scan'] = {str(k): [[float(w), float(v[0][2]), float(v[1][2])] for w, v in zip(win, T0[k])] for k in T0}
    results['a4_scan'] = [[float(s), float(v[0][4]), float(v[1][4])] for s, v in zip(svals, a4)]
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1, 2, figsize=(4.9, 1.9))
    for k, lab, c in ((0, 'no nuisance term', 'C0'), (6, r'nuisance order $q^6$', 'C1'), (8, r'nuisance order $q^8$', 'C2')):
        v = np.array([t[0][2] for t in T0[k]]); e = np.array([t[1][2] for t in T0[k]])
        ax[0].errorbar(win, v, e, marker='o', ms=3, lw=1, color=c, label=lab, capsize=1.5)
    ax[0].axhline(-3, color='k', ls='--', lw=0.8)
    ax[0].set_ylim(-4.5, -1.5); ax[0].set_xlabel(r'fit window $q_{\max}\ell_B$')
    ax[0].set_ylabel(r'recovered $\bar T_0$ ($N=1$ LL)'); ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title('(a)', loc='left', fontsize=8)
    ss = np.linspace(0, 0.3, 100)
    ax[1].plot(ss, -1.5 * np.sinh(2 * ss)**2, 'k-', lw=1, label=r'exact $-\frac{3}{2}\sinh^2 2s$')
    ax[1].errorbar(svals, [v[0][4] for v in a4], [v[1][4] for v in a4], fmt='o', ms=3, color='C0', label='fit', capsize=1.5)
    ax[1].set_xlabel('metric deformation $s$'); ax[1].set_ylabel(r'spin-4 harmonic of $\bar T$')
    ax[1].legend(fontsize=6, frameon=False); ax[1].set_title('(b)', loc='left', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figV1.pdf'), bbox_inches='tight')
    os.makedirs(os.path.join(HERE, 'res4'), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, 'res4', 'llval.json'), 'w'), indent=1)
    print(tab)


if __name__ == '__main__':
    main()
