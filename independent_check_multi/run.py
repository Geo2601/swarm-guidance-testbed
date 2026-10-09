import numpy as np, sys, json
from multiprocessing import Pool
from msim import init_scene, run_trial

METHODS = ['gwo_avg', 'gwo_group', 'hungarian', 'bottleneck']
NTR = 100


def cells():
    out = []
    for N in (4, 8):
        for vs in (0, 5):
            for M in (1, 2, 3, 5):
                for S in ((None,) if M == 1 else (25, 60)):
                    out.append((N, M, S, vs))
    return out


def run_cell(cell):
    N, M, S, vs = cell
    cid = N * 1000 + M * 100 + (S or 0) + vs * 7
    res = {}
    for meth in METHODS:
        Ts, viol, ok = [], 0, 0
        for t in range(NTR):
            p, q, vt = init_scene(cid * 100000 + t, N, M, S or 0, vs)
            r = run_trial(meth, p, q, vt, seed=5_000_000 + cid * 1000 + t)
            if r['T'] is not None:
                ok += 1; Ts.append(r['T'])
            viol += r['viol']
        res[meth] = (ok, float(np.median(Ts)) if Ts else None, viol)
    return cell, res


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(run_cell, cells())
    json.dump([[list(c), r] for c, r in out], open('grid.json', 'w'))
    print(f"{'N':>2} {'M':>2} {'S':>3} {'vt':>2} | " + " | ".join(f"{m:^18}" for m in METHODS))
    for c, r in out:
        N, M, S, vs = c
        cols = []
        for m in METHODS:
            ok, med, viol = r[m]
            cols.append(f"{ok:3d}% T={med if med is not None else '-':>4} v={viol:3d}%")
        print(f"{N:>2} {M:>2} {str(S or '-'):>3} {vs:>2} | " + " | ".join(cols))
