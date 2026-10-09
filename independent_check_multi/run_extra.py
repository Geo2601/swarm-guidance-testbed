import numpy as np
from msim import init_scene, run_trial

# (a) single vehicle, single target at origin
print("(a) N=1, M=1, 300 trials")
for vs in (0, 5):
    for meth in ('gwo_noboids', 'hungarian'):
        ok, Ts = 0, []
        for t in range(300):
            p, q, vt = init_scene(900000 + vs * 1000 + t, 1, 1, 0, vs)
            r = run_trial(meth, p, q, vt, seed=7_000_000 + t)
            if r['T'] is not None:
                ok += 1; Ts.append(r['T'])
        print(f"  vt={vs} {('GWO' if meth.startswith('gwo') else 'lead'):5s} capt={100*ok/300:5.1f}%  medT={np.median(Ts) if Ts else '-'}")

# (b) N=4, M=5, S=60: simultaneous criterion D_max(k)<=Rcap (cond. (3)) at some step k<=kmax
print("(b) N=4, M=5, S=60: cumulative vs simultaneous, 100 trials (same scenes as grid)")
N, M, S = 4, 5, 60
for vs in (0, 5):
    cid = N * 1000 + M * 100 + S + vs * 7
    for meth in ('gwo_avg', 'gwo_group', 'hungarian', 'bottleneck'):
        for keep_all in ((False, True) if meth in ('hungarian', 'bottleneck') else (False,)):
            cum = sim = 0
            for t in range(100):
                p, q, vt = init_scene(cid * 100000 + t, N, M, S, vs)
                r = run_trial(meth, p, q, vt, seed=5_000_000 + cid * 1000 + t, simult_mode=True, keep_all=keep_all)
                cum += r['T'] is not None; sim += r['simult'] is not None
            tag = meth + (' (assign over all targets)' if keep_all else '')
            print(f"  vt={vs} {tag:38s} cumulative={cum:3d}%  simultaneous={sim:3d}%")
# also N=8 for reference
print("(b') N=8, M=5, S=60 simultaneous")
N = 8
for vs in (0, 5):
    cid = N * 1000 + M * 100 + S + vs * 7
    for meth in ('gwo_avg', 'gwo_group', 'hungarian', 'bottleneck'):
        for keep_all in ((False, True) if meth in ('hungarian', 'bottleneck') else (False,)):
            sim = 0
            for t in range(100):
                p, q, vt = init_scene(cid * 100000 + t, N, M, S, vs)
                r = run_trial(meth, p, q, vt, seed=5_000_000 + cid * 1000 + t, simult_mode=True, keep_all=keep_all)
                sim += r['simult'] is not None
            tag = meth + (' (assign over all targets)' if keep_all else '')
            print(f"  vt={vs} {tag:38s} simultaneous={sim:3d}%")
