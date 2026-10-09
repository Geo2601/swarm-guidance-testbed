"""Несколько целей: факторная сетка «распределение/наведение × Boids × N × M × разброс × скорость».

Запуск:  python -m experiments.multi_target --seeds 100 --jobs 8
Выход:   results/multi_trials.csv (по испытаниям), results/multi_summary.csv, results/multi_tests.csv
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from swarmsim.config import Config
from swarmsim.multi import run_trial_multi
from swarmsim.stats import holm, mcnemar_exact

METHODS = ["gwo_avg", "gwo_group", "gwo_assign", "greedy", "hungarian", "bottleneck", "ring"]
BOIDS = {"none": (False, {}), "paper": (True, {}), "safe": (True, dict(sep_law="inverse", k_sep=100.0))}


def cells(quick=False):
    for n, m, spread, vt in itertools.product([4, 8], [1, 2, 3, 5], [25.0, 60.0], [0.0, 5.0]):
        if m == 1 and spread != 25.0:
            continue
        yield dict(n=n, m=m, spread=spread, vt=vt)


def job(args):
    cell, method, bk, seeds, t_max = args
    use, kw = BOIDS[bk]
    cfg = Config(n_agents=cell["n"], target_speed=cell["vt"], t_max=t_max, gwo_k_max=t_max, **kw)
    rows = []
    for s in seeds:
        r = run_trial_multi(cfg, method, use, s, m=cell["m"], spread=cell["spread"])
        rows.append(dict(**cell, method=method, boids=bk, t_max=t_max, seed=s, **r))
    return rows


def summarize(df):
    df = df.assign(_tm=df["t_max"])
    g = df.groupby(["t_max", "n", "m", "spread", "vt", "boids", "method"])
    out = g.apply(lambda x: pd.Series(dict(
        all_cap=100 * x.all_captured.mean(),
        frac_cap=100 * x.frac_captured.mean(),
        simult=100 * x.simultaneous.mean(),
        t_med=x.loc[x.all_captured, "t_cap"].median(),
        t_cens=x.t_cap.fillna(x._tm.iloc[0] + 1).median(),
        viol=100 * x.collision.mean(),
        viol_pre=100 * x.collision_pre.mean())), include_groups=False).reset_index()
    return out


def tests(df):
    """Парные тесты в каждой ячейке: bottleneck против gwo_avg по доле полного захвата и нарушениям."""
    rows = []
    for key, x in df.groupby(["t_max", "n", "m", "spread", "vt", "boids"]):
        a = x[x.method == "bottleneck"].set_index("seed")
        b = x[x.method == "gwo_avg"].set_index("seed")
        for col in ("all_captured", "collision"):
            r = mcnemar_exact(a[col].to_numpy(bool), b.loc[a.index, col].to_numpy(bool))
            rows.append(dict(zip(["t_max", "n", "m", "spread", "vt", "boids"], key), measure=col,
                             bottleneck_only=r["b"], gwo_only=r["c"], p=r["p"]))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    return t


def from_csv(path):
    """Пересчитать сводку и тесты из сохранённых испытаний."""
    f = f"{path}/multi_trials.csv"
    df = pd.read_csv(f if os.path.exists(f) else f + ".gz")
    summarize(df).to_csv(f"{path}/multi_summary.csv", index=False)
    tests(df).to_csv(f"{path}/multi_tests.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--t-max", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--out", default="results")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    work = [(c, mth, bk, seeds, tm) for tm in a.t_max for c in cells() for mth in a.methods for bk in BOIDS]
    with Pool(a.jobs) as pool:
        rows = [r for chunk in pool.imap_unordered(job, work, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/multi_trials.csv", index=False)
    s = summarize(df)
    s.to_csv(f"{a.out}/multi_summary.csv", index=False)
    if {"bottleneck", "gwo_avg"} <= set(a.methods):
        t = tests(df)
        t.to_csv(f"{a.out}/multi_tests.csv", index=False)
    print("испытаний:", len(df), "| ячеек:", len(s))


if __name__ == "__main__":
    main()
