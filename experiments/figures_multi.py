"""Рисунки по нескольким целям.
  fig_multi_capture.png — доля испытаний с захватом всех целей в зависимости от числа целей
  fig_multi_traj.png    — траектории одного испытания: GWO (формула (5) из [1]) против bottleneck и ring
Запуск: python -m experiments.figures_multi  (после experiments.multi_target)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swarmsim.config import Config
from swarmsim.multi import run_trial_multi

NAMES = {("gwo_avg", "paper"): "GWO+Boids (формула 5 из [1])", ("gwo_group", "paper"): "GWO+Boids, группы по центроидам",
         ("gwo_assign", "paper"): "GWO+Boids, назначенная цель", ("greedy", "none"): "Упреждение, ближайшая цель",
         ("hungarian", "none"): "Упреждение, венгерский (сумма)", ("bottleneck", "none"): "Упреждение, bottleneck (max)",
         ("ring", "none"): "Bottleneck + кольцо ролей"}
COL = {"gwo_avg": "#c0392b", "gwo_group": "#e67e22", "gwo_assign": "#d4a017", "greedy": "#7f8c8d",
       "hungarian": "#2980b9", "bottleneck": "#1f3a93", "ring": "#27ae60"}


def main(src="results/multi", out="results/figures"):
    os.makedirs(out, exist_ok=True)
    s = pd.read_csv(f"{src}/multi_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for ax, (n, vt) in zip(axs.flat, [(4, 0.0), (4, 5.0), (8, 0.0), (8, 5.0)]):
        for (mth, bk), name in NAMES.items():
            x = s[(s.t_max == 30) & (s.n == n) & (s.vt == vt) & (s.method == mth) & (s.boids == bk)
                  & ((s.m == 1) | (s.spread == 60.0))].sort_values("m")
            ax.plot(x.m, x.all_cap, marker="o", color=COL[mth], label=name, lw=2 if mth in ("gwo_avg", "bottleneck") else 1.2)
        ax.set_title(f"{n} аппарата · цели {vt:g} м/с · разброс 60 м · k_max = 30".replace("8 аппарата", "8 аппаратов"), fontsize=10)
        ax.set_xticks([1, 2, 3, 5]); ax.grid(alpha=.3); ax.set_ylim(-3, 103)
        ax.set_xlabel("Число целей M")
    axs[0, 0].set_ylabel("Захвачены все цели, % испытаний"); axs[1, 0].set_ylabel("Захвачены все цели, % испытаний")
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{out}/fig_multi_capture.png", dpi=130)

    cfg = Config(n_agents=4)
    seed = 3
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (mth, b, title) in zip(axs, [("gwo_avg", True, "GWO+Boids"), ("bottleneck", False, "Упреждение + bottleneck"),
                                         ("ring", False, "Bottleneck + кольцо ролей")]):
        r = run_trial_multi(cfg, mth, b, seed, m=3, spread=60.0, record=True)
        P, Q = r["P"], r["Q"]
        for i in range(P.shape[1]):
            ax.plot(P[:, i, 0], P[:, i, 1], color="#2c4fa3", lw=1)
            ax.scatter(P[-1, i, 0], P[-1, i, 1], color="#2c4fa3", marker="x", s=40, zorder=3)
        for j in range(Q.shape[1]):
            ok = r["T_m"][j] >= 0
            ax.scatter(Q[0, j, 0], Q[0, j, 1], marker="^", s=80, color="#27ae60" if ok else "#d62728", zorder=4)
            ax.add_patch(plt.Circle(Q[0, j], 5, fill=False, color="#999", lw=.8))
        if mth == "gwo_avg":
            c = Q[0].mean(0)
            ax.scatter(*c, marker="+", s=120, color="k"); ax.annotate("среднее центроидов", c, (c[0] + 8, c[1] - 14), fontsize=8)
        res = f"захвачено {int((r['T_m'] >= 0).sum())} из 3" + (f", k = {int(r['t_cap'])}" if r["all_captured"] else ", к k = 30 не все")
        ax.set_title(f"{title}\n{res}", fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlim(-125, 125); ax.set_ylim(-125, 125)
    fig.suptitle("Одно и то же испытание (сид 3): 4 аппарата, 3 неподвижные цели в круге 60 м; зелёный — цель захвачена", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig_multi_traj.png", dpi=120)
    print("рисунки в", out)


if __name__ == "__main__":
    main()
