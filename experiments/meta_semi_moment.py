"""Численная проверка рекуррентности момента L = e × v (e = p − q) для двух схем интегрирования.

Одиночный аппарат, закон gwo, неподвижная цель. Применённое ускорение (после ограничения a_max) раскладывается на
радиальную и поперечную составляющие: u_k = −κ_k e_k + ζ_k n_k, n_k — единичный вектор, перпендикулярный e_k;
s_k = min(1, V_max / |v_k + Δt u_k|) — множитель ограничения скорости. Проверяются тождества
  полунеявная:  L_{k+1} = s_k [ L_k + Δt (e_k × n_k) ζ_k ],
  явная:        L_{k+1} = s_k [ (1 + κ_k Δt²) L_k + Δt (e_{k+1} × n_k) ζ_k ].
Перекрёстно: формула одной схемы, применённая к траекториям другой (должна не выполняться).
Траектории строятся собственным циклом, повторяющим sim.run_trial, и сверяются с run_trial(record=True).
Сцена сдвигается на s·(1, 1)/√2, как в meta_shift. Результат: results/meta_semi/moment_check.csv, moment_check.txt.
Запуск:  python -m experiments.meta_semi_moment
"""
import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401
from swarmsim import sim
from swarmsim.config import SCENARIO_1
from swarmsim.guidance import LAWS
from swarmsim.scenario import make_initial

OUT = "results/meta_semi"
SEEDS = range(25)
SHIFTS = [0.0, 300.0, 10000.0]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _mk(off):
    def mk(cfg, seed):
        p, v, q, vq, rng = make_initial(cfg, seed)
        return p + off, v, q + off, vq, rng
    return mk


def trajectory(cfg, seed, shift, law="gwo"):
    off = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)
    p, v, q, vq, rng = _mk(off)(cfg, seed)
    steps, P = [], [p.copy()]
    for k in range(cfg.t_max + 1):
        if np.linalg.norm(p - q, axis=1).min() <= cfg.r_capture or k == cfg.t_max:
            break
        u = sim._clip_norm(LAWS[law](p, v, q, vq, k, rng, cfg), cfg.a_max)
        w = v + cfg.dt * u
        s = min(1.0, cfg.v_max / max(np.linalg.norm(w[0]), 1e-12))
        if cfg.integrator == "semi_implicit":
            v1 = sim._clip_norm(w, cfg.v_max)
            p1 = p + cfg.dt * v1
        else:
            p1 = p + cfg.dt * v
            v1 = sim._clip_norm(w, cfg.v_max)
        steps.append(dict(k=k, p=p[0].copy(), v=v[0].copy(), u=u[0].copy(), s=s, p1=p1[0].copy(), v1=v1[0].copy(), q=q.copy()))
        p, v = p1, v1
        q = q + cfg.dt * vq
        P.append(p.copy())
    # сверка со стендом
    sim.make_initial = _mk(off)
    ref = sim.run_trial(cfg, law, False, seed, record=True)
    sim.make_initial = make_initial
    assert ref["P"].shape == np.array(P).shape and np.array_equal(ref["P"], np.array(P)), "траектория не совпала со стендом"
    return steps


def check(steps, dt):
    rows = []
    for st in steps:
        e, v, u, s, e1, v1 = st["p"] - st["q"], st["v"], st["u"], st["s"], st["p1"] - st["q"], st["v1"]
        r2 = e @ e
        kap = -(u @ e) / r2
        n = np.array([-e[1], e[0]]) / np.sqrt(r2)
        zeta = u @ n
        L, L1 = cross(e, v), cross(e1, v1)
        semi = s * (L + dt * cross(e, n) * zeta)
        expl = s * ((1 + kap * dt * dt) * L + dt * cross(e1, n) * zeta)
        scale = abs(L) + dt * np.sqrt(r2) * abs(zeta) + 1e-9          # масштаб слагаемых
        rows.append(dict(k=st["k"], dist=np.sqrt(r2), L=L, L1=L1, kappa=kap, kappa_dt2=kap * dt * dt, zeta=zeta, s=s,
                         decomp_err=np.linalg.norm(u - (-kap * e + zeta * n)),
                         res_semi=abs(L1 - semi) / scale, res_expl=abs(L1 - expl) / scale,
                         abs_res_semi=abs(L1 - semi), abs_res_expl=abs(L1 - expl)))
    return rows


def main():
    rows = []
    for integ in ["explicit", "semi_implicit"]:
        cfg = SCENARIO_1.with_(n_agents=1, integrator=integ)
        for sh in SHIFTS:
            for sd in SEEDS:
                for r in check(trajectory(cfg, sd, sh), cfg.dt):
                    r.update(integrator=integ, shift=sh, seed=sd)
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/moment_check.csv", index=False)
    g = df.groupby("integrator")
    summ = pd.DataFrame({
        "trajectories": g.apply(lambda h: h[["shift", "seed"]].drop_duplicates().shape[0]),
        "steps": g.size(),
        "max_decomp_err": g.decomp_err.max(),
        "semi_formula_max_rel": g.res_semi.max(),
        "semi_formula_median_rel": g.res_semi.median(),
        "expl_formula_max_rel": g.res_expl.max(),
        "expl_formula_median_rel": g.res_expl.median(),
        "steps_s<1_%": g.s.apply(lambda x: 100 * (x < 1).mean()),
        "kappa_dt2_median": g.kappa_dt2.median(),
        "kappa_dt2_p90": g.kappa_dt2.quantile(0.9),
        "kappa_dt2_max": g.kappa_dt2.max(),
    })
    # роль множителя (1 + κΔt²) в явной схеме: какая доля |L_{k+1}| приходится на усиление κΔt² L_k
    ex = df[df.integrator == "explicit"]
    amp = (ex.s * ex.kappa * ex.L).abs() / (ex.L1.abs() + 1e-9)
    pd.set_option("display.width", 250)
    txt = (summ.T.to_string() + "\n\nЯвная схема: |s κ Δt² L_k| / |L_(k+1)| — медиана "
           f"{amp.median():.3f}, 90-й перцентиль {amp.quantile(0.9):.3f}\n")
    # по шагам: κ вдали (насыщение a_max) и вблизи цели
    near = df[df.dist < 30].groupby("integrator").kappa.median()
    far = df[df.dist > 60].groupby("integrator").kappa.median()
    txt += f"κ (Δt = 1 с), медиана: дальность > 60 м — {far.to_dict()}; < 30 м — {near.to_dict()}\n"
    print(txt)
    with open(f"{OUT}/moment_check.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
