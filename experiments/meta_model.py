"""Расчёт 1. Полуаналитическая модель вероятности захвата одним аппаратом (неподвижная цель).

Идея. Детерминированная часть законов вида u = clip(w_g (X − p)) при E[X] = q — центральная сила: ускорение
направлено на цель. Центральная сила не меняет момент L = e × v (e = p − q) в непрерывном времени, поэтому без шума
аппарат идёт точно в цель (закон attract ловит 100 % при Δt = 1 с). Промах создаёт только поперечный шум ζ —
составляющая ускорения, перпендикулярная линии визирования. При явной схеме p ← p + vΔt, v ← clip(v + uΔt)
момент меняется точно так:
    L_{k+1} = s_k [ g_k L_k + Δt x_{k+1} ζ_k ],   g_k = 1 + κ_k Δt²,
а при полунеявной схеме (v ← clip(v + uΔt), затем p ← p + vΔt) центральная сила момент не меняет:
    L_{k+1} = s_k [ L_k + Δt x_k ζ_k ]
(схема берётся из cfg.integrator, т. е. из переменной окружения SWARMSIM_INTEGRATOR),
где x — продольная координата (по линии визирования), κ_k = (центральная составляющая ускорения)/дальность
(в насыщении a_max/r, вне насыщения w_g), s_k ≤ 1 — множитель ограничения скорости. Пролёт мимо цели со скоростью V
даёт промах b = |L|/V; отсчёт, попавший на продольную координату x* при пролёте, засчитывает захват, если
x*² + b² ≤ R², т. е. |L| ≤ V √(R² − x*²).

Допущения модели (проверяются сравнением со стендом):
  1) продольное движение детерминировано: x, v_x — по среднему ускорению E[u_∥];
  2) моменты ζ (среднее и дисперсия) считаются при нулевом поперечном отклонении, по одношаговой квадратуре
     распределения предложения X (метод Монте-Карло по случайным коэффициентам A, C, r, l… — не по траекториям);
  3) ζ_k независимы между шагами (так и есть: коэффициенты разыгрываются заново на каждом шаге), L — гауссова
     случайная величина (вариант 'gauss') или сумма с эмпирическим распределением ζ (вариант 'emp');
  4) в g_k дальность r_k = √(x_k² + E[L_k²]/v_k²).
Вероятность захвата за горизонт K: 1 − P(|L_i| > T_i для всех отсчётов-кандидатов i с |x_i| < R), T_i = |v_i| √(R² − x_i²).
Усреднение: начальная дальность D ~ U(85, 110) м (сетка), направление φ ~ U(0, 2π) (сетка).

Законы: gwo, gwo_rel, woa, woa_spiral, woa_encircle, hho, hho_hard, attract (у pso в предложении есть скорость —
другая структура, в модель не входит).
Запуск:  python -m experiments.meta_model   (p₁ по законам в базовой постановке);
         сравнение со стендом — experiments.meta_model_check → results/meta/model_vs_sim.csv.
Параметр along=True — вариант с линеаризованным продольным разбросом (проверка чувствительности, в сводку не входит).
"""
import os
import numpy as np
import pandas as pd

from swarmsim import metaguide as mg
from swarmsim.config import SCENARIO_1

OUT = os.environ.get("META_OUT", "results/meta")


# --- одношаговые выборки предложения X для одиночного аппарата (N = 1: X_rand = X_m = p) ---------------------
def _uni(rng, n):
    return rng.random(n)


def proposal(law, p, q, k, rng, cfg):
    n, d = p.shape
    a = mg._a(k, cfg)
    if law in ("gwo", "gwo_rel"):
        x = np.zeros_like(p)
        for _ in range(3):
            A = 2 * a * rng.random((n, d)) - a
            C = 2 * rng.random((n, d))
            x += q - A * (np.abs(C * q - p) if law == "gwo" else np.abs(C * (q - p)))
        return x / 3.0
    if law.startswith("woa"):
        ps = {"woa": 0.5, "woa_spiral": 1.0, "woa_encircle": 0.0}[law]
        a2 = -1.0 - k / cfg.gwo_k_max
        A = (2 * a * _uni(rng, n) - a)[:, None]
        C = (2 * _uni(rng, n))[:, None]
        l = ((a2 - 1.0) * _uni(rng, n) + 1.0)[:, None]
        enc = (_uni(rng, n) < 1.0 - ps)[:, None]
        srch = np.abs(A) >= 1.0
        x_s = p - A * np.abs(C * p - p)            # поиск: X_rand = p при N = 1
        x_e = q - A * np.abs(C * q - p)
        x_sp = np.abs(q - p) * np.exp(l) * np.cos(2 * np.pi * l) + q
        return np.where(enc, np.where(srch, x_s, x_e), x_sp)
    if law in ("hho", "hho_hard"):
        E = (a * (2 * _uni(rng, n) - 1))[:, None]
        if law == "hho_hard":
            return q - E * np.abs(q - p)
        box = 150.0
        lb, ub = q - box, q + box
        aE = np.abs(E)
        r_ex = (_uni(rng, n) < 0.5)[:, None]
        x1 = p - _uni(rng, n)[:, None] * np.abs(p - 2 * _uni(rng, n)[:, None] * p)
        x2 = (q - p) - _uni(rng, n)[:, None] * ((ub - lb) * _uni(rng, n)[:, None] + lb)
        x_ex = np.where(r_ex, x1, x2)
        r = _uni(rng, n)[:, None]
        J = (2 * (1 - _uni(rng, n)))[:, None]
        hard = q - E * np.abs(q - p)
        soft = (q - p) - E * np.abs(J * q - p)
        y = q - E * np.abs(J * q - p)                # B = X = X_m при N = 1
        f0 = np.linalg.norm(p - q, axis=1, keepdims=True)
        lev = 0.01 * rng.normal(size=(n, d)) * mg._LEVY_SIGMA / np.abs(rng.normal(size=(n, d))) ** (1 / 1.5)
        z = y + rng.random((n, d)) * lev
        fy = np.linalg.norm(y - q, axis=1, keepdims=True)
        fz = np.linalg.norm(z - q, axis=1, keepdims=True)
        dive = np.where(fy < f0, y, np.where(fz < f0, z, p))
        x_in = np.where(r >= 0.5, np.where(aE < 0.5, hard, soft), dive)
        x = np.where(aE >= 1, x_ex, x_in)
        return np.clip(x, lb, ub)
    if law == "attract":
        return np.repeat(q[None, :], n, axis=0)
    raise ValueError(law)


def u_moments(law, e, q, k, cfg, rng, n_mc):
    p = np.repeat((q + e)[None, :], n_mc, axis=0)
    x = proposal(law, p, q, k, rng, cfg)
    u = mg._wg(k, cfg) * (x - p)
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    return np.where(nrm > cfg.a_max, u * (cfg.a_max / np.maximum(nrm, 1e-12)), u)


def trajectory(law, cfg, q, phi, D, n_mc=2000, n_l=4000, seed=0, mode="gauss", with_cdf=False, along=False):
    """Вероятность захвата для одного (φ, D). Возвращает (p_capture, число пролётов-кандидатов)."""
    rng = np.random.default_rng(seed)
    eh = np.array([np.cos(phi), np.sin(phi)])
    nh = np.array([-np.sin(phi), np.cos(phi)])
    dt, K, R, V = cfg.dt, cfg.t_max, cfg.r_capture, cfg.v_max
    semi = cfg.integrator == "semi_implicit"
    x, vx = D, 0.0
    L = np.zeros(n_l)                       # выборка L (гаусс или эмпирика) — дешёвая скалярная рекурсия
    alive = np.ones(n_l, dtype=bool)        # ещё не захвачен
    cand = 0
    F = np.zeros(K + 1)                     # доля захваченных к отсчёту k
    dx = np.zeros(n_l)                      # along=True: продольный разброс около средней траектории
    dv = np.zeros(n_l)
    for k in range(K + 1):
        xs = x + dx if along else np.full(n_l, x)
        vs = np.abs(vx + dv) if along else np.full(n_l, abs(vx))
        inside = np.abs(xs) <= R
        if inside.any():
            cand += 1
            T = np.where(vs > 1e-9, vs * np.sqrt(np.maximum(R * R - xs * xs, 0.0)), np.inf)
            alive &= ~(inside & (np.abs(L) <= T))
        F[k] = 1.0 - alive.mean()
        if k == K:
            break
        u = u_moments(law, x * eh, q, k, cfg, rng, n_mc)
        up, un = u @ eh, u @ nh
        abar = up.mean()
        wg = mg._wg(k, cfg)
        free = min(cfg.a_max, wg * abs(x))                  # центральная составляющая без шума
        eff = (-np.sign(x) * abar) / free if free > 1e-12 else 1.0
        r = np.sqrt(x * x + (L * L).mean() / max(vx * vx, 1.0))
        kap = eff * min(wg, cfg.a_max / max(r, 1e-9))
        vx_new = vx + dt * abar
        s = min(1.0, V / abs(vx_new)) if abs(vx_new) > 1e-12 else 1.0
        if mode == "gauss":
            z = un.mean() + un.std() * rng.standard_normal(n_l)
        else:
            z = un[rng.integers(0, n_mc, n_l)]
        if semi:
            # полунеявная схема: v ← s(v + Δt u), затем p ← p + Δt v; центральная сила момент не меняет,
            # плечо — продольная координата на текущем шаге: L_{k+1} = s_k [L_k + Δt x_k ζ_k]
            x_next = x + dt * s * vx_new
            L = s * (L + dt * x * z)
        else:
            x_next = x + dt * vx
            L = s * ((1 + kap * dt * dt) * L + dt * x_next * z)
        if along:
            xi = (up - abar)[rng.integers(0, n_mc, n_l)]
            if semi:
                dv = s * (dv + dt * xi)
                dx = dx + dt * dv
            else:
                dx = dx + dt * dv
                dv = s * (dv + dt * xi)
        x, vx = x_next, s * vx_new
    if with_cdf:
        return F
    return 1.0 - alive.mean(), cand


def cdf1(law, cfg, shift=0.0, phis=None, Ds=None, mode="gauss", n_mc=2000, along=False):
    """Модельная F_1(k) — доля захватов к итерации k (усреднение по φ и D)."""
    q = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)
    lo, hi = cfg.start_dist_range
    phis = np.deg2rad(np.arange(0, 360, 7.5)) if phis is None else phis
    Ds = lo + (np.arange(25) + 0.5) * (hi - lo) / 25 if Ds is None else Ds
    Fs = [trajectory(law, cfg, q, ph, D, n_mc=n_mc, seed=i, mode=mode, with_cdf=True, along=along)
          for i, (ph, D) in enumerate((ph, D) for ph in phis for D in Ds)]
    return np.mean(Fs, axis=0)


def p1(law, cfg, shift=0.0, phis=None, Ds=None, mode="gauss", n_mc=2000, along=False):
    q = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)
    lo, hi = cfg.start_dist_range
    phis = np.deg2rad(np.arange(0, 360, 7.5)) if phis is None else phis
    Ds = lo + (np.arange(25) + 0.5) * (hi - lo) / 25 if Ds is None else Ds   # середины ячеек: фаза отсчёта равномерна
    vals = [trajectory(law, cfg, q, ph, D, n_mc=n_mc, seed=i, mode=mode, along=along)[0]
            for i, (ph, D) in enumerate((ph, D) for ph in phis for D in Ds)]
    return float(np.mean(vals))


if __name__ == "__main__":
    import time
    cfg = SCENARIO_1.with_(n_agents=1)
    for law in ["attract", "gwo", "gwo_rel", "woa", "woa_spiral", "woa_encircle", "hho", "hho_hard"]:
        t = time.time()
        print(law, round(100 * p1(law, cfg), 1), round(time.time() - t, 1), "c", flush=True)
