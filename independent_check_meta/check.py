"""Расчёт 5. Вторая реализация ключевых ячеек — без импорта swarmsim, по формулам.

Зачем: защита от ошибок кода основного стенда. Код написан заново по формулам и иначе устроен: все
испытания считаются одновременно (массивы испытания × аппарат × координата), свой генератор случайных чисел,
свой порядок розыгрыша. Совпадение должно быть статистическим (одинаковые вероятности), а не испытание в испытание.
Ограничение: код написан тем же исполнителем, что и стенд; он ловит ошибки программирования, но не ошибки
прочтения формул.

Модель [1]: плоскость, Δt = 1 с, V_max = 10 м/с, a_max = 2 м/с², K = 30; аппараты в покое на дальности
U(85, 110) м от цели под случайными углами (интервал между аппаратами ≥ 10 м); порядок: p ← p + vΔt, затем
v ← clip(v + uΔt); захват — какой-либо аппарат на отсчёте не дальше 5 м. Регулятор u = clip(w_g(k)(X − p)),
w_g = a(k)/(2√2), a(k) = 2(1 − k/K). Без Boids.
  GWO:  X = (1/3) Σ_{l=1..3} [q − A_l ⊙ |C_l ⊙ q − p|], A = 2a r − a, C = 2r′ (покоординатно);
  WOA:  скаляры A, C на аппарат; l = (a₂ − 1) r + 1, a₂ = −1 − k/K; с вероятностью 1/2 окружение
        (|A| < 1: q − A|Cq − p|; иначе поиск X_rand − A|C X_rand − p|), иначе спираль |q − p| e^l cos 2πl + q;
  HHO:  по Heidari и др. (2019), область q ± 150 м, обрезка предложения по области.
Ячейки: сдвиг сцены s ∈ {0, 100, 300, 1000} м для N = 1; избыточность N ∈ {1, 2, 4} при сдвиге 0.
Сравнение — с results/meta (основной стенд), критерий: z-тест разности долей, |z| < 2,6 (≈ 1 % двусторонний).
Запуск (из корня репозитория):  python independent_check_meta/check.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd

K, DT, VMAX, AMAX, R = 30, 1.0, 10.0, 2.0, 5.0
LEVY_SIG = (math.gamma(2.5) * math.sin(math.pi * 0.75) / (math.gamma(1.25) * 1.5 * 2 ** 0.25)) ** (1 / 1.5)


def clip(x, lim):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, lim / np.maximum(n, 1e-12))


def start(T, N, rng):
    p = np.zeros((T, N, 2))
    for t in range(T):
        pts = []
        while len(pts) < N:
            r, a = rng.uniform(85, 110), rng.uniform(0, 2 * np.pi)
            c = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.hypot(*(c - o)) >= 10.0 for o in pts):
                pts.append(c)
        p[t] = pts
    return p


def prop_gwo(p, q, k, rng):
    a = 2 * (1 - k / K)
    acc = np.zeros_like(p)
    for _ in range(3):
        A = a * (2 * rng.random(p.shape) - 1)
        C = 2 * rng.random(p.shape)
        acc += q - A * np.abs(C * q - p)
    return acc / 3


def prop_woa(p, q, k, rng):
    T, N, _ = p.shape
    a = 2 * (1 - k / K)
    a2 = -1 - k / K
    A = (a * (2 * rng.random((T, N)) - 1))[..., None]
    C = (2 * rng.random((T, N)))[..., None]
    l = ((a2 - 1) * rng.random((T, N)) + 1)[..., None]
    enc = (rng.random((T, N)) < 0.5)[..., None]
    j = rng.integers(0, N, (T, N))
    xr = np.take_along_axis(p, j[..., None].repeat(2, -1), axis=1)
    search = xr - A * np.abs(C * xr - p)
    encircle = q - A * np.abs(C * q - p)
    spiral = np.abs(q - p) * np.exp(l) * np.cos(2 * np.pi * l) + q
    return np.where(enc, np.where(np.abs(A) >= 1, search, encircle), spiral)


def prop_hho(p, q, k, rng):
    T, N, _ = p.shape
    E = (2 * (1 - k / K) * (2 * rng.random((T, N)) - 1))[..., None]
    lb, ub = q - 150.0, q + 150.0
    xm = p.mean(axis=1, keepdims=True)
    j = rng.integers(0, N, (T, N))
    xr = np.take_along_axis(p, j[..., None].repeat(2, -1), axis=1)
    u = lambda: rng.random((T, N))[..., None]
    ex1 = xr - u() * np.abs(xr - 2 * u() * p)
    ex2 = (q - xm) - u() * ((ub - lb) * u() + lb)
    explore = np.where(u() < 0.5, ex1, ex2)
    J = 2 * (1 - u())
    hard = q - E * np.abs(q - p)
    soft = (q - p) - E * np.abs(J * q - p)
    base = np.where(np.abs(E) >= 0.5, p, xm)
    y = q - E * np.abs(J * q - base)
    lev = 0.01 * rng.normal(size=p.shape) * LEVY_SIG / np.abs(rng.normal(size=p.shape)) ** (1 / 1.5)
    z = y + rng.random(p.shape) * lev
    f0 = np.linalg.norm(p - q, axis=-1, keepdims=True)
    dive = np.where(np.linalg.norm(y - q, axis=-1, keepdims=True) < f0, y,
                    np.where(np.linalg.norm(z - q, axis=-1, keepdims=True) < f0, z, p))
    r = u()
    exploit = np.where(r >= 0.5, np.where(np.abs(E) < 0.5, hard, soft), dive)
    x = np.where(np.abs(E) >= 1, explore, exploit)
    return np.clip(x, lb, ub)


SCHEME = os.environ.get("SWARMSIM_INTEGRATOR", "explicit")  # та же переменная, что у стенда
PROPS = dict(gwo=prop_gwo, woa=prop_woa, hho=prop_hho)


def run(law, T, N, shift, seed):
    rng = np.random.default_rng(seed)
    off = shift * np.array([1.0, 1.0]) / np.sqrt(2)
    p = start(T, N, rng) + off
    q = np.broadcast_to(off, (T, 1, 2)).copy()
    v = np.zeros_like(p)
    cap = np.zeros(T, dtype=bool)
    for k in range(K + 1):
        cap |= (np.linalg.norm(p - q, axis=-1) <= R).any(axis=1)
        if k == K:
            break
        x = PROPS[law](p, q, k, rng)
        u = clip(2 * (1 - k / K) / (2 * np.sqrt(2)) * (x - p), AMAX)
        if SCHEME == "semi_implicit":
            v = clip(v + DT * u, VMAX)
            p = p + DT * v
        else:
            p = p + DT * v
            v = clip(v + DT * u, VMAX)
    return cap.mean()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    meta = os.path.join(here, "..", os.environ.get("META_OUT", "results/meta"))
    T = 4000
    sh = pd.read_csv(os.path.join(meta, "shift_summary.csv"))
    hh = pd.read_csv(os.path.join(meta, "hho_shift.csv"))
    red = pd.read_csv(os.path.join(meta, "redundancy_summary.csv"))
    hred = pd.read_csv(os.path.join(meta, "hho_redundancy.csv"))
    rows = []

    def ref_shift(law, s):
        src = hh if law == "hho" else sh
        return src[(src.law == law) & (src.n == 1) & (src.vt == 0) & (src["shift"] == s)].cap.iloc[0], 1000

    def ref_red(law, n):
        src = hred if law == "hho" else red
        g = src[(src.law == law) & (src.vt == 0)]
        if n == 1:
            return g.p1.iloc[0], 1000
        return g[(g.n == n) & (~g.boids.astype(bool))].obs.iloc[0], 1000

    for i, law in enumerate(("gwo", "woa", "hho")):
        for j, s in enumerate((0.0, 100.0, 300.0, 1000.0)):
            rows.append(("сдвиг", law, 1, s, 100 * run(law, T, 1, s, 1000 + 10 * i + j), *ref_shift(law, s)))
        for j, n in enumerate((2, 4)):
            rows.append(("избыточность", law, n, 0.0, 100 * run(law, T, n, 0.0, 2000 + 10 * i + j), *ref_red(law, n)))
    df = pd.DataFrame(rows, columns=["опыт", "закон", "N", "сдвиг", "вторая_реализация", "стенд", "n_стенд"])
    p1, p2 = df["вторая_реализация"] / 100, df["стенд"] / 100
    pp = (p1 * T + p2 * 1000) / (T + 1000)
    df["z"] = (p1 - p2) / np.sqrt(np.maximum(pp * (1 - pp), 1e-9) * (1 / T + 1 / 1000))
    df["совпадает"] = df.z.abs() < 2.6
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("ячеек:", len(df), "расхождений |z| ≥ 2,6:", int((~df["совпадает"]).sum()))
    df.to_csv(os.path.join(here, "out_check.csv" if SCHEME == "explicit" else f"out_check_{SCHEME}.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
