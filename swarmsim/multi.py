"""Несколько целей.

Модель нескольких целей по работе [1] (с. 4–5, шаги 1–5 и п. 4 «Критерии захвата»):
  * цели q_1..q_M; при M >= 3 — K-means (K = 3), центроиды c_1, c_2, c_3; при M < 3 цели «дублируются»;
  * каждый аппарат (включая лидеров) считает три предложения X^(i) = c_i − A ⊙ |C ⊙ c_i − p_j|
    и усредняет их, формула (5); роли α, β, δ в формулу не входят;
  * при этом в тексте шага 2 сказано «рой делится на несколько групп» — второе прочтение, при котором
    каждая группа идёт к своему центроиду;
  * захват цели m — первый момент T_m, когда до неё ближе R_cap хотя бы один аппарат; T_cap = max T_m
    («накопительный» критерий). Условие завершения (3) и шаг 7.2 требуют, чтобы все цели были в радиусе
    захвата одновременно («одновременный» критерий). Считаются оба.

Методы распределения целей и наведения (guidance):
  gwo_avg      — формула (5) из [1] буквально: среднее по трём центроидам;
  gwo_group    — прочтение шага 2: аппараты делятся между центроидами (сбалансированно, венгерским методом),
                 каждый идёт к своему центроиду по формуле GWO;
  gwo_assign   — минимальная правка: GWO-формула, но «добыча» — назначенная аппарату цель (венгерский метод);
  greedy       — каждый аппарат: упреждение на ближайшую по времени встречи незахваченную цель;
  hungarian    — упреждение; назначение минимизирует сумму времён встречи (сбалансированно);
  bottleneck   — упреждение; назначение минимизирует максимальное время встречи — это прямо целевая
                 функция из [1] T_cap = max T_m; среди таких — минимальная сумма.
  ring         — распределение ролей: на каждую цель один «ведущий» аппарат (как в bottleneck), остальные
                 занимают места на кольце радиуса ring_radius вокруг цели, равномерно по углу от направления
                 подхода ведущего; места назначаются венгерским методом. Кольцо окружения задаётся явно,
                 а не возникает из сил Boids.
Назначения пересчитываются на каждой итерации по незахваченным целям.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .boids import boids_accel, post_separate
from .scenario import make_initial

BIG = 1e6


def _clip_norm(x, lim):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > lim, x * (lim / np.maximum(n, 1e-12)), x)


def _min_pair(p):
    if p.shape[0] < 2:
        return np.inf
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(p.shape[0], 1)
    return d[iu].min()


# ---------- геометрия встречи ----------

def intercept_time(p, q, vq, V):
    """Время встречи с целью (q, vq) из точки p при скорости V; матрица N×M. Без решения — BIG."""
    r = q[None, :, :] - p[:, None, :]                  # N×M×d
    a = (vq * vq).sum(-1)[None, :] - V * V             # 1×M
    b = 2 * (r * vq[None, :, :]).sum(-1)
    c = (r * r).sum(-1)
    t = np.full(c.shape, BIG)
    lin = np.abs(a) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = np.where(b < 0, -c / b, BIG)
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
    t1 = np.where(t1 > 1e-9, t1, BIG)
    t2 = np.where(t2 > 1e-9, t2, BIG)
    tq = np.where(disc >= 0, np.minimum(t1, t2), BIG)
    t = np.where(lin, np.broadcast_to(tl, c.shape), tq)
    return t


def intercept_vdes(p, q, vq, V):
    """Желаемая скорость на точку встречи для пар (p_i, q_i, vq_i); без решения — на текущее положение."""
    r = q - p
    t = np.array([intercept_time(p[i:i + 1], q[i:i + 1], vq[i:i + 1], V)[0, 0] for i in range(len(p))])
    ok = t < BIG / 2
    v = np.where(ok[:, None], (r + vq * np.where(ok, t, 1.0)[:, None]) / np.where(ok, t, 1.0)[:, None], 0.0)
    n = np.linalg.norm(r, axis=1, keepdims=True)
    v = np.where(ok[:, None], v, V * r / np.maximum(n, 1e-12))
    return v


# ---------- назначение ----------

def _balanced_columns(n_agents, m):
    """Повтор столбцов, чтобы каждый аппарат получил цель, а цели были покрыты равномерно."""
    reps = int(np.ceil(n_agents / m))
    return np.tile(np.arange(m), reps)


def assign_sum(cost):
    """Минимум суммы; сбалансированно (каждая цель — примерно N/M аппаратов). Возвращает индекс цели на аппарат."""
    n, m = cost.shape
    cols = _balanced_columns(n, m)
    rows, cidx = linear_sum_assignment(cost[:, cols])
    out = np.empty(n, dtype=int)
    out[rows] = cols[cidx]
    return out


def assign_bottleneck(cost):
    """Минимум максимального времени по целям (каждая цель получает хотя бы одного аппарата, если N >= M),
    затем минимум суммы при этом пороге; лишние аппараты — сбалансированно по сумме."""
    n, m = cost.shape
    if n < m:
        return assign_sum(cost)
    vals = np.unique(cost)
    lo, hi = 0, len(vals) - 1
    while lo < hi:  # минимальный порог, при котором существует паросочетание, покрывающее все цели
        mid = (lo + hi) // 2
        g = csr_matrix((cost.T <= vals[mid]).astype(int))   # цели × аппараты
        match = maximum_bipartite_matching(g, perm_type="column")
        if (match >= 0).sum() == m:
            hi = mid
        else:
            lo = mid + 1
    tau = vals[lo]
    c1 = np.where(cost <= tau, cost, BIG)
    rows, cols = linear_sum_assignment(c1.T)               # каждой цели — свой аппарат
    out = np.full(n, -1)
    out[cols] = rows
    rest = np.where(out < 0)[0]
    if len(rest):
        out[rest] = assign_sum(cost[rest])
    return out


# ---------- K-means для центроидов [1] ----------

def kmeans3(x, init):
    c = init.copy()
    for _ in range(50):
        lab = np.argmin(((x[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(3)])
        if np.allclose(new, c):
            break
        c = new
    return c


def centroids(qa, state, dup_rule="first", km_init="farthest", rng=None):
    """Три центроида по [1]: M >= 3 — K-means (тёплый старт с прошлой итерации), M < 3 — дублирование.
    dup_rule: 'first' — [q1, q2, q1]; 'last' — [q1, q2, q2]; 'mean' — [q1, q2, (q1+q2)/2] (не дублирование, контроль).
    km_init: 'farthest' — детерминированная «дальняя» инициализация; 'random' — случайные три цели (каждую итерацию)."""
    m = len(qa)
    if m >= 3:
        if km_init == "random":
            state["c"] = kmeans3(qa, qa[rng.choice(m, 3, replace=False)])
            return state["c"]
        if state.get("c") is None or len(state["c"]) != 3:
            # детерминированная инициализация: самая дальняя точка от центра и далее «дальние»
            idx = [int(np.argmax(((qa - qa.mean(0)) ** 2).sum(1)))]
            for _ in range(2):
                d = np.min(((qa[:, None, :] - qa[idx][None, :, :]) ** 2).sum(-1), axis=1)
                idx.append(int(np.argmax(d)))
            state["c"] = qa[idx]
        state["c"] = kmeans3(qa, state["c"])
        return state["c"]
    if m == 2 and dup_rule == "last":
        return qa[[0, 1, 1]]
    if m == 2 and dup_rule == "mean":
        return np.array([qa[0], qa[1], qa.mean(0)])
    return qa[np.resize(np.arange(m), 3)]  # [q1, q2, q1] или [q1, q1, q1]


def gwo_point(p, leaders, k, rng, cfg):
    """X^GWO для каждого аппарата; leaders — массив N×3×d (у каждого аппарата свои три «добычи»)."""
    n, d = p.shape
    a = max(2.0 * (1.0 - k / cfg.gwo_k_max), 0.0)
    x = np.zeros_like(p)
    for i in range(3):
        XL = leaders[:, i, :]
        A = 2 * a * rng.random((n, d)) - a
        C = 2 * rng.random((n, d))
        x += XL - A * np.abs(C * XL - p)
    return x / 3.0, cfg.gwo_weight_scale * a


# ---------- начальные условия ----------

def make_targets(cfg, seed, m, spread, min_sep):
    """Цели: одна — в начале координат (как в сценариях [1]); несколько — в круге радиуса spread
    с интервалом не меньше min_sep. Отдельный генератор: старт аппаратов и шум GWO те же, что при M = 1."""
    p, v, q0, vq0, rng = make_initial(cfg, seed)
    if m == 1:
        return p, v, q0[None, :], vq0[None, :], rng
    r2 = np.random.default_rng([seed, 7])
    pts, tries = [], 0
    while len(pts) < m:
        tries += 1
        if tries > 20000:
            raise RuntimeError("не удаётся расставить цели: уменьшите min_sep или увеличьте spread")
        if tries % 500 == 0:
            pts = []          # перезапуск, если набор зашёл в тупик
        rad = spread * np.sqrt(r2.random())
        phi = 2 * np.pi * r2.random()
        c = np.array([rad * np.cos(phi), rad * np.sin(phi)])
        if not pts or np.min(np.linalg.norm(np.array(pts) - c, axis=1)) >= min_sep:
            pts.append(c)
    q = np.array(pts)
    vq = np.repeat(vq0[None, :], m, axis=0)   # группа целей движется общим курсом
    return p, v, q, vq, rng


# ---------- испытание ----------

def ring_vdes(p, v, qa, va, cfg, ring_radius, k_pos=0.5):
    """Желаемые скорости для метода ring."""
    n = p.shape[0]
    cost = intercept_time(p, qa, va, cfg.v_max)
    g = assign_bottleneck(cost)
    m = len(qa)
    striker = np.full(m, -1)
    for j in range(m):
        idx = np.where(g == j)[0]
        if len(idx):
            striker[j] = idx[np.argmin(cost[idx, j])]
    v_des = np.zeros_like(p)
    for j in range(m):
        idx = np.where(g == j)[0]
        if not len(idx):
            continue
        s = striker[j]
        v_des[s] = intercept_vdes(p[s:s + 1], qa[j:j + 1], va[j:j + 1], cfg.v_max)[0]
        rest = idx[idx != s]
        if not len(rest):
            continue
        base = np.arctan2(*(p[s] - qa[j])[::-1])
        ang = base + 2 * np.pi * np.arange(1, len(rest) + 1) / (len(rest) + 1)
        slots = qa[j][None, :] + ring_radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        c = np.linalg.norm(p[rest][:, None, :] - slots[None, :, :], axis=-1)
        r_i, c_i = linear_sum_assignment(c)
        for a_, b_ in zip(r_i, c_i):
            i = rest[a_]
            vd = va[j] + k_pos * (slots[b_] - p[i]) / cfg.dt
            nn = np.linalg.norm(vd)
            v_des[i] = vd if nn <= cfg.v_max else vd * cfg.v_max / nn
    return v_des


def run_trial_multi(cfg, method, use_boids, seed, m=1, spread=40.0, min_sep=10.0,
                    gwo_drop_captured=False, record=False, ring_radius=15.0,
                    dup_rule="first", km_init="farthest", hysteresis=0.0):
    """hysteresis > 0: аппарат сохраняет прежнюю цель, если новое назначение не лучше прежнего по сумме времён
    больше чем на эту долю (защита от «дрожания» назначений). switches — число смен цели аппаратами."""
    p, v, q, vq, rng = make_targets(cfg, seed, m, spread, min_sep)
    n = p.shape[0]
    T_m = np.full(m, -1)
    min_pair = _min_pair(p)
    pairs = [min_pair]
    t_sim = -1
    state = {}
    km_rng = np.random.default_rng([seed, 11])
    prev_g, switches, decisions = None, 0, 0
    P, Q = [p.copy()], [q.copy()]
    for k in range(cfg.t_max + 1):
        dist = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)   # N×M
        hit = dist.min(axis=0) <= cfg.r_capture
        T_m[(T_m < 0) & hit] = k
        if t_sim < 0 and hit.all():
            t_sim = k
        if (T_m >= 0).all() or k == cfg.t_max:
            break
        alive = np.where(T_m < 0)[0]
        qa, va = q[alive], vq[alive]

        if method in ("gwo_avg", "gwo_group"):
            src = q if not gwo_drop_captured else qa
            c = centroids(src, state, dup_rule, km_init, km_rng)
            if method == "gwo_avg":
                leaders = np.repeat(c[None, :, :], n, axis=0)
            else:
                cost = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=-1)
                g = assign_sum(cost)
                leaders = np.repeat(c[g][:, None, :], 3, axis=1)
            x, w = gwo_point(p, leaders, k, rng, cfg)
            u = w * (x - p)
        elif method == "ring":
            u = (ring_vdes(p, v, qa, va, cfg, ring_radius) - v) / cfg.dt
        else:
            cost = intercept_time(p, qa, va, cfg.v_max)
            if method == "greedy":
                g = np.argmin(cost, axis=1)
            elif method in ("hungarian", "gwo_assign"):
                g = assign_sum(cost)
            elif method == "bottleneck":
                g = assign_bottleneck(cost)
            else:
                raise ValueError(method)
            gt = alive[g]                                   # глобальные номера целей
            if prev_g is not None and hysteresis > 0:
                keep = np.isin(prev_g, alive)
                if keep.all():
                    old = np.searchsorted(alive, prev_g)
                    c_old = cost[np.arange(n), old].sum() if method != "bottleneck" else cost[np.arange(n), old].max()
                    c_new = cost[np.arange(n), g].sum() if method != "bottleneck" else cost[np.arange(n), g].max()
                    if c_new > (1 - hysteresis) * c_old:
                        g, gt = old, prev_g
            if prev_g is not None:
                valid = np.isin(prev_g, alive)
                switches += int(((gt != prev_g) & valid).sum())
                decisions += int(valid.sum())
            prev_g = gt
            if method == "gwo_assign":
                leaders = np.repeat(qa[g][:, None, :], 3, axis=1)
                x, w = gwo_point(p, leaders, k, rng, cfg)
                u = w * (x - p)
            else:
                v_des = intercept_vdes(p, qa[g], va[g], cfg.v_max)
                u = (v_des - v) / cfg.dt
        if use_boids:
            u = u + boids_accel(p, v, cfg)
        u = _clip_norm(u, cfg.a_max)
        if cfg.integrator == "explicit":
            p = p + cfg.dt * v
            v = _clip_norm(v + cfg.dt * u, cfg.v_max)
        else:
            v = _clip_norm(v + cfg.dt * u, cfg.v_max)
            p = p + cfg.dt * v
        if use_boids and cfg.post_separate:
            p = post_separate(p, cfg.post_separate)
        q = q + cfg.dt * vq
        mp = _min_pair(p)
        pairs.append(mp)
        min_pair = min(min_pair, mp)
        if record:
            P.append(p.copy()); Q.append(q.copy())
    all_cap = bool((T_m >= 0).all())
    t_cap = int(T_m.max()) if all_cap else np.nan
    last = int(T_m.max()) if all_cap else len(pairs)
    out = dict(switch_rate=switches / decisions if decisions else 0.0, all_captured=all_cap, t_cap=t_cap, frac_captured=float((T_m >= 0).mean()),
               simultaneous=t_sim >= 0, t_sim=t_sim if t_sim >= 0 else np.nan,
               min_pair=float(min_pair), collision=bool(min_pair < cfg.d_min),
               collision_pre=bool(min(pairs[:last]) < cfg.d_min) if last > 0 else False)
    if record:
        out.update(P=np.array(P), Q=np.array(Q), T_m=T_m)
    return out
