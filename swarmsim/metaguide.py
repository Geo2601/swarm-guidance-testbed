"""Популяционные метаэвристики как законы наведения при известной цели.

Единый регулятор, как в [1] (шаги 4 и 6.1–6.2):  u = w_g(k) (X_new − p),  w_g(k) = a(k) / (2√d),
a(k) = 2 (1 − k / k_max).  Меняется только оператор, предлагающий точку X_new («новую позицию» метаэвристики).
Цель q известна и играет роль «добычи» / аттрактора, как центроиды в [1].

Операторы (для одного аппарата p с текущей скоростью v):
  gwo       — [1]:  X = (1/3) Σ_i [ q − A_i ⊙ |C_i ⊙ q − p| ],  A_i = 2a r − a,  C_i = 2 r'   (уже есть в guidance.gwo)
  gwo_rel   — тот же GWO, но член D записан через разность:  D_i = |C_i ⊙ (q − p)|.
              Правило эквивариантно к сдвигу системы координат; служит причинной проверкой.
  woa       — Whale Optimization Algorithm (Mirjalili, Lewis, 2016), как в исходной реализации: на аппарат
              скаляры A = 2a r1 − a, C = 2 r2, l ∈ [a2, 1], a2 = −1 − k/k_max, b = 1; с вероятностью 1/2
              окружение (|A| < 1: X = q − A |C q − p|; |A| ≥ 1: X = X_rand − A |C X_rand − p|), иначе
              спираль X = |q − p| e^{bl} cos(2πl) + q (модуль поэлементный).
  pso       — PSO с коэффициентом сжатия (Clerc, Kennedy, 2002): χ = 0,7298, c1 = c2 = 2,05; личный и
              глобальный аттракторы равны цели:  X = p + χ [ v + (c1 r1 + c2 r2) ⊙ (q − p) ].
  pso_hist  — «роботизированный» PSO: аттракторы — лучшая посещённая точка аппарата и лучшая точка группы
              по критерию «расстояние до цели» (цель в аттракторы не подставляется).
  attract   — среднее поле GWO:  X = q  (детерминированное притяжение с тем же w_g).
"""
import numpy as np

from .guidance import LAWS

CHI, C1, C2 = 0.7298, 2.05, 2.05


def _a(k, cfg):
    return max(2.0 * (1.0 - k / cfg.gwo_k_max), 0.0)


def _wg(k, cfg):
    return cfg.gwo_weight_scale * _a(k, cfg)


def x_gwo_rel(p, q, k, rng, cfg):
    n, d = p.shape
    a = _a(k, cfg)
    x = np.zeros_like(p)
    for _ in range(3):
        A = 2 * a * rng.random((n, d)) - a
        C = 2 * rng.random((n, d))
        x += q[None, :] - A * np.abs(C * (q[None, :] - p))
    return x / 3.0


def x_woa(p, q, k, rng, cfg, p_spiral=0.5):
    n, d = p.shape
    a = _a(k, cfg)
    a2 = -1.0 - k / cfg.gwo_k_max
    x = np.empty_like(p)
    for i in range(n):
        r1, r2 = rng.random(), rng.random()
        A, C = 2 * a * r1 - a, 2 * r2
        l = (a2 - 1.0) * rng.random() + 1.0
        if rng.random() < 1.0 - p_spiral:
            if abs(A) >= 1.0:
                xr = p[rng.integers(n)]
                x[i] = xr - A * np.abs(C * xr - p[i])
            else:
                x[i] = q - A * np.abs(C * q - p[i])
        else:
            x[i] = np.abs(q - p[i]) * np.exp(l) * np.cos(2 * np.pi * l) + q
    return x


def x_pso(p, v, q, rng):
    n, d = p.shape
    r1, r2 = rng.random((n, d)), rng.random((n, d))
    return p + CHI * (v + (C1 * r1 + C2 * r2) * (q[None, :] - p))


_HIST = {}


def x_pso_hist(p, v, q, k, rng):
    key = id(rng)
    dist = np.linalg.norm(p - q[None, :], axis=1)
    if k == 0 or key not in _HIST:
        _HIST[key] = dict(pb=p.copy(), pbf=dist.copy())
    st = _HIST[key]
    better = dist < st["pbf"]
    st["pb"][better] = p[better]
    st["pbf"][better] = dist[better]
    g = st["pb"][np.argmin(st["pbf"])]
    n, d = p.shape
    r1, r2 = rng.random((n, d)), rng.random((n, d))
    return p + CHI * (v + C1 * r1 * (st["pb"] - p) + C2 * r2 * (g[None, :] - p))


def _wrap(proposal):
    def law(p, v, q, vq, k, rng, cfg):
        return _wg(k, cfg) * (proposal(p, v, q, k, rng, cfg) - p)
    return law


LAWS.update({
    "gwo_rel": _wrap(lambda p, v, q, k, rng, cfg: x_gwo_rel(p, q, k, rng, cfg)),
    "woa": _wrap(lambda p, v, q, k, rng, cfg: x_woa(p, q, k, rng, cfg)),
    "pso": _wrap(lambda p, v, q, k, rng, cfg: x_pso(p, v, q, rng)),
    "pso_hist": _wrap(lambda p, v, q, k, rng, cfg: x_pso_hist(p, v, q, k, rng)),
    "attract": _wrap(lambda p, v, q, k, rng, cfg: np.repeat(q[None, :], p.shape[0], axis=0)),
})

META = ["gwo", "gwo_rel", "woa", "pso", "pso_hist", "attract"]


# --- Второе отображение «точка метаэвристики → ускорение» (проверка устойчивости выводов к регулятору) ---
# Аппарат стремится попасть в X_new за один шаг: v_des = (X_new − p)/Δt (с ограничением V_max),
# u = (v_des − v)/Δt (ограничение a_max накладывается в sim). Для PSO это совпадает с командой скорости χ(...).
def _wrap_v(proposal):
    def law(p, v, q, vq, k, rng, cfg):
        x = proposal(p, v, q, k, rng, cfg)
        vd = (x - p) / cfg.dt
        n = np.linalg.norm(vd, axis=1, keepdims=True)
        vd = np.where(n > cfg.v_max, vd * (cfg.v_max / np.maximum(n, 1e-12)), vd)
        return (vd - v) / cfg.dt
    return law


def _x_gwo(p, v, q, k, rng, cfg):
    n, d = p.shape
    a = _a(k, cfg)
    x = np.zeros_like(p)
    for _ in range(3):
        A = 2 * a * rng.random((n, d)) - a
        C = 2 * rng.random((n, d))
        x += q[None, :] - A * np.abs(C * q[None, :] - p)
    return x / 3.0


LAWS.update({
    "gwo_v": _wrap_v(_x_gwo),
    "gwo_rel_v": _wrap_v(lambda p, v, q, k, rng, cfg: x_gwo_rel(p, q, k, rng, cfg)),
    "woa_v": _wrap_v(lambda p, v, q, k, rng, cfg: x_woa(p, q, k, rng, cfg)),
    "pso_v": _wrap_v(lambda p, v, q, k, rng, cfg: x_pso(p, v, q, rng)),
    "attract_v": _wrap_v(lambda p, v, q, k, rng, cfg: np.repeat(q[None, :], p.shape[0], axis=0)),
})


# --- Ветви WOA по отдельности (E2б): только спираль / только окружение и поиск ---
LAWS.update({
    "woa_spiral": _wrap(lambda p, v, q, k, rng, cfg: x_woa(p, q, k, rng, cfg, p_spiral=1.0)),
    "woa_encircle": _wrap(lambda p, v, q, k, rng, cfg: x_woa(p, q, k, rng, cfg, p_spiral=0.0)),
})


# ---------------------------------------------------------------------------------------------------------------
# Harris hawks optimization (HHO) как закон наведения.
# Heidari A. A., Mirjalili S., Faris H. и др. Harris hawks optimization: Algorithm and applications //
# Future Generation Computer Systems. 2019. Т. 97. С. 849–872. doi 10.1016/j.future.2019.02.028.
# Уравнения — по исходному описанию; сверены с реализациями mealpy 3.0.3 (OriginalHHO) и NiaPy (hho.py):
#   E = 2E0(1 − k/K), E0 ~ U(−1, 1); J = 2(1 − r);
#   |E| ≥ 1, r < 0,5: X_rand − r1 |X_rand − 2 r2 X|             (абсолютный член 2 r2 X)
#   |E| ≥ 1, r ≥ 0,5: (X_rabbit − X_m) − r3 (LB + r4 (UB − LB))   (разность минус точка области)
#   |E| < 1, r ≥ 0,5, |E| ≥ 0,5: ΔX − E |J X_rabbit − X|, ΔX = X_rabbit − X   (мягкое сжатие; ΔX — вектор, а не точка)
#   |E| < 1, r ≥ 0,5, |E| < 0,5: X_rabbit − E |ΔX|                 (жёсткое сжатие; эквивариантно)
#   |E| < 1, r < 0,5: Y = X_rabbit − E |J X_rabbit − B|, B = X (|E| ≥ 0,5) или X_m (|E| < 0,5);
#                     Z = Y + S ⊙ LF; принимается Y, иначе Z, если ближе к цели, чем X; иначе X остаётся.
# Кролик X_rabbit = цель q (известна). Область поиска [LB, UB] = q ± box (сдвигается вместе со сценой, поэтому
# неэквивариантность возникает только из формул алгоритма). clip=True — предложение обрезается по области, как
# в исходном коде (возврат за границу).
_LEVY_SIGMA = 0.6965745025576968  # (Γ(1+β) sin(πβ/2) / (Γ((1+β)/2) β 2^((β−1)/2)))^(1/β), β = 1,5


def _levy(d, rng):
    u = rng.normal(size=d) * _LEVY_SIGMA
    v = rng.normal(size=d)
    return 0.01 * u / np.abs(v) ** (1 / 1.5)


def x_hho(p, q, k, rng, cfg, box=150.0, clip=True, only_hard=False, only_soft=False, j_one=False):
    n, d = p.shape
    E1 = _a(k, cfg)
    xm = p.mean(axis=0)
    lb, ub = q - box, q + box
    f0 = np.linalg.norm(p - q[None, :], axis=1)
    x = np.empty_like(p)
    for i in range(n):
        E = E1 * (2 * rng.random() - 1)
        if only_hard:
            x[i] = q - E * np.abs(q - p[i])
            continue
        if only_soft:
            # контроль: только мягкое сжатие (4) [4]; j_one=True — J ≡ 1, т. е. без члена Jq: остаётся разность
            # ΔX = q − p, использованная как положение (X − p = q − 2p − E|q − p| не выражается через разности)
            J = 1.0 if j_one else 2 * (1 - rng.random())
            x[i] = (q - p[i]) - E * np.abs(J * q - p[i])
            continue
        if abs(E) >= 1:
            if rng.random() < 0.5:
                xr = p[rng.integers(n)]
                x[i] = xr - rng.random() * np.abs(xr - 2 * rng.random() * p[i])
            else:
                x[i] = (q - xm) - rng.random() * ((ub - lb) * rng.random() + lb)
        else:
            r = rng.random()
            if r >= 0.5 and abs(E) < 0.5:
                x[i] = q - E * np.abs(q - p[i])
            elif r >= 0.5:
                J = 2 * (1 - rng.random())
                x[i] = (q - p[i]) - E * np.abs(J * q - p[i])
            else:
                J = 2 * (1 - rng.random())
                base = p[i] if abs(E) >= 0.5 else xm
                y = q - E * np.abs(J * q - base)
                if np.linalg.norm(y - q) < f0[i]:
                    x[i] = y
                else:
                    z = y + rng.random(d) * _levy(d, rng)
                    x[i] = z if np.linalg.norm(z - q) < f0[i] else p[i]
    if clip:
        x = np.clip(x, lb, ub)
    return x


LAWS.update({
    "hho": _wrap(lambda p, v, q, k, rng, cfg: x_hho(p, q, k, rng, cfg)),
    "hho_noclip": _wrap(lambda p, v, q, k, rng, cfg: x_hho(p, q, k, rng, cfg, clip=False)),
    "hho_hard": _wrap(lambda p, v, q, k, rng, cfg: x_hho(p, q, k, rng, cfg, only_hard=True)),
    "hho_soft": _wrap(lambda p, v, q, k, rng, cfg: x_hho(p, q, k, rng, cfg, only_soft=True)),
    "hho_soft_j1": _wrap(lambda p, v, q, k, rng, cfg: x_hho(p, q, k, rng, cfg, only_soft=True, j_one=True)),
})


# ---------------------------------------------------------------------------------------------------------------
# Инвариантные контроли к опыту с поворотом осей (расчёт 6).
# spso2011 — SPSO-2011 (Zambrano-Bigiarini, Clerc, Rojas, CEC 2013, doi 10.1109/CEC.2013.6557848): точка x′ берётся
#   равномерно в круге H(G, |G − x|), G = x + c(p_i − x)/2 при p_i = l_i (здесь оба аттрактора — цель q);
#   X = x + w v + (x′ − x), w = 1/(2 ln 2), c = 1/2 + ln 2. Шум изотропный — закон эквивариантен к сдвигу и повороту.
# gwo_iso — GWO со скалярными A, C на каждого «лидера» и без покоординатного модуля: X = (1/3) Σ [q − A_l C_l (q − p)].
#   Среднее то же (E[A] = 0), шум направлен вдоль линии визирования — эквивариантен к сдвигу и повороту.
W_SPSO, C_SPSO = 1.0 / (2.0 * np.log(2.0)), 0.5 + np.log(2.0)


def _disk(n, d, rng):
    """Равномерно в единичном шаре размерности d (по объёму)."""
    g = rng.normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * rng.random((n, 1)) ** (1.0 / d)


def x_spso2011(p, v, q, rng):
    n, d = p.shape
    G = p + 0.5 * C_SPSO * (q[None, :] - p)
    rad = np.linalg.norm(G - p, axis=1, keepdims=True)
    xp = G + rad * _disk(n, d, rng)
    return p + W_SPSO * v + (xp - p)


def x_gwo_iso(p, q, k, rng, cfg):
    n, d = p.shape
    a = _a(k, cfg)
    x = np.zeros_like(p)
    for _ in range(3):
        A = 2 * a * rng.random((n, 1)) - a
        C = 2 * rng.random((n, 1))
        x += q[None, :] - A * C * (q[None, :] - p)
    return x / 3.0


LAWS.update({
    "spso2011": _wrap(lambda p, v, q, k, rng, cfg: x_spso2011(p, v, q, rng)),
    "gwo_iso": _wrap(lambda p, v, q, k, rng, cfg: x_gwo_iso(p, q, k, rng, cfg)),
})

# gwo_c1 — GWO с C = 1 в члене D: X = (1/3) Σ [q − A_l ⊙ |q − p|]. Случайные числа C разыгрываются, как в gwo
#   (порядок розыгрыша тот же), но не используются. При цели в начале координат (q = 0) |C ⊙ q − p| = |q − p|,
#   поэтому при s = 0 и неподвижной цели закон совпадает с gwo испытание в испытание; член через разность делает
#   его эквивариантным к сдвигу. Это более чистый контроль, чем gwo_rel (тот отличается от gwo уже при s = 0).
def x_gwo_c1(p, q, k, rng, cfg):
    n, d = p.shape
    a = _a(k, cfg)
    x = np.zeros_like(p)
    for _ in range(3):
        A = 2 * a * rng.random((n, d)) - a
        _C = 2 * rng.random((n, d))  # noqa: F841  (розыгрыш сохраняет поток случайных чисел gwo)
        x += q[None, :] - A * np.abs(q[None, :] - p)
    return x / 3.0


LAWS.update({
    "gwo_c1": _wrap(lambda p, v, q, k, rng, cfg: x_gwo_c1(p, q, k, rng, cfg)),
})
