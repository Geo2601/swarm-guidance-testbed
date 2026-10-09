"""Independent multi-target re-implementation of the model in [1] (see README): GWO+Boids with
K-means target centroids (formula (5)), group reading of step 2, and assignment + lead pursuit baselines.
Written without importing swarmsim; used to cross-check swarmsim/multi.py.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

PRM = dict(dt=1.0, Vmax=10.0, amax=2.0, Rcap=5.0, dmin=5.0, ks=0.1, ka=0.1, kc=0.1,
           kmax=30, nb=30.0, d=2)
BIG = 1e9


def clipn(x, m):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.where(n > m, m / np.maximum(n, 1e-12), 1.0)


# ---------------------------------------------------------------- geometry
def sample_far(rng, n, gen, sep=10.0, tries=10000):
    for _ in range(tries):
        P = gen(n)
        if n < 2:
            return P
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        D[np.eye(n, dtype=bool)] = np.inf
        if D.min() >= sep:
            return P
    raise RuntimeError("placement failed")


def init_scene(seed, N, M, S, vtgt):
    rng = np.random.default_rng(seed)

    def gen_drones(n):
        th = rng.uniform(0, 2 * np.pi, n); r = rng.uniform(85, 110, n)
        return np.c_[r * np.cos(th), r * np.sin(th)]

    def gen_tg(n):
        th = rng.uniform(0, 2 * np.pi, n); r = S * np.sqrt(rng.uniform(0, 1, n))
        return np.c_[r * np.cos(th), r * np.sin(th)]

    p = sample_far(rng, N, gen_drones)
    q = np.zeros((1, 2)) if M == 1 else sample_far(rng, M, gen_tg)
    h = rng.uniform(0, 2 * np.pi)
    vt = vtgt * np.array([np.cos(h), np.sin(h)])
    return p, q, vt


# ---------------------------------------------------------------- k-means (K=3)
def kmeans3(X, seed=12345, n_init=10, iters=100):
    M = len(X)
    if M < 3:
        return np.array([X[0], X[1 % M], X[0]])  # duplicate: M=1 -> [q,q,q]; M=2 -> [q1,q2,q1]
    if M == 3:
        return X.copy()
    rng = np.random.default_rng(seed)
    best, bestJ = None, np.inf
    for _ in range(n_init):
        # k-means++ init
        C = [X[rng.integers(M)]]
        for _k in range(2):
            d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(X[rng.choice(M, p=d2 / d2.sum())])
        C = np.array(C)
        for _it in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
            Cn = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(3)])
            if np.allclose(Cn, C):
                break
            C = Cn
        J = np.min(((X[:, None] - C[None]) ** 2).sum(-1), axis=1).sum()
        if J < bestJ - 1e-9:
            best, bestJ = C, J
    return best


# ---------------------------------------------------------------- GWO / Boids
def gwo_X(p, centers_per_drone, a, rng):
    """centers_per_drone: (N,3,2). X_j^(i) = c_i - A*|C*c_i - p_j|, A=2a r1 - a, C=2 r2; mean over i."""
    N = len(p)
    r1 = rng.random((N, 3, 2)); r2 = rng.random((N, 3, 2))
    A = 2 * a * r1 - a; C = 2 * r2
    c = centers_per_drone
    Xi = c - A * np.abs(C * c - p[:, None, :])
    return Xi.mean(1)


def boids(p, v, prm=PRM):
    N = len(p); u = np.zeros_like(p)
    for i in range(N):
        diff = p - p[i]; r = np.linalg.norm(diff, axis=1)
        nb = (np.arange(N) != i) & (r <= prm['nb'])
        if not nb.any():
            continue
        sep = nb & (r < prm['dmin'])
        FS = -prm['ks'] * (diff[sep] / np.maximum(r[sep], 1e-9)[:, None] ** 2).sum(0) if sep.any() else 0.0
        FA = prm['ka'] * (v[nb].mean(0) - v[i])
        FC = prm['kc'] * (p[nb].mean(0) - p[i])
        u[i] = FS + FA + FC
    return u


def balanced_assign(cost):
    """cost (N, K): each column gets floor(N/K) or ceil(N/K) rows; min total cost.
    If N<K: rectangular min-sum (each row a distinct column)."""
    N, K = cost.shape
    if N <= K:
        r, c = linear_sum_assignment(cost)
        out = np.full(N, -1); out[r] = c
        return out
    fl = N // K
    cols, extra = [], []
    for k in range(K):
        cols += [k] * fl; extra += [0.0] * fl
        if N % K:
            cols += [k]; extra += [1e6]   # optional slot, penalised -> used exactly N-K*fl times
    cols = np.array(cols)
    C = cost[:, cols] + np.array(extra)[None]
    r, c = linear_sum_assignment(C)
    out = np.full(N, -1); out[r] = cols[c]
    return out


# ---------------------------------------------------------------- lead pursuit
def lead_time(p, q, vt, V):
    """time to a straight-line meeting point at speed V from p to target (q, vt); inf if impossible."""
    r = q - p
    a = vt @ vt - V * V; b = 2 * r @ vt; c = r @ r
    if abs(a) < 1e-12:
        return -c / b if b < 0 else np.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    ts = [t for t in ((-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)) if t > 0]
    return min(ts) if ts else (0.0 if c == 0 else np.inf)


def time_matrix(p, q, vt, V):
    return np.array([[lead_time(pi, qm, vt, V) for qm in q] for pi in p])


def lead_control(p, v, qa, vt, prm=PRM):
    """qa: (N,2) aimed target for each drone."""
    u = np.zeros_like(p)
    for i in range(len(p)):
        t = lead_time(p[i], qa[i], vt, prm['Vmax'])
        aim = qa[i] + vt * t if np.isfinite(t) else qa[i]
        dv = aim - p[i]; n = np.linalg.norm(dv)
        vdes = prm['Vmax'] * dv / n if n > 1e-9 else np.zeros(2)
        u[i] = (vdes - v[i]) / prm['dt']
    return clipn(u, prm['amax'])


def bottleneck_assign(T):
    """T (N,K). Size-min(N,K) matching minimising the max cost, ties -> min sum.
    Extra rows (N>K) go to their min-cost column."""
    N, K = T.shape
    Tf = np.where(np.isfinite(T), T, BIG)
    vals = np.unique(Tf)
    m = min(N, K)

    def solve(thr):
        C = np.where(Tf <= thr, Tf, BIG * 10)
        r, c = linear_sum_assignment(C)
        ok = (C[r, c] < BIG * 10).sum() == m
        return ok, r, c

    lo, hi = 0, len(vals) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(vals[mid])[0]:
            hi = mid
        else:
            lo = mid + 1
    ok, r, c = solve(vals[lo])
    out = np.full(N, -1); out[r] = c
    for i in np.where(out < 0)[0]:
        out[i] = int(np.argmin(Tf[i]))
    return out


# ---------------------------------------------------------------- trial
def run_trial(method, p, q, vt, seed, prm=PRM, simult_mode=None, keep_all=False, order='pv'):
    """method in gwo_avg, gwo_group, hungarian, bottleneck, gwo_noboids.
    Returns dict: T (cumulative all-captured step or None), viol (min pair dist<dmin at any step),
    simult (first step with D_max<=Rcap, if simult_mode run to kmax)."""
    rng = np.random.default_rng(seed)
    p = p.copy(); q = q.copy(); N = len(p); M = len(q)
    v = np.zeros_like(p)
    capt = np.full(M, -1)
    viol = False; simult = None
    def pair_min(P):
        if len(P) < 2: return np.inf
        D = np.linalg.norm(P[:, None] - P[None], axis=2); D[np.eye(len(P), dtype=bool)] = np.inf
        return D.min()
    viol |= pair_min(p) < prm['dmin']
    T = None
    for k in range(prm['kmax']):
        if method.startswith('gwo'):
            a = 2 * (1 - k / prm['kmax'])
            cents = kmeans3(q)                                 # recomputed every iteration, all targets
            if method == 'gwo_group':
                Dc = np.linalg.norm(p[:, None] - cents[None], axis=2)
                g = balanced_assign(Dc)
                cpd = np.repeat(cents[g][:, None, :], 3, axis=1)
            else:
                cpd = np.repeat(cents[None], N, axis=0)
            X = gwo_X(p, cpd, a, rng)
            u = (a / (2 * np.sqrt(prm['d']))) * (X - p)
            if method != 'gwo_noboids':
                u = u + boids(p, v, prm)
            u = clipn(u, prm['amax'])
        else:
            idx = np.where(capt < 0)[0]
            if keep_all or len(idx) == 0:
                idx = np.arange(M)
            Tm = time_matrix(p, q[idx], vt, prm['Vmax'])
            Tc = np.where(np.isfinite(Tm), Tm, 1e6 + np.linalg.norm(p[:, None] - q[idx][None], axis=2))
            asg = balanced_assign(Tc) if method == 'hungarian' else bottleneck_assign(Tc)
            qa = np.array([q[idx[j]] if j >= 0 else p[i] for i, j in enumerate(asg)])
            u = lead_control(p, v, qa, vt, prm)
        # update: position by previous velocity, then velocity
        if order == 'pv':
            p = p + v * prm['dt']
            v = clipn(v + u * prm['dt'], prm['Vmax'])
        else:
            v = clipn(v + u * prm['dt'], prm['Vmax'])
            p = p + v * prm['dt']
        q = q + vt * prm['dt']
        viol |= pair_min(p) < prm['dmin']
        dist = np.linalg.norm(p[:, None] - q[None], axis=2)   # (N,M)
        within = dist.min(0) <= prm['Rcap']
        capt[(capt < 0) & within] = k + 1
        if simult_mode and simult is None and within.all():
            simult = k + 1
        if T is None and (capt >= 0).all():
            T = int(capt.max())
            if not simult_mode:
                break
    return dict(T=T, viol=bool(viol), simult=simult)
