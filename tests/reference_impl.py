"""Deliberately naive reference implementation of CTAL's refresh.

Written directly from the paper's equations with explicit Python loops and
numpy, sharing NO code with ``ctal.core``.  The test-suite checks that the
vectorized implementation matches it on random inputs.
"""

import math

import numpy as np


def reference_refresh(logits_w, z_w, z_s, labels, memory_prev, cfg, lam, rho, kappa,
                      perms, nbr=None, views_override=None):
    """``views_override=(hL, hT, hR)`` replaces the computed views before ranking.

    Used for inputs with exact ties: two mathematically equal losses can differ
    in the last ulp between implementations, which changes float-equality ties.
    """
    logits_w = np.asarray(logits_w, dtype=np.float64)
    z_w = np.asarray(z_w, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    labels = np.asarray(labels)
    m = np.asarray(memory_prev, dtype=np.float64)
    perms = np.asarray(perms)
    N, K = logits_w.shape
    q = cfg.q

    # Eq. (4): -log softmax(logits)[y]
    hL = np.zeros(N)
    for i in range(N):
        mx = max(logits_w[i])
        lse = mx + math.log(sum(math.exp(v - mx) for v in logits_w[i]))
        hL[i] = lse - logits_w[i, labels[i]]
    # Eq. (5)
    hT = np.zeros(N)
    for i in range(N):
        best = max(m[i, kk] for kk in range(K) if kk != labels[i])
        hT[i] = max(best - m[i, labels[i]], 0.0)
    # Eq. (6)
    def nz(v):
        return v / max(np.linalg.norm(v), cfg.eps_z)
    hR = np.zeros(N)
    zh = np.zeros_like(z_w)
    for i in range(N):
        a, b = nz(z_w[i]), nz(z_s[i])
        zh[i] = a
        hR[i] = min(max(0.5 * (1.0 - float(np.dot(a, b))), 0.0), 1.0)

    views_computed = (hL.copy(), hT.copy(), hR.copy())
    if views_override is not None:
        hL, hT, hR = (np.asarray(v, dtype=np.float64) for v in views_override)

    # Eq. (7)
    def U(h):
        out = np.zeros(N)
        for i in range(N):
            I = [j for j in range(N) if labels[j] == labels[i]]
            less = sum(1 for j in I if h[j] < h[i])
            eq = sum(1 for j in I if h[j] == h[i])
            out[i] = (less + 0.5 * eq) / len(I)
        return out
    UL, UT, UR = U(hL), U(hT), U(hR)
    # Eq. (8)
    ex = lambda u: min(max((u - q) / (1 - q), 0.0), 1.0)
    xL = np.array([ex(u) for u in UL]); xT = np.array([ex(u) for u in UT]); xR = np.array([ex(u) for u in UR])
    # Eq. (9)
    c = (xT + xR) / 2
    tau = np.sqrt(xL * c)

    # Graph: k largest cosine among same-label j != i
    if nbr is None:
        nbr = []
        for i in range(N):
            cand = [j for j in range(N) if j != i and labels[j] == labels[i]]
            sims = sorted(((float(np.dot(zh[i], zh[j])), j) for j in cand), key=lambda t: -t[0])
            nbr.append([j for _, j in sims[:cfg.k]])
    nbr = np.asarray(nbr)

    # Eq. (10), (11)
    s = np.array([sum(tau[j] for j in nbr[i]) / cfg.k for i in range(N)])
    P = perms.shape[0]
    s_null = np.zeros((P, N))
    for p in range(P):
        for i in range(N):
            s_null[p, i] = sum(tau[perms[p, j]] for j in nbr[i]) / cfg.k

    # Eq. (12), (13)
    a = np.zeros(N)
    for i in range(N):
        I = [j for j in range(N) if labels[j] == labels[i]]
        pool = [s_null[p, j] for p in range(P) for j in I]
        Lc = np.quantile(pool, cfg.eta0, method="linear")
        Hc = np.quantile(pool, cfg.eta1, method="linear")
        a[i] = min(max((s[i] - Lc) / max(Hc - Lc, cfg.eps_a), 0.0), 1.0)

    # Eq. (14), (15)
    r = xL * (1 + rho * c + kappa * a)
    v = cfg.omega + (1 - cfg.omega) * np.exp(-lam * r)
    w = v / v.mean()
    return dict(h_L=views_computed[0], h_T=views_computed[1], h_R=views_computed[2], U_L=UL, U_T=UT, U_R=UR, x_L=xL, x_T=xT, x_R=xR,
                c=c, tau=tau, nbr=nbr, s=s, s_null=s_null, a=a, r=r, v=v, w=w)
