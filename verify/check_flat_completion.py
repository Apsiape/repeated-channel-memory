"""Numerical check of the flat completion, Lemma 5.2 (used in Theorem 3 and Remark 4.5).

Given unitaries V_1..V_n in M_m (normalized trace), build the completed family
in N' = M_m (x) l^inf(Z_2^L): the V_i (as constants) plus pairs
w_{l,c}^{+/-} = x_l (x) 1 +/- y_l (x) chi_{l,c}, where x_l = alpha_l e_l runs over
the eigenvectors of the frame deficiency, x_l = (U_+ + U_-)/2 and y_l = (U_+ - U_-)/2
by the polar trick, and chi_{l,c} are distinct nontrivial characters of Z_2^L.

Checks: every new element is unitary; the old moment matrix is an exact principal
submatrix of the new one; the weighted frame operator of the whole family is flat on
its span (so the Schur channel x -> X' o x has a flat probe of rank rank(X')).
Run: python verify/check_flat_completion.py   (exit code 0 on pass)
"""
import itertools
import sys

import numpy as np

rng = np.random.default_rng(7)


def haar(m):
    z = (rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def polar_pair(x):
    """Unitaries U_+, U_- with x = (U_+ + U_-)/2, for a contraction x."""
    p, s, qh = np.linalg.svd(x)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    v = p @ qh
    q = qh.conj().T
    up = v @ q @ np.diag(np.exp(1j * theta)) @ qh
    um = v @ q @ np.diag(np.exp(-1j * theta)) @ qh
    return up, um


def run(n, m, seed_weights=None):
    V = [haar(m) for _ in range(n)]
    p = np.full(n, 1.0 / n) if seed_weights is None else seed_weights
    # vectors in L^2(M_m, tr_m): vec(a)/sqrt(m)
    vec = lambda a: a.reshape(-1) / np.sqrt(m)
    S = sum(p[i] * np.outer(vec(V[i]), vec(V[i]).conj()) for i in range(n))
    ev, U = np.linalg.eigh(S)
    span = U[:, ev > 1e-10]
    r = span.shape[1]
    c = ev.max()
    D = c * span @ span.conj().T - S
    dev, dU = np.linalg.eigh(D)
    dirs = [(mu, dU[:, k]) for k, mu in enumerate(dev) if mu > 1e-12]
    # e_l as matrices, unit in normalized HS norm
    E = [(mu, (u * np.sqrt(m)).reshape(m, m)) for mu, u in dirs]
    M = int(np.ceil(max(np.linalg.norm(e, 2) ** 2 for _, e in E))) if E else 1
    npairs = M * len(E)
    L = int(np.ceil(np.log2(npairs + 1)))
    A = list(itertools.product([0, 1], repeat=L))
    chars = [s for s in itertools.product([0, 1], repeat=L) if any(s)][:npairs]
    chi = lambda s, a: (-1) ** (sum(si * ai for si, ai in zip(s, a)) % 2)
    # elements of N' as lists over A; L^2 vector: concat vec(f(a)) / sqrt(m |A|)
    lvec = lambda f: np.concatenate([f[a].reshape(-1) for a in range(len(A))]) / np.sqrt(m * len(A))
    family, weights = [], []
    for i in range(n):
        family.append([V[i]] * len(A))
        weights.append(p[i])
    k = 0
    max_unitary_err = 0.0
    for mu, e in E:
        alpha2 = mu / (M * c + mu)
        x = np.sqrt(alpha2) * e
        assert np.linalg.norm(x, 2) <= 1 + 1e-12
        up, um = polar_pair(x)
        y = (up - um) / 2
        q = c / (2 * (1 - alpha2))
        for _ in range(M):
            s = chars[k]
            k += 1
            for sign in (+1, -1):
                f = [x + sign * chi(s, a) * y for a in A]
                for g in f:
                    max_unitary_err = max(max_unitary_err, np.linalg.norm(g.conj().T @ g - np.eye(m)))
                family.append(f)
                weights.append(q)
    weights = np.array(weights)
    vecs = np.array([lvec(f) for f in family])
    Gram = vecs.conj() @ vecs.T          # <f_i, f_j> = tau'(f_i^* f_j)
    X = np.array([[np.trace(V[i].conj().T @ V[j]) / m for j in range(n)] for i in range(n)])
    frame = (vecs.T * weights) @ vecs.conj()
    fev = np.linalg.eigvalsh(frame)
    nz = fev[fev > 1e-9]
    rprime = np.linalg.matrix_rank(Gram, tol=1e-9)
    total = weights.sum()
    return {
        "n": n, "m": m, "rank_X": r, "copies_M": M, "new_unitaries": len(family) - n,
        "n_prime": len(family), "rank_Xprime": int(rprime),
        "max_unitarity_error": float(max_unitary_err),
        "principal_submatrix_error": float(np.abs(Gram[:n, :n] - X).max()),
        "frame_flatness_spread": float((nz.max() - nz.min()) / nz.max()),
        "frame_rank": int(len(nz)),
        "normalized_flat_value_times_rank": float(nz.mean() / total * len(nz)),
        "unit_diagonal_error": float(np.abs(np.diag(Gram) - 1).max()),
    }


if __name__ == "__main__":
    ok = True
    for n, m in [(3, 2), (4, 2), (4, 3), (6, 3), (5, 4)]:
        res = run(n, m)
        print(res)
        ok &= res["max_unitarity_error"] < 1e-9
        ok &= res["principal_submatrix_error"] < 1e-12
        ok &= res["frame_flatness_spread"] < 1e-9
        ok &= res["frame_rank"] == res["rank_Xprime"]
        ok &= res["unit_diagonal_error"] < 1e-12
    w = np.array([0.5, 0.2, 0.2, 0.1])
    res = run(4, 3, w)
    print(res)
    ok &= res["frame_flatness_spread"] < 1e-9 and res["max_unitarity_error"] < 1e-9
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
