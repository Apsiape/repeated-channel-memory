"""Numerical stress test of the Schur builder (18) and the transfer lemma, Lemma A.2.

Not a proof. Checks every intermediate inequality in the proof of Lemma A.2 on random noncommutative
instances and on random nearby finite tracial factorizations with enlarged baths. Each line reports the
worst ratio of an observed quantity to its bound; every ratio must be at most 1.
Run: python verify/check_transfer_lemma.py   (exit code 0 on pass)
"""
import sys

import numpy as np
from scipy.linalg import expm, schur

rng = np.random.default_rng(20261006)
I1 = 1j


def tr(X):
    return np.trace(X) / X.shape[0]


def n2(X):  # normalized 2-norm on square matrices
    return np.sqrt(np.real(np.trace(X.conj().T @ X)) / X.shape[1])


def rand_unitary(k):
    Z = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / abs(np.diag(R)))


def instance(q, a, k):
    """PVMs e[x][u] in M_k (some outcomes may be zero: padding)."""
    e = []
    for x in range(q):
        V = rand_unitary(k)
        cuts = np.sort(rng.integers(0, k + 1, size=a - 1))
        bounds = [0, *cuts, k]
        proj = []
        for u in range(a):
            D = np.zeros((k, k))
            D[bounds[u]:bounds[u + 1], bounds[u]:bounds[u + 1]] = np.eye(bounds[u + 1] - bounds[u])
            proj.append(V @ D @ V.conj().T)
        e.append(proj)
    return e


worst = {}


def record(name, ratio):
    worst[name] = max(worst.get(name, 0.0), ratio)


def run(q, a, k, s, theta):
    d = 1 + 2 * q * a
    e = instance(q, a, k)
    Ik = np.eye(k)
    # labels: 0 -> identity; (x,u) -> U label 1+2j, V label 2+2j
    specs = [(1.0, 0.0, None)]
    for x in range(q):
        for u in range(a):
            specs.append((I1, 1 - I1, (x, u)))
            specs.append((1.0, I1 - 1, (x, u)))
    w = [c * Ik + (b * e[j[0]][j[1]] if j else 0) for c, b, j in specs]
    for W in w:
        assert np.allclose(W.conj().T @ W, Ik)
    # (U5): Gram from p alone
    p = {(x, u, y, v): tr(e[x][u] @ e[y][v]).real for x in range(q) for u in range(a) for y in range(q) for v in range(a)}
    m = {(x, u): p[(x, u, x, u)] for x in range(q) for u in range(a)}
    B = np.zeros((d, d), complex)
    for kk, (ck, bk, j) in enumerate(specs):
        for ll, (cl, bl, h) in enumerate(specs):
            val = np.conj(ck) * cl
            if h:
                val += np.conj(ck) * bl * m[h]
            if j:
                val += np.conj(bk) * cl * m[j]
            if j and h:
                val += np.conj(bk) * bl * p[(*j, *h)]
            B[kk, ll] = val
    Bdirect = np.array([[tr(w[kk].conj().T @ w[ll]) for ll in range(d)] for kk in range(d)])
    assert np.allclose(B, Bdirect, atol=1e-12), "U5 mismatch"
    # target Choi (output index, input index ordering): <aa'|J|bb'> with J = (1/d) sum Psi(|k><l|) (x) |k><l|
    JPhi = np.zeros((d * d, d * d), complex)
    for kk in range(d):
        for ll in range(d):
            JPhi[kk * d + kk, ll * d + ll] = B[kk, ll] / d
    # nearby finite tracial factorization, bath dim mb = k*s
    mb = k * s
    U0 = np.zeros((d * mb, d * mb), complex)
    for kk in range(d):
        U0[kk * mb:(kk + 1) * mb, kk * mb:(kk + 1) * mb] = np.kron(w[kk].conj().T, np.eye(s))
    H = rng.normal(size=(d * mb, d * mb)) + 1j * rng.normal(size=(d * mb, d * mb))
    H = (H + H.conj().T) / 2
    H /= np.linalg.norm(H, 2)
    U1 = expm(1j * theta * H) @ U0
    A = [[U1[aa * mb:(aa + 1) * mb, bb * mb:(bb + 1) * mb] for bb in range(d)] for aa in range(d)]
    JPsi = np.zeros((d * d, d * d), complex)
    for aa in range(d):
        for kk in range(d):
            for bb in range(d):
                for ll in range(d):
                    JPsi[aa * d + kk, bb * d + ll] = np.trace(A[aa][kk] @ A[bb][ll].conj().T) / mb / d
    delta = 0.5 * np.abs(np.linalg.eigvalsh(JPsi - JPhi)).sum()
    t = np.sqrt(d * delta)
    if t == 0 or t > 1:
        return None
    Wk = [A[kk][kk].conj().T for kk in range(d)]
    leak = sum(1 - n2(W) ** 2 for W in Wk)
    record("leakage/(d delta)", leak / (d * delta))
    z = []
    for W in Wk:
        Us, sv, Vh = np.linalg.svd(W)
        z.append(Us @ Vh)
    record("sum||z-W||^2/t^2", sum(n2(z[i] - Wk[i]) ** 2 for i in range(d)) / t ** 2)
    Uk = [z[0].conj().T @ zz for zz in z]
    Im = np.eye(mb)
    # relations
    worst_pair = 0
    for x in range(q):
        for u in range(a):
            jidx = x * a + u
            Uju, Vjv = Uk[1 + 2 * jidx], Uk[2 + 2 * jidx]
            record("||U+V-(1+i)||/4t", n2(Uju + Vjv - (1 + I1) * Im) / (4 * t))
        S = sum(Uk[1 + 2 * (x * a + u)] for u in range(a))
        record("||sumU-(1+(a-1)i)||/(2sqrt2 a t)", n2(S - (1 + (a - 1) * I1) * Im) / (2 * np.sqrt(2) * a * t))
    X, P = {}, {}
    for x in range(q):
        for u in range(a):
            Uu = Uk[1 + 2 * (x * a + u)]
            X[(x, u)] = (Uu - I1 * Im) / (1 - I1)
            # Uu is unitary, hence normal: its complex Schur form gives orthonormal eigenvectors
            T, Z = schur(Uu, output="complex")
            lam = np.diag(T)
            keep = np.abs(lam - 1) <= np.abs(lam - I1)
            Zk = Z[:, keep]
            P[(x, u)] = Zk @ Zk.conj().T
            record("||P-X||/10t", n2(P[(x, u)] - X[(x, u)]) / (10 * t))
    E = {}
    for x in range(q):
        Sx = sum(P[(x, u)] for u in range(a))
        record("||S-I||/12at", n2(Sx - Im) / (12 * a * t))
        T = np.vstack([P[(x, u)] for u in range(a)])
        Us, sv, Vh = np.linalg.svd(T, full_matrices=False)
        Y = Us @ Vh
        assert np.allclose(Y.conj().T @ Y, Im)
        record("||Y-T||/12at", np.sqrt(np.real(np.trace((Y - T).conj().T @ (Y - T))) / mb) / (12 * a * t))
        for u in range(a):
            Yu = Y[u * mb:(u + 1) * mb, :]
            E[(x, u)] = Yu.conj().T @ Yu
            record("||E-X||/34at", n2(E[(x, u)] - X[(x, u)]) / (34 * a * t))
        assert np.allclose(sum(E[(x, u)] for u in range(a)), Im)
    dev = max(abs(tr(E[(x, u)] @ E[(y, v)]) - p[(x, u, y, v)])
              for x in range(q) for u in range(a) for y in range(q) for v in range(a))
    record("||p'-p||_inf/(91 a t)", dev / (91 * a * t))
    return delta, dev


count = 0
for trial in range(60):
    q = int(rng.integers(1, 3))
    a = int(rng.integers(2, 4))
    k = int(rng.integers(2, 5))
    s = int(rng.integers(1, 3))
    for theta in (1e-4, 1e-3, 1e-2, 3e-2, 0.1):
        r = run(q, a, k, s, theta)
        if r:
            count += 1
print("instances with 0<t<=1:", count)
for key, val in worst.items():
    flag = "OK" if val <= 1 + 1e-9 else "VIOLATION"
    print(f"{key:40s} worst ratio {val:.4f}  {flag}")
# unit-circle identity
th = np.linspace(0, 2 * np.pi, 100001)
zz = np.exp(1j * th)
lhs = np.abs(np.abs(1 + 1j - zz) ** 2 - 1)
rhs = np.sqrt(2) * np.abs(zz - 1) * np.abs(zz - 1j)
print("unit-circle identity max err", np.max(np.abs(lhs - rhs)),
      " min(rhs - dist)", np.min(rhs - np.minimum(np.abs(zz - 1), np.abs(zz - 1j))))
ok = count > 0 and all(val <= 1 + 1e-9 for val in worst.values())
ok = ok and np.max(np.abs(lhs - rhs)) < 1e-9
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
