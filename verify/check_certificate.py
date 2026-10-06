"""First checker of the exact certificate for inequality (19) of Appendix B,

    D_U(X) <= (3/10) R_U(X),   D_U(X) = Tr X^2 Q_U,   R_U(X) = (1/3)||[U, I_3 (x) X]||_2^2,

with Q_U = (1/6) sum_{i<j} (u_ij - u_ji)^*(u_ij - u_ji) - (25/27) I. Appendix B turns (19) into Theorem 4.

Pure fractions.Fraction arithmetic; it does not import the second checker (certificates/rebuild_full_orbits.py)
or python-flint. Certificate format (Appendix F): symbols 0..8 = u_ij (index 3i+j), 9..17 = adjoints, 18 = X
(Hermitian). Each block gives column polynomials p_alpha = sum_k Q[k][alpha] w_k and a Gram matrix G (upper
triangle, row-major); its term is Re Tr sum G_ab p_a^* p_b, averaged over the six simultaneous permutations of the
qutrit labels. Relations are trace multipliers of unitarity relations, entering with -multiplier.

Checks:
  (1) exact: sum of averaged SOS terms - sum multiplier * averaged relations == target, as real cyclic trace classes,
      with the target derived here from the definitions of R and D (unitarity used only to reduce R);
  (2) exact: every Gram matrix is positive definite (Fraction LDL^T, all pivots > 0);
  (3) numeric: every averaged relation vanishes on Haar-random unitaries for bath sizes m = 1..4
      (the second checker verifies exactly that each relation is a block-unitarity consequence);
  (4) numeric: (3/10)R - D computed from matrix definitions equals the SOS side on random contacts (m = 1..5),
      and is nonnegative; also at the boundary contact F - 2 P_Omega.
Run: python verify/check_certificate.py [path to certificate]   (exit code 0 on pass)
"""
import json
import os
import sys
import itertools
from fractions import Fraction
from collections import defaultdict

import numpy as np

XS = 18
PERMS = list(itertools.permutations(range(3)))


def adj_sym(s):
    if s == XS:
        return XS
    return s + 9 if s < 9 else s - 9


def adjoint(word):
    return tuple(adj_sym(s) for s in reversed(word))


def canonical(word):
    word = tuple(word)
    if not word:
        return ()
    cands = []
    for w in (word, adjoint(word)):
        cands.extend(w[k:] + w[:k] for k in range(len(w)))
    return min(cands)


def permute(word, perm):
    out = []
    for s in word:
        if s == XS:
            out.append(XS)
        else:
            base = s % 9
            i, j = divmod(base, 3)
            out.append(3 * perm[i] + perm[j] + (9 if s >= 9 else 0))
    return tuple(out)


def avg_add(poly, word, coeff):
    if coeff == 0:
        return
    share = coeff / 6
    for perm in PERMS:
        poly[canonical(permute(word, perm))] += share


def frac(s):
    return Fraction(str(s))


def gram_matrix(block):
    c = block["columns"]
    upper = [frac(x) for x in block["gram_upper"]]
    assert len(upper) == c * (c + 1) // 2, "gram_upper length"
    G = [[Fraction(0)] * c for _ in range(c)]
    k = 0
    for i in range(c):
        for j in range(i, c):
            G[i][j] = upper[k]
            G[j][i] = upper[k]
            k += 1
    return G


def positive_definite(G):
    n = len(G)
    A = [row[:] for row in G]
    for k in range(n):
        piv = A[k][k]
        if piv <= 0:
            return False
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            f = A[i][k] / piv
            for j in range(k + 1, n):
                if A[k][j] != 0:
                    A[i][j] -= f * A[k][j]
    return True


def target_poly(C):
    t = defaultdict(Fraction)
    # (3/10) R = C * [2 Tr X^2 - (2/3) sum_ij Re Tr(X u_ij^* X u_ij)]
    avg_add(t, (XS, XS), 2 * C)
    for i in range(3):
        for j in range(3):
            a = 3 * i + j
            avg_add(t, (XS, a + 9, XS, a), -C * Fraction(2, 3))
    # - D = -(1/6) sum_{i!=j} [Tr X^2 u_ij^* u_ij - Re Tr X^2 u_ij^* u_ji] + (25/27) Tr X^2
    avg_add(t, (XS, XS), Fraction(25, 27))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            a, b = 3 * i + j, 3 * j + i
            avg_add(t, (XS, XS, a + 9, a), Fraction(-1, 6))
            avg_add(t, (XS, XS, a + 9, b), Fraction(1, 6))
    return t


def sos_poly(data, check_pd=True):
    poly = defaultdict(Fraction)
    pd_ok = True
    for block in data["blocks"]:
        words = [tuple(w) for w in block["words"]]
        c = block["columns"]
        G = gram_matrix(block)
        if check_pd and not positive_definite(G):
            pd_ok = False
        Q = [[Fraction(0)] * c for _ in words]
        for k, row in enumerate(block["q_sparse_rows"]):
            for col, val in row:
                Q[k][col] = frac(val)
        M = [[sum((Q[k][a] * G[a][b] for a in range(c) if Q[k][a] != 0), Fraction(0)) for b in range(c)]
             for k in range(len(words))]
        for k, wk in enumerate(words):
            for l, wl in enumerate(words):
                h = sum((M[k][b] * Q[l][b] for b in range(c) if Q[l][b] != 0), Fraction(0))
                if h != 0:
                    avg_add(poly, adjoint(wk) + wl, h)
    return poly, pd_ok


def relation_poly(data):
    poly = defaultdict(Fraction)
    for rel, mult in zip(data["relations"], data["relation_multipliers"]):
        lam = frac(mult)
        if lam == 0:
            continue
        for word, a in rel:
            avg_add(poly, tuple(word), -lam * int(a))
    return poly


# ---------- numerics ----------
rng = np.random.default_rng(2026)


def haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def herm(m):
    a = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    return (a + a.conj().T) / 2


def eval_word(word, U, X, m):
    mats = {}
    for i in range(3):
        for j in range(3):
            blk = U[i * m:(i + 1) * m, j * m:(j + 1) * m]
            mats[3 * i + j] = blk
            mats[3 * i + j + 9] = blk.conj().T
    mats[XS] = X
    P = np.eye(m, dtype=complex)
    for s in word:
        P = P @ mats[s]
    return np.trace(P).real


def eval_avg(word, U, X, m):
    return sum(eval_word(permute(word, p), U, X, m) for p in PERMS) / 6


def direct_side(U, X, m):
    blk = lambda i, j: U[i * m:(i + 1) * m, j * m:(j + 1) * m]
    comm = U @ np.kron(np.eye(3), X) - np.kron(np.eye(3), X) @ U
    R = (np.linalg.norm(comm) ** 2) / 3
    A = sum((blk(i, j) - blk(j, i)).conj().T @ (blk(i, j) - blk(j, i)) for i in range(3) for j in range(i + 1, 3)) / 6
    Qm = A - 25 / 27 * np.eye(m)
    D = np.trace(X @ X @ Qm).real
    return 0.3 * R - D


def sos_numeric(data, U, X, m):
    total = 0.0
    for block in data["blocks"]:
        words = [tuple(w) for w in block["words"]]
        c = block["columns"]
        G = np.array([[float(x) for x in row] for row in gram_matrix(block)])
        Q = np.zeros((len(words), c))
        for k, row in enumerate(block["q_sparse_rows"]):
            for col, val in row:
                Q[k, col] = float(frac(val))
        H = Q @ G @ Q.T
        for k, wk in enumerate(words):
            for l, wl in enumerate(words):
                if H[k, l] != 0:
                    total += H[k, l] * eval_avg(adjoint(wk) + wl, U, X, m)
    return total


def canonical_U0():
    F = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            F[3 * i + j, 3 * j + i] = 1
    om = np.zeros(9)
    for i in range(3):
        om[4 * i] = 1 / np.sqrt(3)
    return F - 2 * np.outer(om, om)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "certificates", "werner_commutator_c03_exact.json")
    data = json.load(open(path))
    C = frac(data["coefficient"])
    print("coefficient", C, "blocks", len(data["blocks"]), "relations", len(data["relations"]))
    sos, pd_ok = sos_poly(data)
    rel = relation_poly(data)
    tgt = target_poly(C)
    keys = set(sos) | set(rel) | set(tgt)
    diff = {k: sos[k] + rel[k] - tgt[k] for k in keys}
    nonzero = {k: v for k, v in diff.items() if v != 0}
    print("(2) all Gram blocks positive definite (exact):", pd_ok)
    print("(1) exact identity SOS - relations == target:", not nonzero, "| classes:", len(keys),
          "| mismatched:", len(nonzero))
    # (3) relations vanish on unitaries
    worst = 0.0
    for m in [1, 2, 3, 4]:
        for _ in range(2):
            U = haar(3 * m)
            X = herm(m)
            for rel_terms in data["relations"]:
                val = sum(int(a) * eval_avg(tuple(w), U, X, m) for w, a in rel_terms)
                worst = max(worst, abs(val))
    print("(3) max |averaged relation| on random unitaries:", worst)
    # (4) direct vs SOS side
    devs, mins = [], []
    for m in [1, 2, 3, 4, 5]:
        for _ in range(2):
            U = haar(3 * m)
            X = herm(m)
            d = direct_side(U, X, m)
            s = sos_numeric(data, U, X, m)
            devs.append(abs(d - s) / max(1.0, abs(d)))
            mins.append(d)
    U0 = canonical_U0()
    X0 = herm(3)
    d0 = direct_side(U0, X0, 3)
    s0 = sos_numeric(data, U0, X0, 3)
    print("(4) max relative |direct - SOS| on random contacts:", max(devs), "| min direct value:", min(mins))
    print("    saturating contact U0: direct", d0, "SOS", s0)
    ok = pd_ok and not nonzero and worst < 1e-9 and max(devs) < 1e-9 and min(mins) >= -1e-9
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
