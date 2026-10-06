"""Exact check of the constants of the fuel block, Appendix E.2, and the upper slope of Theorem 6(a).

Builds the contact U on system and bath qutrits from its definition (SWAP outside span{|ii>}, the reflection
I - 2 v v^T / 281 on it, v = (9, 10, 10)) in exact rational arithmetic, computes its antisymmetric bath effect
A = M_U^*(I) = (1/6) sum_{i<j} (u_ij - u_ji)^*(u_ij - u_ji), and checks
    A = diag(212521, 221941, 221941) / 236883,
    Delta_* = Tr(rho A) - 25/27 = 8737/4263894      for rho = diag(9/40, 31/80, 31/80),
    d_* = log2 3 - S(rho) = [(9/40) ln(27/40) + (31/40) ln(93/80)] / ln 2,
    c_* = d_* / Delta_* = 19.8967... < 19.897,
with d_* evaluated to 60 significant digits. Also checks that U is unitary and that the lower slope
5/(3 ln 2) is below c_*.
Run: python verify/check_fuel_constants.py   (exit code 0 on pass)
"""
import sys
from fractions import Fraction as F

from mpmath import mp, mpf, log

mp.dps = 60
v = [9, 10, 10]
nv = sum(x * x for x in v)                      # 281


def U_entry(r, c):
    """<r|U|c> for r = (i, a), c = (j, b): system index first, bath index second."""
    (i, a), (j, b) = r, c
    if i == a and j == b:                       # reflection on span{|kk>}
        return F(int(i == j)) - F(2 * v[i] * v[j], nv)
    if i == a or j == b:
        return F(0)
    return F(int(i == b and a == j))            # SWAP off the diagonal subspace


idx = [(i, a) for i in range(3) for a in range(3)]
U = [[U_entry(r, c) for c in idx] for r in idx]
unitary = all(sum(U[k][r] * U[k][c] for k in range(9)) == F(int(r == c)) for r in range(9) for c in range(9))


def block(i, j):                                # bath operator u_ij = <i|U|j>
    return [[U[3 * i + a][3 * j + b] for b in range(3)] for a in range(3)]


def mat_sub(X, Y):
    return [[X[a][b] - Y[a][b] for b in range(3)] for a in range(3)]


def gram(X):                                    # X^* X for a real matrix
    return [[sum(X[k][a] * X[k][b] for k in range(3)) for b in range(3)] for a in range(3)]


A = [[F(0)] * 3 for _ in range(3)]
for i in range(3):
    for j in range(i + 1, 3):
        G = gram(mat_sub(block(i, j), block(j, i)))
        A = [[A[a][b] + G[a][b] / 6 for b in range(3)] for a in range(3)]
A_expected = [[F(212521, 236883) if a == b == 0 else F(221941, 236883) if a == b else F(0) for b in range(3)]
              for a in range(3)]
rho = [F(9, 40), F(31, 80), F(31, 80)]
delta = sum(rho[k] * A[k][k] for k in range(3)) - F(25, 27)
d_star = (mpf(9) / 40 * log(mpf(27) / 40) + mpf(31) / 40 * log(mpf(93) / 80)) / log(2)
d_alt = log(3, 2) + sum(mpf(r.numerator) / r.denominator * log(mpf(r.numerator) / r.denominator, 2) for r in rho)
c_star = d_star / (mpf(delta.numerator) / delta.denominator)
lower = 5 / (3 * log(2))

checks = {
    "U is unitary": unitary,
    "A = diag(212521, 221941, 221941)/236883": A == A_expected,
    "Delta_* = 8737/4263894": delta == F(8737, 4263894),
    "d_* = log2 3 - S(rho)": abs(d_star - d_alt) < mpf(10) ** -50,
    "c_* < 19.897": c_star < mpf("19.897"),
    "5/(3 ln 2) < c_*": lower < c_star,
}
print("d_* =", mp.nstr(d_star, 20), "| c_* =", mp.nstr(c_star, 20), "| ratio to the lower slope:",
      mp.nstr(c_star / lower, 6))
for name, ok in checks.items():
    print("PASS" if ok else "FAIL", name)
ok = all(checks.values())
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
