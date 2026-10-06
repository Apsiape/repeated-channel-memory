"""Numerical search for violations of the score envelope, Theorem 4, inequality (6):

    Tr X^2 Q_U <= (3/5) [Tr X^2 - Tr (T_U X)^2]   for Hermitian X on the bath,

where U = sum_ij |i><j| (x) u_ij is a contact on C^3 (x) C^m, the support-test bath instrument is
    M_-(X) = (1/6) sum_{i<j} (u_ij - u_ji) X (u_ij - u_ji)^*,
    M_+(X) = (1/3) sum_i u_ii X u_ii^* + (1/6) sum_{i<j} (u_ij + u_ji) X (u_ij + u_ji)^*,
T_U = M_- + M_+, and Q_U = M_-^*(I) - p I with p = 25/27.

Written from these definitions only; it does not use the certificate. Reports the largest ratio
Tr X^2 Q / [Tr X^2 - Tr (T X)^2] found by random search and by local maximization started near the boundary
contact U_0 = F - 2 P_Omega (tensored with the identity on an extra bath factor) and near random contacts.
A ratio above 3/5 would refute the theorem; none is evidence, not proof. The exact proof is the certificate
(check_certificate.py and certificates/rebuild_full_orbits.py).
Run: python verify/check_envelope_search.py   (exit code 0 on pass)
"""
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

rng = np.random.default_rng(11)
p0 = 25 / 27
CONSTANT = 3 / 5


def blocks(U, m):
    return [[U[i * m:(i + 1) * m, j * m:(j + 1) * m] for j in range(3)] for i in range(3)]


def kraus(U, m):
    u = blocks(U, m)
    minus = [(u[i][j] - u[j][i]) / np.sqrt(6) for i in range(3) for j in range(i + 1, 3)]
    plus = [u[i][i] / np.sqrt(3) for i in range(3)] + \
           [(u[i][j] + u[j][i]) / np.sqrt(6) for i in range(3) for j in range(i + 1, 3)]
    return minus, plus


def ratio_parts(U, m, X):
    minus, plus = kraus(U, m)
    A = sum(k.conj().T @ k for k in minus)
    Q = A - p0 * np.eye(m)
    TX = sum(k @ X @ k.conj().T for k in minus + plus)
    num = np.trace(X @ X @ Q).real
    den = (np.trace(X @ X) - np.trace(TX @ TX)).real
    return num, den


def haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def herm(n, scale=1.0):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (a + a.conj().T) / 2


def canonical_U0():
    F = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            F[3 * i + j, 3 * j + i] = 1
    om = np.zeros(9)
    for i in range(3):
        om[4 * i] = 1 / np.sqrt(3)
    return F - 2 * np.outer(om, om)


def sanity():
    U0 = canonical_U0()
    assert np.allclose(U0 @ U0.T, np.eye(9))
    minus, plus = kraus(U0, 3)
    T_star_I = sum(k.conj().T @ k for k in minus + plus)
    T_I = sum(k @ k.conj().T for k in minus + plus)
    A = sum(k.conj().T @ k for k in minus)
    return bool(np.allclose(T_star_I, np.eye(3)) and np.allclose(T_I, np.eye(3)) and np.allclose(A, p0 * np.eye(3)))


def neg_ratio(params, m, base):
    k = 3 * m
    a = params[:k * k].reshape(k, k)
    H = (a + a.T) / 2 + 1j * (a - a.T) / 2
    U = base @ expm(1j * 0.5 * H)
    b = params[k * k:].reshape(m, m)
    X = (b + b.T) / 2 + 1j * (b - b.T) / 2 + np.eye(m)
    num, den = ratio_parts(U, m, X)
    if den < 1e-12:
        return 0.0 if num <= 1e-12 else -1e6
    return -num / den


if __name__ == "__main__":
    ok = sanity()
    print("boundary contact U_0: T_U and T_U^* unital, antisymmetric effect = p I:", ok)
    worst = 0.0
    for m in [1, 2, 3, 4, 6]:
        for _ in range(3000 if m <= 3 else 800):
            U = haar(3 * m)
            X = herm(m)
            num, den = ratio_parts(U, m, X)
            if den > 1e-10:
                worst = max(worst, num / den)
            elif num > 1e-9:
                print("VIOLATION with zero gap", m, num, den)
                worst = np.inf
    print("random search, largest ratio:", worst)
    best = 0.0
    for m in [3, 6]:
        base0 = np.kron(canonical_U0(), np.eye(m // 3))   # extra bath factor as the inner index
        k = 3 * m
        for trial in range(12):
            base = base0 if trial % 2 == 0 else haar(k)
            x0 = np.concatenate([0.05 * rng.normal(size=k * k), 0.5 * rng.normal(size=m * m)])
            res = minimize(neg_ratio, x0, args=(m, base), method="L-BFGS-B", options={"maxiter": 400})
            best = max(best, -res.fun)
    print("local maximization, largest ratio:", best)
    largest = max(worst, best)
    ok = ok and largest <= CONSTANT
    print(f"largest ratio found {largest:.4f} against the constant 3/5")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
