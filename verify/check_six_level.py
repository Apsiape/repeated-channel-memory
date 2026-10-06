"""Checks for the six-level boundary direction, Theorem 6(b) and Section 7.

1. The ray of Section 7.2: bath unitaries u_1..u_6 on a qubit, Schur channel Psi_t with q_ij(t) = Tr rho_t u_i^* u_j,
   rho_t = (I + tZ)/2. Checks that the half-sum defects vanish, w(Psi_t) = t, the Choi spectrum of Psi_0 is
   (1/2, 1/4, 1/4), the Choi rank is 3 for |t| < 1, and d_J(Psi_t, Psi_0) = sqrt(15)|t|/12, equation (16).
2. Search for violations of the unitary-moment inequality (14), uniform in the matrix size m:
       |Im tau(V_1^* V_2)| <= (18 + 12 sqrt 2) E_V,
   E_V = ||V_4 - (V_1+V_2)/sqrt2||_2^2 + ||V_5 - (V_1+V_3)/sqrt2||_2^2 + ||V_6 - (V_2+V_3)/sqrt2||_2^2,
   by random search and by local maximization near the exact anticommuting configuration.
3. Search for violations of the supporting bound (13) on general unitary contacts with a flat bath C^m,
   random and near the saturating Schur contact.
Searches are evidence, not proof; the proofs are in Section 7.
Run: python verify/check_six_level.py   (exit code 0 on pass)
"""
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

rng = np.random.default_rng(3)
I2 = np.eye(2)
Xp = np.array([[0, 1], [1, 0]], dtype=complex)
Zp = np.array([[1, 0], [0, -1]], dtype=complex)
s2 = np.sqrt(2)
U_RAY = [I2, 1j * Zp, 1j * Xp, (I2 + 1j * Zp) / s2, (I2 + 1j * Xp) / s2, 1j * (Zp + Xp) / s2]


def ray_checks():
    u = U_RAY
    unit = max(np.abs(v.conj().T @ v - I2).max() for v in u)
    defects = [np.linalg.norm(u[3] - (u[0] + u[1]) / s2), np.linalg.norm(u[4] - (u[0] + u[2]) / s2),
               np.linalg.norm(u[5] - (u[1] + u[2]) / s2)]

    def q(t):
        rho = (I2 + t * Zp) / 2
        return np.array([[np.trace(rho @ u[i].conj().T @ u[j]) for j in range(6)] for i in range(6)])

    def choi(t):
        # Schur channel Psi(|i><j|) = q_ji |i><j|; normalized Choi = (1/6) sum_ij q_ji |ii><jj|
        J = np.zeros((36, 36), dtype=complex)
        qt = q(t)
        for i in range(6):
            for j in range(6):
                J[7 * i, 7 * j] = qt[j, i] / 6
        return J

    def w(t):
        qt = q(t)
        e = np.eye(6)
        a = [e[3] - (e[0] + e[1]) / s2, e[4] - (e[0] + e[2]) / s2, e[5] - (e[1] + e[2]) / s2]
        E = sum((ak.conj() @ qt @ ak).real for ak in a)
        L = sum(1 - qt[i, i].real for i in range(6))
        return qt[0, 1].imag - 35 * E - 240 * L

    ok = unit < 1e-12 and max(defects) < 1e-12
    ev0 = np.sort(np.linalg.eigvalsh(choi(0.0)))[::-1][:4]
    ok &= np.allclose(ev0, [0.5, 0.25, 0.25, 0.0], atol=1e-12)
    print("unitarity error", unit, "| half-sum defects", [float(d) for d in defects])
    print("top Choi eigenvalues at t = 0:", [round(float(x), 12) for x in ev0])
    for t in [0.3, -0.5, 0.9]:
        dJ = 0.5 * np.abs(np.linalg.eigvalsh(choi(t) - choi(0.0))).sum()
        rank = int((np.linalg.eigvalsh(choi(t)) > 1e-12).sum())
        print(f"t = {t}: w = {w(t):.12f}, d_J = {dJ:.12f}, sqrt(15)|t|/12 = {np.sqrt(15) * abs(t) / 12:.12f}, "
              f"Choi rank {rank}")
        ok &= abs(w(t) - t) < 1e-12 and abs(dJ - np.sqrt(15) * abs(t) / 12) < 1e-12 and rank == 3
    return bool(ok)


C14 = 18 + 12 * s2


def ratio(V):
    m = V[0].shape[0]
    tr = lambda A: np.trace(A) / m
    n2 = lambda A: tr(A.conj().T @ A).real
    E = n2(V[3] - (V[0] + V[1]) / s2) + n2(V[4] - (V[0] + V[2]) / s2) + n2(V[5] - (V[1] + V[2]) / s2)
    return abs(tr(V[0].conj().T @ V[1]).imag), E


def haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def exact_config(m):
    """Anticommuting reflections h, k on C^m (m even): V1 = I, V2 = i h, V3 = i k, V4..V6 the half sums."""
    k2 = m // 2
    h = np.kron(Zp, np.eye(k2))
    kk = np.kron(Xp, np.eye(k2))
    I = np.eye(m)
    V = [I, 1j * h, 1j * kk]
    V += [(V[0] + V[1]) / s2, (V[0] + V[2]) / s2, (V[1] + V[2]) / s2]
    return V


def neg(params, base, m):
    V = []
    off = 0
    for b in base:
        a = params[off:off + m * m].reshape(m, m)
        off += m * m
        H = (a + a.T) / 2 + 1j * (a - a.T) / 2
        V.append(b @ expm(1j * H))
    num, E = ratio(V)
    if E < 1e-14:
        return 0.0 if num < 1e-10 else -1e9
    return -num / E


def general_contact_witness(m, U):
    """w = Im q_12 - 35 E - 240 L for a flat bath C^m and a general contact U in U(6m), and the bound (13)."""
    D = [U[i * m:(i + 1) * m, i * m:(i + 1) * m] for i in range(6)]
    q = np.array([[np.trace(D[i].conj().T @ D[j]) / m for j in range(6)] for i in range(6)])
    e = np.eye(6)
    a = [e[3] - (e[0] + e[1]) / s2, e[4] - (e[0] + e[2]) / s2, e[5] - (e[1] + e[2]) / s2]
    E = sum((ak @ q @ ak).real for ak in a)
    L = sum(1 - q[i, i].real for i in range(6))
    w = q[0, 1].imag - 35 * E - 240 * L
    bound = -(17 - 12 * s2) * E - (119 - 84 * s2) * L
    return w, bound


if __name__ == "__main__":
    ok = ray_checks()
    worst = 0.0
    for m in [1, 2, 3, 4, 6]:
        for _ in range(4000):
            V = [haar(m) for _ in range(6)]
            num, E = ratio(V)
            if E > 1e-12:
                worst = max(worst, num / E)
    best = 0.0
    for m in [2, 4, 6]:
        base = exact_config(m)
        for trial in range(15):
            x0 = 10 ** rng.uniform(-4, -1) * rng.normal(size=6 * m * m)
            res = minimize(neg, x0, args=(base, m), method="L-BFGS-B", options={"maxiter": 1500})
            best = max(best, -res.fun)
    print(f"inequality (14): largest ratio, random {worst:.4f}, near the anticommuting configuration {best:.4f}; "
          f"constant {C14:.4f}")
    ok &= max(worst, best) <= C14
    worst_gap = -np.inf
    for m in [1, 2, 3, 4]:
        for _ in range(3000):
            w, bound = general_contact_witness(m, haar(6 * m))
            worst_gap = max(worst_gap, w - bound)
    for m, mult in [(2, 1), (4, 2)]:
        base = np.zeros((6 * m, 6 * m), dtype=complex)
        for i in range(6):
            base[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.kron(np.eye(mult), U_RAY[i])
        for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
            for _ in range(300):
                a = rng.normal(size=(6 * m, 6 * m)) + 1j * rng.normal(size=(6 * m, 6 * m))
                w, bound = general_contact_witness(m, base @ expm(1j * eps * (a + a.conj().T) / 2))
                worst_gap = max(worst_gap, w - bound)
    print("bound (13) on general contacts: largest value of w minus the bound (must be <= 0):", worst_gap)
    ok &= worst_gap <= 1e-9
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
