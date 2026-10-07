"""Checks for Theorem 7 (strong converse outside the line), Proposition 8.1, Corollary 8.2 and Appendix G.

Exact part: the constants of the proof (the tilt interval, the exponent 275/2592, 6.6, 0.306, the ideal exponent 1/8),
the constants of Corollary 8.2 and of the sparse-testing remark of Section 6.
Numerical part, on qubit tests: K_2 is the set of unital qubit channels, whose extreme points are unitary
conjugations, so the closure value p0 of any effect E is the largest eigenvalue of a real 4x4 matrix (exact up to
floating point). Random contacts U on C^2 (x) C^D test Lemma G.1 (both inequalities), Lemma G.2 (both, and the
pre-margin form), Lemma G.3 at the edge t = ln(1 + eta), and Proposition G.4 on three-round devices against an
explicit expansion over all records. A short local search pushes the Lemma G.2 ratio. Finally the coin device of
Proposition 7.1 is simulated for amplitude damping over two rounds.
Run: python -u check_strong_converse.py   (exit code 0 if and only if every check passes)
"""
import sys
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize
from scipy.special import expit

rng = np.random.default_rng(20261006)
FAIL = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


# ---------------------------------------------------------------- exact constants
D_ = Fr(1)                                   # coefficients are homogeneous in Delta; take Delta = 1
f = lambda t: t * t / 8 - Fr(5, 12) * D_ * t  # t^2/8 - 5 Delta t / 12
check(f(Fr(5, 18)) == Fr(-275, 2592), "exponent at t = 5 Delta/18 equals -275/2592 Delta^2")
check(Fr(275, 2592) > Fr(1, 10), "275/2592 > 1/10")
check(Fr(5, 18) + Fr(5, 18) < Fr(5, 3) and Fr(1, 3) < Fr(5, 6), "f decreases on [5/18, 1/3] (vertex at 5/3)")
check(2 * Fr(1, 4) ** 2 == Fr(1, 8), "ideal Hoeffding exponent 2 (Delta/4)^2 = Delta^2/8")
check(Fr(3, 4) - Fr(1, 3) == Fr(5, 12), "threshold minus tilt centre = 5 Delta/12")
mp.mp.dps = 40
check(2592 * mp.log(2) / 275 < mp.mpf('6.6'), "2592 ln2 / 275 = %s < 6.6" % mp.nstr(2592 * mp.log(2) / 275, 8))
cq = 275 * mp.log(mp.e, 2) / 1296
check(cq > mp.mpf('0.306'), "memory coefficient 275 log2(e)/1296 = %s > 0.306" % mp.nstr(cq, 8))
grid = [mp.mpf(k) / 2000 for k in range(1, 2001)]
check(all(mp.mpf(5) * u / 18 <= mp.log(1 + u / 3) <= u / 3 for u in grid),
      "5 Delta/18 <= ln(1 + Delta/3) <= Delta/3 on a grid of (0, 1]")
ok = True
for _ in range(2000):
    p, eta = rng.uniform(0, 1), rng.uniform(1e-3, 1)
    for t in (np.log1p(eta) * rng.uniform(0.01, 1), np.log1p(eta)):
        v = np.expm1(t) / (1 + p * np.expm1(t))
        ok &= v * (1 / eta + p) <= 1 + 1e-12
        a = p * np.exp(t * (1 - p)) + (1 - p) * np.exp(-t * p)
        ok &= np.log(a) <= t * t / 8 + 1e-15
check(ok, "v(1/eta + p) <= 1 when e^t - 1 <= eta; Hoeffding's lemma ln a <= t^2/8 (2000 random cases)")

# Corollary 8.2 (random checks) and the sparse-testing remark of Section 6
check(2 * Fr(1, 8) ** 2 == Fr(1, 32) and Fr(1, 32) >= Fr(1, 128) and 2 * Fr(1, 16) ** 2 == Fr(1, 128),
      "Corollary 8.2 Hoeffding exponents: real side 2(1/8)^2 = 1/32 >= 1/128, ideal sides 2(1/16)^2 = 1/128")
check(Fr(7, 8) - Fr(3, 4) == Fr(1, 8) and Fr(15, 16) - Fr(7, 8) == Fr(1, 16) and 1 - Fr(15, 16) == Fr(1, 16),
      "Corollary 8.2 gaps: threshold 7/8 sits 1/8 above 3/4 and 1/16 below 15/16, which sits 1/16 below 1")
check(3 * mp.mpf('6.6') == mp.mpf('19.8') and mp.mpf('19.8') <= 20,
      "Corollary 8.2: two tails at most 2(1-eps)/3 leave 6.6(B+1)/(n Delta^2) >= (1-eps)/3, i.e. B >= (1-eps) n Delta^2/19.8 - 1")
sigma = 5 * mp.sqrt(2) / 27
check(abs(1 / (sigma * mp.sqrt(2)) - mp.mpf(27) / 10) < mp.mpf(10) ** -35 and abs(sigma ** 2 - Fr(25, 27) * Fr(2, 27)) < 1e-30,
      "sparse-testing remark: sigma^2 = p(1-p) and 1/(sigma sqrt 2) = 27/10 exactly")

# ---------------------------------------------------------------- qubit tests and contacts
d = 2
OMEGA = np.zeros(d * d, complex)
for i in range(d):
    OMEGA[i * d + i] = 1 / np.sqrt(d)        # |i>_A |i>_R
PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]


def p0_unital(E):
    """max over unital qubit channels of Tr[E J]; extreme points are unitary conjugations a0 I + i a.sigma."""
    V = np.stack([OMEGA] + [1j * np.kron(s, np.eye(2)) @ OMEGA for s in PAULI[1:]], axis=1)
    return float(np.linalg.eigvalsh(np.real(V.conj().T @ E @ V)).max())


def random_effect(kind):
    if kind == "projector1":
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v /= np.linalg.norm(v)
        return np.outer(v, v.conj())
    if kind == "projector2":
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        return Q[:, :2] @ Q[:, :2].conj().T
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = G @ G.conj().T
    return H / np.linalg.eigvalsh(H).max()


def haar(n):
    Q, R = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return Q * (np.diag(R) / abs(np.diag(R)))


def blocks(U, D):
    return [[U[a * D:(a + 1) * D, b * D:(b + 1) * D] for b in range(d)] for a in range(d)]


def instrument(U, E, D):
    """Superoperator matrices (row-major vec) of T_U and M_U on the bath."""
    B = blocks(U, D)
    T = sum(np.kron(B[a][b], B[a][b].conj()) for a in range(d) for b in range(d)) / d
    M = np.zeros((D * D, D * D), complex)
    for a in range(d):
        for b in range(d):
            for a2 in range(d):
                for b2 in range(d):
                    M += E[a2 * d + b2, a * d + b] * np.kron(B[a][b], B[a2][b2].conj())
    return T, M / d


app = lambda S, X: (S @ X.reshape(-1)).reshape(X.shape)
adj = lambda S, X: (S.conj().T @ X.reshape(-1)).reshape(X.shape)
hs = lambda X: np.linalg.norm(X)


def choi_round(U, rho, D):
    """J(Psi_{U,rho}) on (out, ref), for the consistency check Tr M(rho) = Tr[E J]."""
    J = np.zeros((4, 4), complex)
    B = blocks(U, D)
    for a in range(d):
        for b in range(d):
            for a2 in range(d):
                for b2 in range(d):
                    J[a * d + b, a2 * d + b2] = np.trace(B[a][b] @ rho @ B[a2][b2].conj().T) / d
    return J


def rand_psd(D, kind):
    if kind == "flatish":
        X = np.eye(D) + 0.05 * (lambda G: G + G.conj().T)(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
    else:
        r = D if kind == "full" else rng.integers(1, D)
        G = rng.normal(size=(D, r)) + 1j * rng.normal(size=(D, r))
        X = G @ G.conj().T
    w, v = np.linalg.eigh(X)
    return (v * np.clip(w, 0, None)) @ v.conj().T


def rand_proj(D, k):
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
    return Q[:, :k] @ Q[:, :k].conj().T


EFFECTS = ["projector1", "projector2", "general"]

# A separating test for amplitude damping (gamma = 0.6) against K_2, found by local search over effects
GAM = 0.6
KAD = [np.array([[1, 0], [0, np.sqrt(1 - GAM)]]), np.array([[0, np.sqrt(GAM)], [0, 0]])]
J_AD = sum(np.outer(np.kron(Kk, np.eye(2)) @ OMEGA, (np.kron(Kk, np.eye(2)) @ OMEGA).conj()) for Kk in KAD)


def effect_from(z):
    H = (z[:16].reshape(4, 4) + 1j * z[16:32].reshape(4, 4))
    w, v = np.linalg.eigh(H + H.conj().T)
    return (v * expit(4 * w)) @ v.conj().T


def margin(z):
    E = effect_from(z)
    return -(np.trace(E @ J_AD).real - p0_unital(E))


best_z, best_m = None, -1
for s0 in range(6):
    r = minimize(margin, rng.normal(size=32), method="Nelder-Mead", options=dict(maxiter=6000, fatol=1e-12))
    if -r.fun > best_m:
        best_m, best_z = -r.fun, r.x
E_AD = effect_from(best_z)
P0_AD = p0_unital(E_AD)
DELTA_AD = np.trace(E_AD @ J_AD).real - P0_AD
check(DELTA_AD > 0.2, "separating test for amplitude damping: q - p0 = %.4f (dist_J to unital channels is 0.3)" % DELTA_AD)
VAD = sum(np.kron(Kk, np.eye(2)[:, [k]]) for k, Kk in enumerate(KAD))   # (sys, env) Stinespring isometry
_c = np.linalg.qr(np.eye(4)[:, [1, 3]] - VAD @ (VAD.conj().T @ np.eye(4)[:, [1, 3]]))[0]
W_AD = np.zeros((4, 4), complex)
W_AD[:, [0, 2]], W_AD[:, [1, 3]] = VAD, _c          # W(psi (x) |0>_env) = V psi


def near_dilation(D, scale):
    """The amplitude-damping dilation on C^2 (x) C^D (bath = env (x) C^{D/2} or env alone), slightly perturbed."""
    U0 = W_AD if D == 2 else np.kron(W_AD, np.eye(D // 2))
    G = rng.normal(size=(d * D, d * D)) + 1j * rng.normal(size=(d * D, d * D))
    return expm(1j * scale * (G + G.conj().T) / 2) @ U0
worst = dict(consist=0, g1a=-np.inf, g1b=-np.inf, pre=-np.inf, g2a=-np.inf, g2b=-np.inf, g3=0.0)
for trial in range(600):
    if trial % 3 == 0:
        D = [2, 4][trial % 2]
        E, p0, U = E_AD, P0_AD, near_dilation(D, rng.uniform(0, 0.3))
    else:
        D = int(rng.integers(2, 5))
        E = random_effect(EFFECTS[trial % 3])
        p0 = p0_unital(E)
        U = haar(d * D)
    T, M = instrument(U, E, D)
    rho = rand_psd(D, "full")
    rho /= np.trace(rho).real
    worst["consist"] = max(worst["consist"], abs(np.trace(app(M, rho)) - np.trace(E @ choi_round(U, rho, D))))
    k = int(rng.integers(1, D + 1))
    P, P2 = rand_proj(D, k), rand_proj(D, k)
    I = np.eye(D)
    worst["g1a"] = max(worst["g1a"], np.trace(P @ adj(M, I)).real - k * p0 - np.trace((I - P2) @ app(T, P)).real)
    worst["g1b"] = max(worst["g1b"], np.trace(P @ app(M, I)).real - k * p0 - np.trace(adj(T, P) @ (I - P2)).real)
    X = rand_psd(D, ["full", "deficient", "flatish"][trial % 3]) if trial % 3 else \
        np.diag([1.0] + [rng.uniform(0, 0.3)] * (D - 1)).astype(complex)
    eta = rng.uniform(0.01, 0.3)
    nx2 = hs(X) ** 2
    Ra, Rb = nx2 - hs(app(T, X)) ** 2, nx2 - hs(adj(T, X)) ** 2
    X2 = X @ X
    worst["pre"] = max(worst["pre"], np.trace(X2 @ (adj(M, I) - p0 * I)).real - np.sqrt(nx2) * np.sqrt(2 * max(Ra, 0)))
    worst["g2a"] = max(worst["g2a"], np.trace(X2 @ (adj(M, I) - (p0 + eta) * I)).real - Ra / (2 * eta))
    worst["g2b"] = max(worst["g2b"], np.trace(X2 @ (app(M, I) - (p0 + eta) * I)).real - Rb / (2 * eta))
    p, t = p0 + eta, np.log1p(eta)
    a = p * np.exp(t * (1 - p)) + (1 - p) * np.exp(-t * p)
    L = np.exp(t * (1 - p)) * M + np.exp(-t * p) * (T - M)
    worst["g3"] = max(worst["g3"], hs(app(L, X)) / (a * hs(X)))
tol = 1e-9
check(worst["consist"] < 1e-12, "Tr M_U(rho) = Tr[E J(Psi_{U,rho})] (max error %.1e)" % worst["consist"])
check(worst["g1a"] <= tol, "Lemma G.1, first inequality: max excess %.2e (600 random cases)" % worst["g1a"])
check(worst["g1b"] <= tol, "Lemma G.1, second inequality: max excess %.2e" % worst["g1b"])
check(worst["pre"] <= tol, "Lemma G.2 pre-margin form ||X|| sqrt(2 R): max excess %.2e" % worst["pre"])
check(worst["g2a"] <= tol, "Lemma G.2, first inequality: max excess %.2e" % worst["g2a"])
check(worst["g2b"] <= tol, "Lemma G.2, second inequality: max excess %.2e" % worst["g2b"])
check(worst["g3"] <= 1 + tol, "Lemma G.3 at t = ln(1 + eta): max ratio ||L X|| / (a ||X||) = %.6f" % worst["g3"])


# local search on Lemma G.2 (first inequality): maximize the excess ratio over U and X
def g2_ratio(z, D, E, p0, eta):
    n = (d * D) ** 2
    H = z[:n].reshape(d * D, d * D)
    U = expm(1j * (H + H.T) / 2)
    G = z[n:n + 2 * D * D].reshape(2, D, D)
    X = (G[0] + 1j * G[1]) @ (G[0] + 1j * G[1]).conj().T
    T, M = instrument(U, E, D)
    lhs = np.trace(X @ X @ (adj(M, np.eye(D)) - (p0 + eta) * np.eye(D))).real
    R = hs(X) ** 2 - hs(app(T, X)) ** 2
    return -(lhs / (R / (2 * eta) + 1e-12)) if R > 1e-10 else 0.0


best = -np.inf
for s in range(4):
    D, E, p0, eta = 2, E_AD, P0_AD, [0.02, 0.05, 0.1, 0.2][s]
    Hlog = -1j * logm(W_AD)
    Hlog = (Hlog + Hlog.conj().T) / 2
    z0 = np.concatenate([np.real(Hlog).ravel() + rng.normal(scale=0.05, size=16),
                         np.array([1, 0, 0, 0.1]) + rng.normal(scale=0.05, size=4), rng.normal(scale=0.05, size=4)])
    res = minimize(g2_ratio, z0, args=(D, E, p0, eta), method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-9, fatol=1e-12))
    best = max(best, -res.fun)
check(0 < best <= 1 + 1e-6, "Lemma G.2 local search from the dilation (eta = 0.02..0.2): "
      "largest lhs / (R/(2 eta)) = %.4f" % best)

# ---------------------------------------------------------------- Proposition G.4 on three-round devices
ok_id, ok_bd, worst_ratio = True, True, 0.0
for trial in range(60):
    n = 3
    if trial % 2 == 0:
        D, E, p0, eta = 2, E_AD, P0_AD, rng.uniform(0.02, 0.2)
    else:
        D = int(rng.integers(2, 4))
        E = random_effect(EFFECTS[trial % 3])
        p0, eta = p0_unital(E), rng.uniform(0.02, 0.3)
    p, t = p0 + eta, np.log1p(eta)
    a = p * np.exp(t * (1 - p)) + (1 - p) * np.exp(-t * p)
    Om = rand_psd(D, ["full", "deficient"][trial % 2])
    Om /= np.trace(Om).real
    insts = [instrument(near_dilation(D, 0.1) if trial % 2 == 0 else haar(d * D), E, D) for _ in range(n)]
    Y = Om.copy()
    for T, M in insts:
        Y = app(np.exp(t * (1 - p)) * M + np.exp(-t * p) * (T - M), Y)
    tilted = np.trace(Y).real
    branches = {(): Om}                      # explicit expansion over all records
    for T, M in insts:
        branches = {x + (o,): app(M if o else T - M, r) for x, r in branches.items() for o in (0, 1)}
    expansion = sum(np.exp(t * (sum(x) - n * p)) * np.trace(r).real for x, r in branches.items())
    total = sum(np.trace(r).real for r in branches.values())
    ok_id &= abs(tilted - expansion) < 1e-12 and abs(total - 1) < 1e-12
    B2 = np.log2(D * np.trace(Om @ Om).real)
    worst_ratio = max(worst_ratio, tilted / (2 ** (B2 / 2) * a ** n))
check(ok_id, "Proposition G.4: Tr L3 L2 L1(Omega) equals the expansion over records; probabilities sum to 1")
check(worst_ratio <= 1 + 1e-12, "Proposition G.4: max E e^{t(S-np)} / (2^{B2/2} a^n) = %.4f (60 devices)" % worst_ratio)

# ---------------------------------------------------------------- Proposition 7.1: the coin device
g, n, prob = 0.6, 2, 0.37
K = [np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])]
Vst = sum(np.kron(Kk, np.eye(2)[:, [k]]) for k, Kk in enumerate(K))       # C^2 -> C^2 (x) C^2 (out, env)
Q_, _ = np.linalg.qr(np.hstack([Vst, rng.normal(size=(4, 2))]))
W = np.zeros((4, 4), complex)                 # W(psi (x) |0>) = V psi on (sys, env)
W[:, [0, 2]] = Vst
comp = Q_[:, 2:] - Vst @ (Vst.conj().T @ Q_[:, 2:])
comp, _ = np.linalg.qr(comp)
W[:, [1, 3]] = comp
check(np.allclose(W.conj().T @ W, np.eye(4)), "coin device: completed Stinespring unitary W")
# registers: A1 R1 A2 R2 coin c1 c2 (qubits); the cells shift cyclically, so round k uses original cell k
psi_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
ket0, ket1 = np.array([1, 0]), np.array([0, 1])


def final_state(coin, cells):
    """Output-reference state of both rounds for a pure coin value and a pure product cell state."""
    rhoA = np.zeros((16, 16), complex)
    for c_vecs in cells:                      # cells given as list of (weight, (vec1, vec2))
        wgt, (v1, v2) = c_vecs
        st = np.kron(np.kron(psi_bell, psi_bell), np.kron(v1, v2))      # A1 R1 A2 R2 c1 c2
        st = st.reshape([2] * 6)
        if coin == 0:
            for (A, C) in ((0, 4), (2, 5)):  # round 1: (A1, c1); round 2: (A2, c2)
                st = np.moveaxis(st, (A, C), (0, 1))
                sh = st.shape
                st = (W @ st.reshape(4, -1)).reshape(sh)
                st = np.moveaxis(st, (0, 1), (A, C))
        st = st.reshape(16, 4)
        rhoA += wgt * st @ st.conj().T        # trace out the cells
    return rhoA


pure_cells = [(1.0, (ket0, ket0))]
mixed_cells = [(0.25, (u, v)) for u in (ket0, ket1) for v in (ket0, ket1)]
rho = prob * final_state(0, pure_cells) + (1 - prob) * final_state(1, mixed_cells)
JPhi = sum(np.outer(np.kron(Kk, np.eye(2)) @ psi_bell, (np.kron(Kk, np.eye(2)) @ psi_bell).conj()) for Kk in K)
Jid = np.outer(psi_bell, psi_bell)
target = prob * np.kron(JPhi, JPhi) + (1 - prob) * np.kron(Jid, Jid)
check(np.allclose(rho, target, atol=1e-13), "coin device: final state = p J(Phi)^{(x)2} + (1-p) J(id)^{(x)2}")
h2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x)
Qm, R = 1 + n * 1, 1
S_beta = h2(prob) + (1 - prob) * n * R
check(abs((Qm - S_beta) - (1 - h2(prob) + prob * n * R)) < 1e-12, "coin device: P = 1 - h2(p) + p n R")
B2 = np.log2(2 ** Qm * (prob ** 2 + (1 - prob) ** 2 * 2.0 ** (-n * R)))
check(B2 >= n * R + 2 * np.log2(prob), "coin device: B2 = %.4f >= nR + 2 log p = %.4f" % (B2, n * R + 2 * np.log2(prob)))

print("ALL PASS" if not FAIL else "FAILURES: %d" % len(FAIL))
sys.exit(1 if FAIL else 0)
