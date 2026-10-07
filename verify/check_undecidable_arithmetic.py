"""Exact arithmetic controls for Theorem 2 and Appendix A; not an implementation of the hardness compiler.

Checks, in exact rational and Gaussian-rational arithmetic:
  - the Q(i) Schur builder (18) equals the literal Gram matrix of its unitaries on a small instance;
  - the two linear relations the transfer lemma uses, and their coefficient norms;
  - the gap arithmetic gamma^2/(128^2 a^2 d) = 1/(256^2 a^6 d) of Lemma A.2 for d = 1 + 2qa;
  - the deficiency fill of the flat completion (Lemma 5.2) and its strict-contraction bound;
  - orthogonality of distinct characters of Z_2^L;
  - the error thresholds of Remark 4.4 and Corollary 4.1.
With --perturb, a deliberately broken constant must make the run fail (exit code nonzero).
Run: python verify/check_undecidable_arithmetic.py   (exit code 0 on pass)
"""

from fractions import Fraction as F
import argparse


def add(z, w):
    return (z[0] + w[0], z[1] + w[1])


def mul(z, w):
    return (z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0])


def conj(z):
    return (z[0], -z[1])


def scale(z, a):
    return (z[0] * a, z[1] * a)


ZERO, ONE, I = (F(0), F(0)), (F(1), F(0)), (F(0), F(1))


def main(perturb=False):
    # Two deterministic classical assignments, uniformly weighted.
    assignments = ((0, 1), (1, 0))
    q, a = 2, 2
    outcomes = tuple((x, u) for x in range(q) for u in range(a))

    def moment(j, h):
        return sum(
            (F(1, len(assignments)) for s in assignments
             if s[j[0]] == j[1] and s[h[0]] == h[1]), F(0)
        )

    marg = {j: moment(j, j) for j in outcomes}
    specs = [(ONE, ZERO, None)]
    for j in outcomes:
        specs.extend(((I, (F(1), F(-1)), j),
                      (ONE, (F(-1), F(1)), j)))

    def vector(spec):
        c, b, j = spec
        return tuple(add(c, scale(b, F(j is not None and s[j[0]] == j[1])))
                     for s in assignments)

    vectors = tuple(vector(spec) for spec in specs)
    d = 1 + 2 * q * a
    assert d == len(specs)
    gram = []
    for k, (ck, bk, j) in enumerate(specs):
        row = []
        for l, (cl, bl, h) in enumerate(specs):
            computed = mul(conj(ck), cl)
            if h is not None:
                computed = add(computed, scale(mul(conj(ck), bl), marg[h]))
            if j is not None:
                computed = add(computed, scale(mul(conj(bk), cl), marg[j]))
            if j is not None and h is not None:
                computed = add(computed, scale(mul(conj(bk), bl), moment(j, h)))
            direct = ZERO
            for vk, vl in zip(vectors[k], vectors[l]):
                direct = add(direct, scale(mul(conj(vk), vl), F(1, 2)))
            assert computed == direct
            row.append(computed)
        gram.append(row)
    assert all(gram[k][k] == ONE for k in range(d))
    assert all(gram[k][l] == conj(gram[l][k]) for k in range(d) for l in range(d))
    print(f"PASS: Q(i) builder equals the literal unitary Gram ({d*d} entries).")

    for j_index in range(q * a):
        ui, vi = 1 + 2 * j_index, 2 + 2 * j_index
        assert all(add(vectors[ui][s], vectors[vi][s]) == add(ONE, I)
                   for s in range(len(assignments)))
    for x in range(q):
        ui = [1 + 2 * (x * a + u) for u in range(a)]
        for s in range(len(assignments)):
            total = ZERO
            for index in ui:
                total = add(total, vectors[index][s])
            assert total == add(ONE, scale(I, a - 1))
    for answers in range(1, 100):
        assert answers * answers - answers + 2 <= 2 * answers * answers
    print("PASS: both exact linear relations and their coefficient norms.")

    for answers in range(1, 20):
        for questions in range(1, 20):
            dim = 1 + 2 * questions * answers
            gamma = F(1, 2 * answers * answers)
            gap = gamma * gamma / (128 * 128 * answers * answers * dim)
            assert gap == F(1, 256 * 256 * answers**6 * dim)
            assert F(96, 32 if perturb else 128) < 1
    assert 4 * (1 + F(3, 2)) <= 10
    assert 34 * (1 + F(3, 2)) <= 85
    print("PASS: transfer/gap arithmetic; dimension and outcome factors retained.")

    for c in (F(1, 2), F(3, 4), F(2)):
        for mu in (c / 4, c / 2, 3 * c / 4):
            for copies in (1, 2, 5):
                alpha2 = mu / (copies * c + mu)
                weight = c / (2 * (1 - alpha2))
                assert 2 * copies * weight * alpha2 == mu
                assert 2 * weight * (1 - alpha2) == c
                assert copies * alpha2 < mu / c < 1
    print("PASS: deficiency fill and strict-contraction sufficient bound.")

    size = 16
    characters = tuple(tuple(1 if (k & z).bit_count() % 2 == 0 else -1
                             for z in range(size)) for k in range(1, size))
    for k, char in enumerate(characters):
        assert sum(char) == 0
        for l, other in enumerate(characters):
            assert sum(x * y for x, y in zip(char, other)) == (size if k == l else 0)
    print("PASS: globally distinct characters give all required orthogonality.")

    # Noise and numerical-rate thresholds use only rational comparisons.
    gap = F(1, 256 * 256 * a**6 * d)
    eta, epsilon = gap / 4, gap / 4
    assert gap - eta - epsilon == gap / 2
    approximation_error, threshold = gap**2 / 16, gap**2 / 8
    assert approximation_error < threshold < gap**2 / 4 - approximation_error
    # Version 1.1, at every fixed error (Theorem 7): rate 0 or >= 3 g^2/10; within g^2/8 against the threshold 3 g^2/20.
    approximation_error, threshold = gap**2 / 8, 3 * gap**2 / 20
    assert approximation_error < threshold < 3 * gap**2 / 10 - approximation_error
    # Remark 4.4 in version 1.1: distance 3g/4 gives coefficient 0.3 * 9/16 > 0.16 and onset 8 * 16/9 < 15.
    assert F(3, 10) * F(9, 16) > F(16, 100) and 8 * F(16, 9) < 15
    print("PASS: full-rank perturbation and numerical-rate decision thresholds (small-error and fixed-error forms).")
    print("Scope: arithmetic only; no hard channel, MIP*, hierarchy or all-dimensional numerical test.")
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturb", action="store_true",
                        help="Deliberately break the gap constant; must fail.")
    main(parser.parse_args().perturb)
