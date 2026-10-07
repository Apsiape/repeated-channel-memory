# Verification scripts

Finite checks that accompany the paper. They support but do not replace the proofs. Exact checks use rational
or Gaussian-rational arithmetic; the others are numerical searches for counterexamples, where finding none is
evidence, not proof.

Run everything from the repository root (about six minutes):

```sh
python -u verify/run_all.py      # exit code 0 if and only if every script passes
```

Requirements: Python 3 with numpy, scipy and mpmath; `certificates/rebuild_full_orbits.py` also needs
python-flint (`pip install python-flint`).

| Script | Paper statement | What it checks | Kind |
|---|---|---|---|
| `check_undecidable_arithmetic.py` | Theorem 2, Lemma A.2, Remark 4.4, Corollary 4.1 | The Q(i) Schur builder (18) equals the Gram matrix of its unitaries; the two linear relations; the gap 1/(256² a⁶ d); the flat-completion fill; character orthogonality; the decision thresholds in their small-error and fixed-error forms. `--perturb` breaks a constant and must fail. | exact |
| `check_transfer_lemma.py` | Lemma A.2 | Every intermediate inequality of the transfer proof, on random noncommutative instances and nearby finite tracial factorizations with enlarged baths. | numerical |
| `check_flat_completion.py` | Lemma 5.2 | The completed family is unitary, keeps the old moment matrix as a principal submatrix, and has a flat weighted frame. | numerical |
| `check_certificate.py` | Theorem 4, via inequality (19) | The certificate's polynomial identity and the positive definiteness of its 13 Gram blocks, in exact rational arithmetic; numerical controls on random contacts. | exact |
| `certificates/rebuild_full_orbits.py` | Theorem 4, via inequality (19) | A second, independent checker: explicit S₃ expansion, exact generation of all 418 block-unitarity relations, Sylvester determinants; rejects three corrupted certificates. | exact |
| `check_envelope_search.py` | Theorem 4, inequality (6) | Random and locally optimized search for contacts that violate the 3/5 envelope. Largest ratio found: about 0.187. | numerical |
| `check_six_level.py` | Theorem 6(b); (13), (14), (16) | The six-level ray (w = t, Choi spectrum and rank, the distance √15\|t\|/12); searches for violations of the moment inequality (14) and of the supporting bound (13) on general contacts. | exact values, numerical searches |
| `check_fuel_constants.py` | Theorem 6(a), Appendix E.2 | Rebuilds the fuel contact from its definition and checks its antisymmetric effect A, Δ* = 8737/4263894, d* = log₂3 − S(ρ) and c* = 19.8967… < 19.897 (60 digits). | exact |
| `check_strong_converse.py` | Theorem 7, Proposition 8.1, Corollary 8.2, Appendix G; the sparse-testing remark of Section 6 | The constants of the proofs (275/2592, 6.6, 0.306, the tilt interval, the corollary's exponents, 27/10). On qubit tests, where the closure value is an exact eigenvalue: Lemmas G.1–G.3 and Proposition G.4 on random and near-dilation contacts, with a local search; the coin device of Proposition 8.1 simulated exactly. | exact constants, numerical searches |

The certificate `certificates/werner_commutator_c03_exact.json` has SHA-256
`f676207833025c93aac72b1afe8ee77b5105160c1c4935b78139102f7e719ca5`; `run_all.py` checks this value against
Appendix F. Its format is described in Appendix F.

Not checked here: the undecidability reduction itself (it rests on the cited theorems of Mastel and Slofstra and of
Navascués, Pironio and Acín), the stochastic path comparison of Appendices C and D, and the resource ledgers of
Appendix E.
