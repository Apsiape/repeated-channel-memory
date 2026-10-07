# No algorithm can tell how much memory a repeated quantum channel needs

Seth Douglas and Nidhal Mghirbi, October 2026.

This repository holds the paper, its LaTeX source and the finite checks that accompany it. It is version 1.1.0
(doi:10.5281/zenodo.23204398) of the Zenodo record doi:10.5281/zenodo.23199340; that concept DOI always resolves to the
latest version. Version 1.0.0 is doi:10.5281/zenodo.23199341.

A device that applies a quantum channel over and over must release each output before the next input arrives.
How much memory does it need? Suppose every fresh qubit the device receives is maximally mixed. Companion work
showed that the answer then turns on one set: the channels that a unitary with a finite, maximally mixed bath
implements, exactly or in the limit. Inside that set, a channel with a flat probe can be served with memory that
grows more slowly than the number of uses. Outside it, a device with sublinear memory fails with probability
tending to one.

- **No algorithm can locate the line.** A computable map sends programs to explicit Schur channels with
  Gaussian-rational entries. If the program halts, logarithmic memory suffices; if it runs forever, every device
  fed maximally mixed qubits needs linear memory, at every fixed error, for example 1/2. The growth rate of the
  least memory cannot be computed, and neither can the growth rate of the smallest bath that sustains closed
  repeated use.
- **Far from the line, failure is almost certain.** Outside the line, a device whose supplied purity grows
  sublinearly fails with probability tending to one, at any memory and exchange; with maximally mixed imports and
  sublinear memory the success probability decays exponentially. For ordinary supplied purity the decay is only
  inverse-linear, and that is sharp: a device can hide linear purity in a rare pure branch. A bounded number of
  randomly placed checks already exposes the cost, whereas in the critical window below, checks on o(n) rounds see
  nothing.
- **Connes' embedding problem is a statement about memory.** It has a positive answer exactly when every
  factorizable channel with a flat probe can be repeated with sublinear memory from maximally mixed qubits;
  Schur channels already suffice.
- **Just past the line, the cost switches on at the statistical scale.** For the qutrit Werner–Holevo family at
  distance h/√n beyond its boundary point, independent uses of the boundary channel reach error 2Φ(h/(2σ)) − 1 in
  the limit, where Φ is the standard normal distribution function and σ = 5√2/27. No device supplied with purity
  o(√n) does better, whatever its memory. For exact service, purity grows linearly in the distance along this
  family but only quadratically along an explicit six-level direction; tensoring the two puts both directions at
  one boundary point.

## Build

```sh
cd paper
python build.py      # three pdflatex passes; fails on undefined references or overfull boxes
```

## Verification

```sh
python -u verify/run_all.py    # about six minutes; exit code 0 if and only if every check passes
```

The scripts need Python 3 with numpy, scipy and mpmath, and python-flint for the second certificate checker.
They check finite instances, exact certificates and constants; they support but do not replace the proofs.
`verify/README.md` lists which paper statements each script checks.

## Changes in version 1.1.0

- New Section 8 and Appendix G: the strong converse (Theorem 7), the rare-pure-branch device (Proposition 8.1) and
  the random-checks corollary (Corollary 8.2), with complete proofs.
- Theorem 2 now holds at every fixed error; Corollary 4.1, Remark 4.4 and Corollary 5.3 are updated accordingly, and
  the open question on fixed size now asks only about fixed dimension.
- Section 6: sparse testing cannot see the critical window.
- New check `verify/check_strong_converse.py`. Theorems 1–6, appendices A–F and all equation numbers keep their
  version 1.0.0 numbering.

## Companion papers

- S. Douglas and N. Mghirbi, The memory and purity cost of immediate delivery, doi:10.5281/zenodo.23198907.
- S. Douglas and N. Mghirbi, Causal quantum-channel simulation: memory beyond entropy, doi:10.5281/zenodo.23070870.
- N. Mghirbi and S. Douglas, Factorizable rational quantum channels at an explicit distance from finite tracial
  baths, doi:10.5281/zenodo.23196593.
- S. Douglas, Bath dimension and initial entropy for closed repeated use of a quantum channel, arXiv:2609.18267.

## Licensing

Paper and documentation: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). Scripts: additionally MIT (`LICENSE-MIT.txt`).
See `RIGHTS.md`.
