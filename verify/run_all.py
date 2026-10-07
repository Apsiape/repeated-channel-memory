"""Run every verification script for "No algorithm can tell how much memory a repeated quantum channel needs".

Each script checks the statements listed next to it and exits with code 0 only if every check passes. The checks are
finite: exact arithmetic where stated, numerical searches elsewhere. They support but do not replace the proofs.
The certificate's SHA-256 is also compared with the value printed in Appendix F.
Run: python -u verify/run_all.py   (about six minutes; exit code 0 if and only if every script passes)
"""
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(HERE, "certificates", "werner_commutator_c03_exact.json")
SHA256 = "f676207833025c93aac72b1afe8ee77b5105160c1c4935b78139102f7e719ca5"

# (script, arguments, paper statements, what is checked)
RUNS = [
    ("check_undecidable_arithmetic.py", [], "Thm 2, Lem A.2, Rem 4.4, Cor 4.1",
     "Q(i) builder (18), linear relations, gap arithmetic, thresholds (exact)"),
    ("check_transfer_lemma.py", [], "Lem A.2", "every inequality of the transfer proof on random instances"),
    ("check_flat_completion.py", [], "Lem 5.2", "unitarity, principal submatrix, flat frame of the completion"),
    ("check_certificate.py", [CERT], "Thm 4 via (19)", "exact certificate: identity and positive definiteness"),
    (os.path.join("certificates", "rebuild_full_orbits.py"), [CERT], "Thm 4 via (19)",
     "exact certificate, second checker: relations generated, corruptions rejected"),
    ("check_envelope_search.py", [], "Thm 4 (6)", "numerical search for violations of the 3/5 envelope"),
    ("check_six_level.py", [], "Thm 6(b), (13), (14), (16)",
     "six-level ray, moment inequality, supporting bound on general contacts"),
    ("check_fuel_constants.py", [], "Thm 6(a), App E.2", "fuel contact: A, Delta_*, d_*, c_* < 19.897 (exact)"),
    ("check_strong_converse.py", [], "Thm 7, Prop 8.1, Cor 8.2, App G",
     "constants (exact); Lemmas G.1-G.3 and Prop G.4 on qubit tests; coin device"),
]


def main():
    failures = 0
    with open(CERT, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    ok = digest == SHA256
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  certificate SHA-256 matches Appendix F", flush=True)
    for script, args, where, what in RUNS:
        start = time.monotonic()
        run = subprocess.run([sys.executable, "-u", os.path.join(HERE, script), *args],
                             capture_output=True, text=True, errors="replace")
        secs = time.monotonic() - start
        ok = run.returncode == 0
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {script:40s} {where:28s} {secs:6.0f} s  {what}", flush=True)
        if not ok:
            print("\n".join((run.stdout + run.stderr).splitlines()[-20:]), flush=True)
    print(f"{len(RUNS) + 1 - failures}/{len(RUNS) + 1} passed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
