"""Second checker of the exact certificate for inequality (19) of Appendix B.

Written independently of verify/check_certificate.py. Expands S_3 averaging
explicitly into ordinary real cyclic trace words, derives the target from the
full 3-by-3 block sums, verifies exactly that every relation is a generated
block-unitarity consequence, and uses Sylvester principal-minor determinants
(python-flint) rather than an LDL factorization. Three deliberately corrupted
certificates must be rejected. It checks the algebraic certificate only; the
probabilistic argument of Appendices C and D is separate.
Run: python verify/certificates/rebuild_full_orbits.py verify/certificates/werner_commutator_c03_exact.json
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from collections import defaultdict, Counter
from functools import lru_cache
from itertools import permutations, combinations
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
from flint import fmpq, fmpq_mat

X = 18
PERMS = tuple(permutations(range(3)))


def adjoint(word):
    return tuple(X if s == X else (s+9 if s<9 else s-9)
                 for s in reversed(word))


@lru_cache(None)
def cyclic_real(word):
    # No system-permutation identification and no discarded moments.
    if not word:
        return ()
    candidates = []
    for w in [word, adjoint(word)]:
        candidates.extend(w[j:]+w[:j] for j in range(len(w)))
    return min(candidates)


@lru_cache(None)
def explicit_average(word):
    counts = Counter()
    for perm in PERMS:
        renamed = []
        for s in word:
            if s == X:
                renamed.append(X)
            else:
                i,j = divmod(s%9,3)
                renamed.append(3*perm[i]+perm[j]+(9 if s>=9 else 0))
        counts[cyclic_real(tuple(renamed))] += 1
    return tuple((w,fmpq(n,6)) for w,n in counts.items())


def new_poly():
    return defaultdict(lambda: fmpq(0))


def add_average(poly, word, value):
    for key, weight in explicit_average(tuple(word)):
        poly[key] += value*weight


def key_of(poly):
    return tuple(sorted((w,str(c)) for w,c in poly.items() if c))


def sign_invariant(word):
    parity = [0,0,0]
    for s in word:
        if s!=X:
            i,j=divmod(s%9,3)
            parity[i]^=1
            parity[j]^=1
    return not any(parity)


def physical_relation_keys():
    # Prune candidate generators by parity only; never discard a term in
    # the certificate expansion. All submitted words are sign invariant.
    prefixes={(X,X)}
    for a in range(9):
        for b in range(9,18):
            for pair in [(a,b),(b,a)]:
                for where in combinations(range(4),2):
                    prefix=[X]*4
                    prefix[where[0]],prefix[where[1]]=pair
                    prefixes.add(tuple(prefix))
    keys=set()
    for prefix in prefixes:
        for j in range(3):
            for k in range(3):
                for row in [False,True]:
                    pairs = [(3*j+i,9+3*k+i) if row
                             else (9+3*i+j,3*i+k) for i in range(3)]
                    if not sign_invariant(prefix+pairs[0]):
                        continue
                    poly=new_poly()
                    for pair in pairs:
                        add_average(poly,prefix+pair,fmpq(1))
                    if j==k:
                        add_average(poly,prefix,fmpq(-1))
                    keys.add(key_of(poly))
    return keys


def rebuild(data, relation_keys, quiet=False):
    assert data["schema"] == "werner-one-root-trace-sos-v1"
    C = fmpq(str(data["coefficient"]))
    assert C > 0
    poly = new_poly()
    for number, block in enumerate(data["blocks"]):
        size=block["columns"]
        assert size<=128
        assert len(block["gram_upper"])==size*(size+1)//2
        gram=[[fmpq(0) for _ in range(size)] for _ in range(size)]
        values=iter(block["gram_upper"])
        for i in range(size):
            for j in range(i,size):
                gram[i][j]=gram[j][i]=fmpq(next(values))
        for k in range(1,size+1):
            determinant=fmpq_mat([row[:k] for row in gram[:k]]).det()
            assert determinant>0,("Sylvester failure",number,k)
        columns=[new_poly() for _ in range(size)]
        assert len(block["words"])==len(block["q_sparse_rows"])
        for word,row in zip(block["words"],block["q_sparse_rows"]):
            word=tuple(word)
            assert all(0<=s<=18 for s in word)
            assert len({j for j,_ in row})==len(row)
            for j,c in row:
                assert 0<=j<size
                columns[j][word]+=fmpq(c)
        for i in range(size):
            for j in range(i,size):
                coefficient=gram[i][j]*(1 if i==j else 2)
                if not coefficient:
                    continue
                for a,ca in columns[i].items():
                    for b,cb in columns[j].items():
                        add_average(poly,adjoint(a)+b,coefficient*ca*cb)
    assert len(data["relations"])==len(data["relation_multipliers"])
    for relation,multiplier in zip(data["relations"],data["relation_multipliers"]):
        zero=new_poly()
        for word,coefficient in relation:
            add_average(zero,word,fmpq(coefficient))
        assert key_of(zero) in relation_keys,"Not a generated block-unitarity consequence"
        for word,coefficient in zero.items():
            poly[word]-=fmpq(multiplier)*coefficient
    # Derived independently of the first checker's averaged-representative formula:
    # R=2Tr X^2-(2/3)sum_ij ReTr(Xu_ij^*Xu_ij),
    # D=(1/6)sum_(i!=j)ReTr[X^2(u_ij^*u_ij-u_ij^*u_ji)]-pTr X^2.
    expected=new_poly()
    expected[cyclic_real((X,X))]+=2*C+fmpq(25,27)
    for i in range(3):
        for j in range(3):
            u=3*i+j
            expected[cyclic_real((X,u+9,X,u))]-=2*C/3
            if i!=j:
                expected[cyclic_real((X,X,u+9,u))]-=fmpq(1,6)
                expected[cyclic_real((X,X,u+9,3*j+i))]+=fmpq(1,6)
    residual=[(w,str(poly[w]-expected[w]))
              for w in set(poly)|set(expected) if poly[w]!=expected[w]]
    assert not residual,("Full-orbit coefficient failure",residual[:3])
    if not quiet:
        print("PASS explicit S_3 expansion, real cyclic/adjoint trace identity",flush=True)
        print("PASS",len(data["blocks"]),"rational Sylvester certificates and",
              len(data["relations"]),"block-unitarity consequences",flush=True)
        print("Expanded real trace classes:",len(set(poly)|set(expected)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("certificate",type=Path)
    args=parser.parse_args()
    assert args.certificate.stat().st_size<=8*1024*1024
    start=time.monotonic()
    raw=args.certificate.read_bytes()
    data=json.loads(raw)
    valid=physical_relation_keys()
    rebuild(data,valid)
    print("Rational commutator coefficient",data["coefficient"],flush=True)
    for label,kind in [("negative Gram","negative"),("wrong positive Gram","positive"),
                       ("wrong relation weight","relation")]:
        corrupt=copy.deepcopy(data)
        if kind=="negative":
            corrupt["blocks"][0]["gram_upper"][0]="-1"
        elif kind=="positive":
            original=fmpq(corrupt["blocks"][0]["gram_upper"][0])
            corrupt["blocks"][0]["gram_upper"][0]=str(original+1)
        else:
            corrupt["relation_multipliers"][0]=str(fmpq(corrupt["relation_multipliers"][0])+1)
        try:
            rebuild(corrupt,valid,quiet=True)
        except AssertionError:
            print("PASS rejected",label,flush=True)
        else:
            raise AssertionError("Accepted corrupted certificate: "+label)
    print("Certificate SHA256",hashlib.sha256(raw).hexdigest(),flush=True)
    print("Runtime",time.monotonic()-start,"seconds",flush=True)


if __name__=="__main__":
    main()
