#!/usr/bin/env python3
"""
Classe les états d'un EZFIO Quantum Package par symétrie de réflexion (x, y, z)
pour un atome centré à l'origine, afin de repérer les composantes P_x, P_y, P_z.

Usage:
    python3 classify_states.py <ezfio> [--comp x|y|z] [--tol 1e-6]

Pour chaque état : énergie (si disponible), poids de parité <P_x>, <P_y>, <P_z>
(+1 = pair, -1 = impair) et étiquette. Un état ^2P_x est impair en x, pair en y et z.
Avec --comp, imprime en dernière ligne la liste des états de cette composante au
format attendu par `qp edit -s`, ex. "[2,5,8]".

Ne lit que les fichiers de l'EZFIO (pas besoin du module ezfio).
"""

import argparse
import gzip
import os
import sys

import numpy as np


def read_array(ezfio, name):
    """Lit un tableau EZFIO (.gz) ; renvoie un ndarray en ordre Fortran."""
    with gzip.open(os.path.join(ezfio, name + ".gz"), "rt") as f:
        tok = f.read().split()
    nd = int(tok[0])
    dims = [int(x) for x in tok[1:1 + nd]]
    vals = tok[1 + nd:]
    try:
        data = np.array([int(x) for x in vals], dtype=np.int64)
    except ValueError:
        data = np.array([float(x.replace("D", "E").replace("d", "e")) for x in vals])
    return data.reshape(dims, order="F")


def read_scalar(ezfio, name):
    with open(os.path.join(ezfio, name)) as f:
        return f.read().split()[0]


def mo_parities(ezfio, tol):
    """Parité (+1/-1) de chaque OM sous x->-x, y->-y, z->-z ; 0 si l'OM est mélangée."""
    power = read_array(ezfio, "ao_basis/ao_power")      # (ao_num, 3)
    coef = read_array(ezfio, "mo_basis/mo_coef")        # (ao_num, mo_num)
    ao_par = np.where(power % 2 == 0, 1, -1)            # (ao_num, 3)
    mo_num = coef.shape[1]
    par = np.zeros((mo_num, 3), dtype=int)
    for m in range(mo_num):
        c2 = coef[:, m] ** 2
        c2 = c2 / c2.sum()
        for k in range(3):
            w_even = c2[ao_par[:, k] == 1].sum()
            if w_even > 1 - tol:
                par[m, k] = 1
            elif w_even < tol:
                par[m, k] = -1
    return par


def occ_list(det_int, n_int):
    occ = []
    for i in range(n_int):
        v = int(det_int[i]) & ((1 << 64) - 1)
        b = 0
        while v:
            if v & 1:
                occ.append(64 * i + b)
            v >>= 1
            b += 1
    return occ


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ezfio")
    p.add_argument("--comp", choices=["x", "y", "z"], default=None,
                   help="composante P à lister pour `qp edit -s`")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="tolérance sur le mélange de parité des OM")
    a = p.parse_args()

    ez = a.ezfio.rstrip("/")
    n_int = int(read_scalar(ez, "determinants/n_int"))
    n_states = int(read_scalar(ez, "determinants/n_states"))
    par = mo_parities(ez, a.tol)
    mixed = np.where((par == 0).any(axis=1))[0]
    if len(mixed):
        print(f"ATTENTION : {len(mixed)} OM sans parité définie "
              f"({', '.join(str(m + 1) for m in mixed[:10])}"
              f"{'...' if len(mixed) > 10 else ''}) ; classement approximatif.",
              file=sys.stderr)

    dets = read_array(ez, "determinants/psi_det")       # (n_int, 2, n_det)
    coef = read_array(ez, "determinants/psi_coef")      # (n_det, n_states)
    n_det = dets.shape[2]
    coef = coef[:n_det, :n_states]

    det_par = np.ones((n_det, 3), dtype=int)
    for d in range(n_det):
        occ = occ_list(dets[:, 0, d], n_int) + occ_list(dets[:, 1, d], n_int)
        det_par[d] = np.prod(par[occ], axis=0)

    energies = [None] * n_states
    try:
        e = read_array(ez, "fci/energy")
        energies = list(e.ravel()[:n_states])
    except (FileNotFoundError, OSError):
        pass

    labels = {(-1, 1, 1): "P_x", (1, -1, 1): "P_y", (1, 1, -1): "P_z",
              (1, 1, 1): "S/D (pair)"}
    print(f"{'état':>4} {'E (Ha)':>16} {'<Px>':>7} {'<Py>':>7} {'<Pz>':>7}  sym")
    picked = []
    for s in range(n_states):
        w = coef[:, s] ** 2
        w = w / w.sum()
        avg = (w[:, None] * det_par).sum(axis=0)
        sign = tuple(int(np.sign(round(v, 3))) for v in avg)
        pure = np.all(np.abs(np.abs(avg) - 1) < 1e-3)
        lab = labels.get(sign, "autre") + ("" if pure else " (mélangé)")
        es = f"{energies[s]:16.8f}" if energies[s] is not None else f"{'-':>16}"
        print(f"{s + 1:4d} {es} {avg[0]:7.3f} {avg[1]:7.3f} {avg[2]:7.3f}  {lab}")
        if a.comp and pure and sign == {"x": (-1, 1, 1), "y": (1, -1, 1),
                                        "z": (1, 1, -1)}[a.comp]:
            picked.append(s + 1)

    if a.comp:
        print(f"\nÉtats P_{a.comp} : [{','.join(str(i) for i in picked)}]")


if __name__ == "__main__":
    main()
