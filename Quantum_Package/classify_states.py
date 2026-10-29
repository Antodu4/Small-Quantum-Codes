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
Version vectorisée (numpy) : lecture des tableaux et parités sans boucle Python
sur les déterminants.
"""

import argparse
import gzip
import os
import sys

import numpy as np


def read_array(ezfio, name):
    """Lit un tableau EZFIO (.gz) ; renvoie un ndarray en ordre Fortran."""
    with gzip.open(os.path.join(ezfio, name + ".gz"), "rt") as f:
        text = f.read()
    nd = int(text.split(None, 1)[0])
    head = text.split(None, nd + 1)
    dims = [int(x) for x in head[1:1 + nd]]
    body = head[nd + 1] if len(head) > nd + 1 else ""
    if any(c in body for c in ".eEdD"):
        data = np.fromstring(body.replace("D", "E").replace("d", "e"), dtype=np.float64, sep=" ")
    else:
        data = np.fromstring(body, dtype=np.int64, sep=" ")
    return data.reshape(dims, order="F")


def read_scalar(ezfio, name):
    with open(os.path.join(ezfio, name)) as f:
        return f.read().split()[0]


def mo_parities(ezfio, tol):
    """Parité (+1/-1) de chaque OM sous x->-x, y->-y, z->-z ; 0 si l'OM est mélangée."""
    power = read_array(ezfio, "ao_basis/ao_power")      # (ao_num, 3)
    coef = read_array(ezfio, "mo_basis/mo_coef")        # (ao_num, mo_num)
    even = (power % 2 == 0)                             # (ao_num, 3)
    c2 = coef ** 2
    c2 = c2 / c2.sum(axis=0)                            # poids normalisés par OM
    w_even = even.T.astype(float) @ c2                  # (3, mo_num)
    par = np.zeros((coef.shape[1], 3), dtype=int)
    par[(w_even > 1 - tol).T] = 1
    par[(w_even < tol).T] = -1
    return par


def popcount64(x):
    """Nombre de bits à 1 de chaque entier d'un tableau uint64."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def det_parities(dets, par):
    """Parité (+1/-1/0) de chaque déterminant, produit des parités des OM occupées.

    dets : (n_int, 2, n_det) entiers 64 bits ; par : (mo_num, 3).
    Pour chaque direction, on compte les OM impaires occupées (popcount sur un masque)
    et on repère les OM sans parité définie (parité 0).
    """
    n_int, _, n_det = dets.shape
    words = dets.view(np.uint64)
    det_par = np.ones((n_det, 3), dtype=int)
    for k in range(3):
        mask_odd = np.zeros(n_int, dtype=np.uint64)
        mask_mix = np.zeros(n_int, dtype=np.uint64)
        for m in np.where(par[:, k] == -1)[0]:
            mask_odd[m // 64] |= np.uint64(1) << np.uint64(m % 64)
        for m in np.where(par[:, k] == 0)[0]:
            mask_mix[m // 64] |= np.uint64(1) << np.uint64(m % 64)
        n_odd = np.zeros(n_det, dtype=np.int64)
        mixed = np.zeros(n_det, dtype=bool)
        for i in range(n_int):
            for s in range(2):
                n_odd += popcount64(words[i, s, :] & mask_odd[i])
                mixed |= (words[i, s, :] & mask_mix[i]) != 0
        det_par[:, k] = np.where(n_odd % 2 == 0, 1, -1)
        det_par[mixed, k] = 0
    return det_par


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
    n_states = int(read_scalar(ez, "determinants/n_states"))
    par = mo_parities(ez, a.tol)
    mixed = np.where((par == 0).any(axis=1))[0]
    if len(mixed):
        print(f"ATTENTION : {len(mixed)} OM sans parité définie "
              f"({', '.join(str(m + 1) for m in mixed[:10])}"
              f"{'...' if len(mixed) > 10 else ''}) ; classement approximatif "
              f"(essayer --tol 1e-3).",
              file=sys.stderr)

    dets = read_array(ez, "determinants/psi_det")       # (n_int, 2, n_det)
    coef = read_array(ez, "determinants/psi_coef")      # (n_det, n_states)
    n_det = dets.shape[2]
    coef = coef[:n_det, :n_states]

    det_par = det_parities(dets, par)

    energies = [None] * n_states
    try:
        e = read_array(ez, "fci/energy")
        energies = list(e.ravel()[:n_states])
    except (FileNotFoundError, OSError):
        pass
    energies += [None] * (n_states - len(energies))

    labels = {(-1, 1, 1): "P_x", (1, -1, 1): "P_y", (1, 1, -1): "P_z",
              (1, 1, 1): "S/D (pair)"}
    target = {"x": (-1, 1, 1), "y": (1, -1, 1), "z": (1, 1, -1)}

    w = coef ** 2
    norms = w.sum(axis=0)                               # (n_states,)
    print(f"{'état':>4} {'E (Ha)':>16} {'<Px>':>7} {'<Py>':>7} {'<Pz>':>7}  sym")
    picked = []
    for s in range(n_states):
        es = f"{energies[s]:16.8f}" if energies[s] is not None else f"{'-':>16}"
        if norms[s] == 0.0:
            print(f"{s + 1:4d} {es} {'-':>7} {'-':>7} {'-':>7}  coefficients nuls")
            continue
        avg = (w[:, s] @ det_par) / norms[s]
        sign = tuple(int(np.sign(round(v, 3))) for v in avg)
        pure = bool(np.all(np.abs(np.abs(avg) - 1) < 1e-3))
        lab = labels.get(sign, "autre") + ("" if pure else " (mélangé)")
        print(f"{s + 1:4d} {es} {avg[0]:7.3f} {avg[1]:7.3f} {avg[2]:7.3f}  {lab}")
        if a.comp and pure and sign == target[a.comp]:
            picked.append(s + 1)

    if a.comp:
        print(f"\nÉtats P_{a.comp} : [{','.join(str(i) for i in picked)}]")


if __name__ == "__main__":
    main()
