#!/usr/bin/env python3
"""
extend_n_states.py — Étend proprement psi_coef d'un EZFIO à un plus grand
                     nombre d'états (n_states -> N), pour pouvoir enchaîner
                     sur un FCI/CIPSI multi-états à partir d'un run converti
                     sur moins d'états (typiquement n_states=1).

Pourquoi c'est nécessaire
-------------------------
Dans QP2, `qp set determinants n_states N` ne redimensionne jamais
`psi_coef` sur le disque : le provider alloue un buffer (N_det, N) avec le
N qu'on vient de fixer et lit le fichier tel quel, sans padding. Si le
fichier ne contient encore que l'ancien nombre d'états, `qp run fci` plante
avec :

    EZFIO Error in : ezfio_read_array_do
    Dimensions of data .../psi_coef.gz different from array.

Ce script lit la forme RÉELLE de psi_coef directement dans le fichier
(indépendamment de la valeur, potentiellement obsolète, de n_states), garde
les états déjà présents tels quels, et complète les nouveaux états avec
l'amorce diagonale par défaut de QP2 (état k -> déterminant k = 1.0).

Usage
-----
    python3 extend_n_states.py EZFIO N_STATES [--dry-run]

Arguments
---------
    EZFIO       répertoire EZFIO à modifier (MODIFIÉ EN PLACE :
                travaillez sur une copie)
    N_STATES    nombre d'états cible (doit être >= au nombre déjà stocké)

Effets
------
    - determinants/n_states              <- N_STATES
    - determinants/psi_coef              <- états existants + amorce diagonale
    - determinants/state_average_weight  <- [1/N_STATES] * N_STATES

Prérequis
---------
    Environnement QP sourcé (module python `ezfio` dans le PYTHONPATH) :
        source ~/qp2/quantum_package.rc
"""

import argparse
import gzip
import os
import sys


def _ezfio_path(ezfio_dir, *parts):
    path = os.path.join(ezfio_dir, *parts) + ".gz"
    if not os.path.exists(path):
        path = path[:-3]
    return path


def _prod(seq):
    p = 1
    for x in seq:
        p *= x
    return p


def read_psi_coef_shape(ezfio_dir):
    """Read the ACTUAL (n_det, n_states) shape of psi_coef straight from the
    array file, independently of the (possibly stale) n_states parameter."""
    path = _ezfio_path(ezfio_dir, "determinants", "psi_coef")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        rank = int(f.readline().split()[0])
        shape = [int(x) for x in f.readline().split()]
        assert rank == 2, f"expected rank 2 for psi_coef, got {rank}"
        n_det, n_states_stored = shape
        values = [float(f.readline().strip()) for _ in range(_prod(shape))]
    # Fortran column-major (n_det, n_states): state k occupies n_det
    # consecutive values.
    states = [values[k * n_det:(k + 1) * n_det] for k in range(n_states_stored)]
    return n_det, states


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ezfio_dir", metavar="EZFIO", help="répertoire EZFIO (modifié en place)")
    parser.add_argument("n_states", metavar="N_STATES", type=int, help="nombre d'états cible")
    parser.add_argument("--dry-run", action="store_true", help="n'écrit rien, affiche seulement le diagnostic")
    args = parser.parse_args()

    if not os.path.isdir(args.ezfio_dir):
        sys.exit(f"Erreur : répertoire EZFIO introuvable : {args.ezfio_dir}")

    n_det, states = read_psi_coef_shape(args.ezfio_dir)
    n_states_stored = len(states)

    print(f"EZFIO              : {args.ezfio_dir}")
    print(f"n_det               : {n_det}")
    print(f"états déjà présents : {n_states_stored}")
    print(f"cible               : {args.n_states}")

    if args.n_states <= n_states_stored:
        print(f"Rien à faire : {args.n_states} <= {n_states_stored} état(s) déjà présent(s).")
        return

    for k in range(n_states_stored, args.n_states):
        v = [0.0] * n_det
        if k < n_det:
            v[k] = 1.0
        states.append(v)
    print(f"Amorce diagonale ajoutée pour les états {n_states_stored + 1} à {args.n_states} "
          f"(état k -> déterminant k = 1.0, comme l'initialisation par défaut de QP2).")

    if args.dry_run:
        print("(dry-run : aucune écriture)")
        return

    try:
        from ezfio import ezfio
    except ImportError:
        sys.exit("Erreur : module `ezfio` introuvable. "
                 "Sourcez d'abord quantum_package.rc.")

    ezfio.set_file(args.ezfio_dir)
    ezfio.set_determinants_n_states(args.n_states)
    ezfio.set_determinants_psi_coef(states)
    ezfio.set_determinants_state_average_weight([1.0 / args.n_states] * args.n_states)
    print(f"OK : {args.ezfio_dir} contient maintenant {args.n_states} états, "
          f"state_average_weight uniforme.")


if __name__ == "__main__":
    main()
