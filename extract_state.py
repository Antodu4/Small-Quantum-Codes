#!/usr/bin/env python3
"""
extract_state.py — Extrait un état d'un EZFIO multi-états et le sauve comme
                   unique état (n_states = 1), pour un calcul CIPSI ciblé
                   avec state_following.

Usage
-----
    python3 extract_state.py EZFIO ISTATE [--no-normalize] [--dry-run]

Arguments
---------
    EZFIO       chemin du répertoire EZFIO à modifier (MODIFIÉ EN PLACE :
                travaillez sur une copie, ex. `cp -r be.no.14s11p be.tgt.14s11p`)
    ISTATE      numéro de l'état à extraire, indexé à partir de 1
                (même numérotation que "Energy of state N" dans la sortie QP)

Options
-------
    --no-normalize   ne pas renormaliser le vecteur extrait (par défaut, il
                     est renormalisé à 1 dans l'espace des déterminants)
    --dry-run        affiche ce qui serait fait sans rien écrire

Effets
------
    - determinants/n_states              <- 1
    - determinants/psi_coef              <- colonne ISTATE uniquement
    - determinants/state_average_weight  <- [1.0]
    - determinants/read_wf               <- True
    psi_det et n_det ne sont PAS modifiés : tous les déterminants sont
    conservés, seuls les coefficients changent.

Prérequis
---------
    Environnement QP sourcé (module python `ezfio` dans le PYTHONPATH) :
        source ~/qp2_cs/quantum_package.rc

Exemple
-------
    cp -r be.no.14s11p be.tgt.14s11p
    python3 extract_state.py be.tgt.14s11p 30
"""

import argparse
import math
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extrait un état d'un EZFIO multi-états (n_states -> 1).",
        epilog="L'EZFIO est modifié en place : travaillez sur une copie.",
    )
    parser.add_argument("ezfio_dir", metavar="EZFIO",
                        help="répertoire EZFIO (modifié en place)")
    parser.add_argument("istate", metavar="ISTATE", type=int,
                        help="état à extraire, indexé à partir de 1")
    parser.add_argument("--no-normalize", action="store_true",
                        help="ne pas renormaliser le vecteur extrait")
    parser.add_argument("--dry-run", action="store_true",
                        help="n'écrit rien, affiche seulement le diagnostic")
    args = parser.parse_args()

    try:
        from ezfio import ezfio
    except ImportError:
        sys.exit("Erreur : module `ezfio` introuvable. "
                 "Sourcez d'abord quantum_package.rc.")

    if not os.path.isdir(args.ezfio_dir):
        sys.exit(f"Erreur : répertoire EZFIO introuvable : {args.ezfio_dir}")

    ezfio.set_file(args.ezfio_dir)
    n_det = ezfio.get_determinants_n_det()
    n_states = ezfio.get_determinants_n_states()

    print(f"EZFIO    : {args.ezfio_dir}")
    print(f"n_det    : {n_det}")
    print(f"n_states : {n_states}")

    if not (1 <= args.istate <= n_states):
        sys.exit(f"Erreur : ISTATE={args.istate} hors de [1, {n_states}].")
    if n_states == 1:
        sys.exit("Erreur : l'EZFIO ne contient déjà qu'un seul état.")

    psi_coef = ezfio.get_determinants_psi_coef()

    # Garde-fou sur le layout : on attend [n_states][n_det]
    if len(psi_coef) != n_states or len(psi_coef[0]) != n_det:
        sys.exit(f"Erreur : layout psi_coef inattendu "
                 f"({len(psi_coef)} x {len(psi_coef[0])}), "
                 f"attendu ({n_states} x {n_det}).")

    c = list(psi_coef[args.istate - 1])
    norm = math.sqrt(sum(x * x for x in c))
    print(f"état {args.istate} : norme = {norm:.12f}")
    if norm < 1e-12:
        sys.exit("Erreur : vecteur de norme nulle, extraction impossible.")

    if not args.no_normalize:
        c = [x / norm for x in c]

    # Déterminant dominant, utile pour vérifier qu'on tient le bon état
    imax = max(range(n_det), key=lambda i: abs(c[i]))
    print(f"coefficient dominant : c[{imax + 1}] = {c[imax]:+.9f}")

    if args.dry_run:
        print("(dry-run : aucune écriture)")
        return

    ezfio.set_determinants_n_states(1)
    ezfio.set_determinants_psi_coef([c])
    ezfio.set_determinants_state_average_weight([1.0])
    ezfio.set_determinants_read_wf(True)
    print(f"OK : {args.ezfio_dir} contient maintenant 1 état "
          f"(ancien état {args.istate}), read_wf=True.")


if __name__ == "__main__":
    main()
