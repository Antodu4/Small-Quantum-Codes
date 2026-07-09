#!/usr/bin/env python3
"""
extract_states.py — Extrait un BLOC d'états d'un EZFIO multi-états et les
                    sauve comme seuls états (n_states = K), pour un calcul
                    CS multi-états avec state_following.

Généralisation de extract_state.py : au lieu d'un seul état, on garde une
fenêtre d'états (ex. la résonance + les états de pseudo-continuum voisins),
de sorte que le Davidson complexe suive un bloc de racines et que la
trajectoire de la résonance reste identifiable quand le continuum tourne.

Usage
-----
    python3 extract_states.py EZFIO STATES [--no-normalize] [--dry-run]

Arguments
---------
    EZFIO       chemin du répertoire EZFIO à modifier (MODIFIÉ EN PLACE :
                travaillez sur une copie, ex. `cp -r be.nat.14s11p be.blk.14s11p`)
    STATES      états à garder, indexés à partir de 1, ex. "42-50" ou "42,47-49"
                (même numérotation que "Energy of state N" dans la sortie QP)

Options
-------
    --no-normalize   ne pas renormaliser chaque vecteur extrait
    --dry-run        affiche ce qui serait fait sans rien écrire

Effets
------
    - determinants/n_states              <- K
    - determinants/psi_coef              <- colonnes STATES uniquement
    - determinants/state_average_weight  <- [1/K] * K
    - determinants/read_wf               <- True
    psi_det et n_det ne sont PAS modifiés : tous les déterminants sont
    conservés, seuls les coefficients changent.

Prérequis
---------
    Environnement QP sourcé (module python `ezfio` dans le PYTHONPATH) :
        source ~/qp2_cs/quantum_package.rc

Exemple
-------
    cp -r be.nat.14s11p be.blk.42-50.14s11p
    python3 extract_states.py be.blk.42-50.14s11p 42-50
"""

import argparse
import math
import os
import sys


def parse_states(spec):
    out = []
    for tok in spec.split(','):
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(
        description="Extrait un bloc d'états d'un EZFIO multi-états (n_states -> K).",
        epilog="L'EZFIO est modifié en place : travaillez sur une copie.",
    )
    parser.add_argument("ezfio_dir", metavar="EZFIO",
                        help="répertoire EZFIO (modifié en place)")
    parser.add_argument("states", metavar="STATES",
                        help='états à garder, ex. "42-50" ou "42,47-49"')
    parser.add_argument("--no-normalize", action="store_true",
                        help="ne pas renormaliser les vecteurs extraits")
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

    keep = parse_states(args.states)
    if len(keep) < 2:
        sys.exit("Erreur : moins de 2 états demandés ; "
                 "utilisez extract_state.py pour un état unique.")

    ezfio.set_file(args.ezfio_dir)
    n_det = ezfio.get_determinants_n_det()
    n_states = ezfio.get_determinants_n_states()

    print(f"EZFIO    : {args.ezfio_dir}")
    print(f"n_det    : {n_det}")
    print(f"n_states : {n_states}")
    print(f"états gardés ({len(keep)}) : {keep}")

    bad = [s for s in keep if not (1 <= s <= n_states)]
    if bad:
        sys.exit(f"Erreur : états hors de [1, {n_states}] : {bad}")

    psi_coef = ezfio.get_determinants_psi_coef()

    # Garde-fou sur le layout : on attend [n_states][n_det]
    if len(psi_coef) != n_states or len(psi_coef[0]) != n_det:
        sys.exit(f"Erreur : layout psi_coef inattendu "
                 f"({len(psi_coef)} x {len(psi_coef[0])}), "
                 f"attendu ({n_states} x {n_det}).")

    block = []
    for s in keep:
        c = list(psi_coef[s - 1])
        norm = math.sqrt(sum(x * x for x in c))
        if norm < 1e-12:
            sys.exit(f"Erreur : état {s} de norme nulle, extraction impossible.")
        if not args.no_normalize:
            c = [x / norm for x in c]
        imax = max(range(n_det), key=lambda i: abs(c[i]))
        print(f"état {s:3d} : norme = {norm:.12f}   "
              f"coef dominant c[{imax + 1}] = {c[imax]:+.9f}")
        block.append(c)

    if args.dry_run:
        print("(dry-run : aucune écriture)")
        return

    k = len(keep)
    ezfio.set_determinants_n_states(k)
    ezfio.set_determinants_psi_coef(block)
    ezfio.set_determinants_state_average_weight([1.0 / k] * k)
    ezfio.set_determinants_read_wf(True)
    print(f"OK : {args.ezfio_dir} contient maintenant {k} états "
          f"(anciens états {keep}), read_wf=True.")
    print(f"Le nouvel indice de chaque état est sa position dans la liste : "
          f"{ {old: new + 1 for new, old in enumerate(keep)} }")


if __name__ == "__main__":
    main()
