#!/usr/bin/env python3
"""
set_density_weights.py — Concentre les state_average_weight d'un EZFIO sur
                         une sélection d'états, pour que save_natorb (ou
                         save_natorb_no_ref) construise les OM naturelles
                         de la densité de CES états seulement.

Usage
-----
    python3 set_density_weights.py EZFIO SELECTION [--dry-run]

    EZFIO      répertoire EZFIO (modifié en place)
    SELECTION  états portant la densité, indexés à partir de 1 DANS la
               fonction d'onde courante (après extraction), ex. "3-5"
"""

import argparse
import sys


def parse_sel(spec):
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
        description="Concentre les poids state-average sur une sélection d'états.")
    parser.add_argument("ezfio_dir", metavar="EZFIO")
    parser.add_argument("selection", metavar="SELECTION",
                        help='états porteurs de densité, ex. "3-5" (indices courants)')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        from ezfio import ezfio
    except ImportError:
        sys.exit("Erreur : module `ezfio` introuvable. "
                 "Sourcez d'abord quantum_package.rc.")

    ezfio.set_file(args.ezfio_dir)
    n_states = ezfio.get_determinants_n_states()
    sel = parse_sel(args.selection)
    bad = [s for s in sel if not (1 <= s <= n_states)]
    if bad:
        sys.exit(f"Erreur : états hors de [1, {n_states}] : {bad}")

    w = [1.0 / len(sel) if (i + 1) in sel else 0.0 for i in range(n_states)]
    print(f"EZFIO    : {args.ezfio_dir}   n_states = {n_states}")
    print(f"densité  : états {sel}")
    print(f"poids    : {['%.4f' % x for x in w]}")

    if args.dry_run:
        print("(dry-run : aucune écriture)")
        return

    ezfio.set_determinants_state_average_weight(w)
    print(f"OK : poids écrits dans {args.ezfio_dir}")


if __name__ == "__main__":
    main()
