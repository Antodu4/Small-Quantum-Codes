#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qp_fci_analysis.py
==================
Analyse d'une sortie (.out) de calcul FCI / CIPSI réalisé avec Quantum Package.

Fonctionnalités
---------------
* Extraction, pour chaque taille de la fonction d'onde (N_det) et chaque état :
  E variationnelle, S^2, variance, PT2, rPT2, E+PT2, E+rPT2.
* Affichage des énergies convergées dans le shell (--energies).
* Affichage du tableau de convergence (--table).
* Énergies extrapolées FCI telles que calculées par QP (dernier bloc « Extrapolated energies »).
* Graphique de convergence d'un ou plusieurs états en fonction de N_det (--plot).

Exemples
--------
    python qp_fci_analysis.py Be_fci_20.out --energies
    python qp_fci_analysis.py Be_fci_20.out --table --states 1
    python qp_fci_analysis.py Be_fci_20.out --plot --states 1 2 3
    python qp_fci_analysis.py Be_fci_20.out --plot --states 1-5 --mode relative \
            --quantity evar ept2 -o conv_be.png
    python qp_fci_analysis.py Be_fci_20.out --plot --states 2-6 --mode excitation --show
    python qp_fci_analysis.py Be_fci_20.out --plot --states 2-6 --mode excitation --logy
    python qp_fci_analysis.py Be_fci_20.out --plot --states 1-3 --ndet-min 1000 --ndet-max 100000

Dépendances : numpy, matplotlib
"""

import argparse
import math
import os
import re
import sys

import numpy as np

HARTREE_TO_EV = 27.211386245988

# --------------------------------------------------------------------------- #
#  Lecture du fichier
# --------------------------------------------------------------------------- #
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"

RE_NDET_BLOCK = re.compile(r"^\s*N_det\s*=\s*(\d+)")
RE_STATE_BLOCK = re.compile(r"^\s*\*\s*State\s+(\d+)\s*$")
RE_FIELD = re.compile(
    r"^\s*(< S\^2 >|E\+rPT2|E\+PT2|rPT2|PT2|PT norm|Variance|E)\s*=\s*(" + _FLOAT + r")"
)
RE_ENERGY_STATE = re.compile(r"^\*\s*Energy of state\s+(\d+)\s+(" + _FLOAT + r")")
RE_S2_STATE = re.compile(r"^\*\s*S\^2 of state\s+(\d+)\s+(" + _FLOAT + r")")
RE_SAVED_DET = re.compile(r"^\*\s*Saved determinants\s+(\d+)")
RE_NUM_DET = re.compile(r"^\*\s*Number of determinants\s+(\d+)")

FIELD_KEYS = {
    "< S^2 >": "s2",
    "E": "evar",
    "Variance": "var",
    "PT norm": "ptnorm",
    "PT2": "pt2",
    "rPT2": "rpt2",
    "E+PT2": "ept2",
    "E+rPT2": "erpt2",
}


def _f(txt):
    """Convertit un flottant Fortran (D ou E) en float."""
    return float(txt.replace("D", "E").replace("d", "e"))


def parse_qp_output(path):
    """
    Lit la sortie QP et renvoie un dict : data[N_det][état] = {champ: valeur}.

    Deux sources sont fusionnées :
      1. les blocs « Summary at N_det » (E, PT2, rPT2, E+PT2, variance, S^2, ...) ;
      2. les blocs « Energy of state » issus de chaque diagonalisation, qui
         donnent l'énergie variationnelle de la dernière fonction d'onde
         (laquelle n'a pas encore de PT2 associée).
    """
    data = {}
    n_states_declared = None

    cur_ndet = None       # N_det du bloc Summary en cours
    cur_state = None      # état en cours de lecture dans un bloc Summary
    last_num_det = 1      # dernier « Number of determinants » vu
    pending = {}          # {état: (E, S2)} en attente de leur N_det

    def flush_pending(ndet):
        for st, vals in pending.items():
            d = data.setdefault(ndet, {}).setdefault(st, {})
            d.setdefault("evar", vals.get("evar"))
            if "s2" in vals:
                d.setdefault("s2", vals["s2"])
        pending.clear()

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            # -- blocs Summary --------------------------------------------------
            m = RE_NDET_BLOCK.match(line)
            if m:
                cur_ndet = int(m.group(1))
                cur_state = None
                continue

            if cur_ndet is not None:
                if line.strip().startswith("N_states"):
                    n_states_declared = int(line.split("=")[1])
                    continue
                m = RE_STATE_BLOCK.match(line)
                if m:
                    cur_state = int(m.group(1))
                    data.setdefault(cur_ndet, {}).setdefault(cur_state, {})
                    continue
                if cur_state is not None:
                    if not line.strip():
                        cur_state = None
                        continue
                    m = RE_FIELD.match(line)
                    if m:
                        key = FIELD_KEYS[m.group(1)]
                        data[cur_ndet][cur_state][key] = _f(m.group(2))
                    continue

            # -- énergies issues des diagonalisations ---------------------------
            m = RE_NUM_DET.match(line)
            if m:
                last_num_det = int(m.group(1))
                continue

            m = RE_ENERGY_STATE.match(line)
            if m:
                st = int(m.group(1))
                if st == 1 and pending:
                    # nouveau bloc sans « Saved determinants » intermédiaire
                    flush_pending(last_num_det)
                pending.setdefault(st, {})["evar"] = _f(m.group(2))
                continue

            m = RE_S2_STATE.match(line)
            if m:
                pending.setdefault(int(m.group(1)), {})["s2"] = _f(m.group(2))
                continue

            m = RE_SAVED_DET.match(line)
            if m:
                flush_pending(int(m.group(1)))
                continue

    if pending:
        flush_pending(last_num_det)

    data = {k: v for k, v in data.items() if v}
    return data, n_states_declared


# --------------------------------------------------------------------------- #
#  Outils d'analyse
# --------------------------------------------------------------------------- #
def all_states(data):
    return sorted({st for d in data.values() for st in d})


def series(data, state, key):
    """Renvoie (N_det, valeurs) pour un état et une grandeur, triés par N_det."""
    xs, ys = [], []
    for ndet in sorted(data):
        v = data[ndet].get(state, {}).get(key)
        if v is not None and math.isfinite(v):
            xs.append(ndet)
            ys.append(v)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


RE_EXTRAP_STATE = re.compile(r"^\s*State\s+(\d+)\s*$")
RE_EXTRAP_ROW = re.compile(
    r"^\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")(?:\s+(" + _FLOAT + r")\s+(" + _FLOAT + r"))?\s*$"
)


def parse_qp_extrapolation(path):
    """
    Lit le DERNIER bloc « Extrapolated energies » de la sortie QP.

    Renvoie {état: [(PT2_min, E_extrap, Excit_au, Excit_eV), ...]}, les lignes
    étant dans l'ordre du fichier (PT2_min croissant en valeur absolue).
    Excit_au / Excit_eV valent None pour l'état 1 (pas de colonnes d'excitation).
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == "Extrapolated energies"]
    if not starts:
        return {}

    out, state = {}, None
    for line in lines[starts[-1] + 1:]:
        txt = line.strip()
        if not txt or re.fullmatch(r"[=\-\s]+", txt) or txt.startswith("minimum PT2"):
            continue
        m = RE_EXTRAP_STATE.match(line)
        if m:
            state = int(m.group(1))
            out[state] = []
            continue
        m = RE_EXTRAP_ROW.match(line)
        if m and state is not None:
            vals = [(_f(g) if g is not None else None) for g in m.groups()]
            out[state].append(tuple(vals))
            continue
        break  # fin du bloc
    return out


def qp_extrap_value(extrap, state, row):
    """Ligne `row` (1 = première) du tableau d'extrapolation de QP pour un état."""
    rows = extrap.get(state)
    if not rows:
        return None
    return rows[min(row, len(rows)) - 1]


def last_value(data, state, key):
    """Dernière valeur disponible (N_det le plus grand) : (N_det, valeur)."""
    xs, ys = series(data, state, key)
    if len(xs) == 0:
        return None, None
    return int(xs[-1]), ys[-1]


def parse_states(tokens, available):
    """Interprète '1 2 5', '1-4', 'all'."""
    if not tokens or tokens == ["all"]:
        return available
    out = []
    for tok in tokens:
        if tok.lower() == "all":
            return available
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    out = sorted(set(out))
    missing = [s for s in out if s not in available]
    if missing:
        sys.exit(f"Erreur : état(s) {missing} absent(s) du fichier "
                 f"(disponibles : {available[0]}–{available[-1]}).")
    return out


# --------------------------------------------------------------------------- #
#  Affichage shell
# --------------------------------------------------------------------------- #
def print_energies(data, states, extrap, row):
    ndet_max = max(data)
    print()
    print("Énergies convergées")
    print("=" * 99)
    print(f"  N_det final (variationnel) : {ndet_max}")
    ndet_pt2 = max((n for n in data if any("pt2" in d for d in data[n].values())),
                   default=None)
    if ndet_pt2 is not None:
        print(f"  N_det du dernier calcul PT2 : {ndet_pt2}")
    if extrap:
        print(f"  E_extrap : extrapolation FCI de QP (dernier bloc « Extrapolated "
              f"energies »), ligne n°{row}")
    else:
        print("  Aucun bloc « Extrapolated energies » trouvé dans le fichier.")
    print("-" * 99)
    print(f"{'État':>5} {'<S^2>':>10} {'E_var (Ha)':>18} {'E+PT2 (Ha)':>18} "
          f"{'E_extrap (Ha)':>18} {'PT2 min':>8} {'ΔE (eV)':>10}")
    print("-" * 99)

    def fmt(v, f="{:18.10f}", w=18):
        return f.format(v) if v is not None else " " * (w - 1) + "-"

    for st in states:
        _, ev = last_value(data, st, "evar")
        _, s2 = last_value(data, st, "s2")
        _, ep = last_value(data, st, "ept2")
        r = qp_extrap_value(extrap, st, row)
        ex = r[1] if r else None
        pmin = r[0] if r else None
        if r is None:
            de = None
        elif r[3] is not None:
            de = r[3]          # excitation fournie par QP
        else:
            de = 0.0 if st == 1 else None
        print(f"{st:>5} {fmt(s2, '{:10.2e}', 10)} {fmt(ev)} {fmt(ep)} {fmt(ex, '{:18.8f}')} "
              f"{fmt(pmin, '{:8.4f}', 8)} {fmt(de, '{:10.4f}', 10)}")
    print("=" * 99)
    print("  ΔE : énergie d'excitation par rapport à l'état 1, d'après l'extrapolation de QP.")
    print()


def print_table(data, states, ndet_min=None, ndet_max=None):
    for st in states:
        print()
        print(f"État {st} : convergence en fonction de N_det")
        print("=" * 104)
        print(f"{'N_det':>10} {'E_var (Ha)':>16} {'PT2 (Ha)':>14} {'E+PT2 (Ha)':>16} "
              f"{'E+rPT2 (Ha)':>16} {'<S^2>':>10} {'Variance':>10}")
        print("-" * 104)
        for ndet in sorted(data):
            if (ndet_min is not None and ndet < ndet_min) or \
               (ndet_max is not None and ndet > ndet_max):
                continue
            d = data[ndet].get(st)
            if not d:
                continue

            def g(k, f):
                return f.format(d[k]) if k in d else "-".rjust(len(f.format(0.0)))

            print(f"{ndet:>10d} {g('evar', '{:16.10f}')} {g('pt2', '{:14.8f}')} "
                  f"{g('ept2', '{:16.10f}')} {g('erpt2', '{:16.10f}')} "
                  f"{g('s2', '{:10.2e}')} {g('var', '{:10.3e}')}")
        print("=" * 104)


# --------------------------------------------------------------------------- #
#  Graphique
# --------------------------------------------------------------------------- #
QUANTITY_LABEL = {"evar": "E$_{var}$", "ept2": "E+PT2", "erpt2": "E+rPT2"}
QUANTITY_STYLE = {"evar": ("o", "-"), "ept2": ("s", "--"), "erpt2": ("^", ":")}


def make_plot(data, states, quantities, mode, logx, logy, extrap, row, output, show, title,
              ndet_min=None, ndet_max=None):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10" if len(states) <= 10 else "tab20")

    for i, st in enumerate(states):
        color = cmap(i % cmap.N)
        ref = None
        if mode == "relative":
            r = qp_extrap_value(extrap, st, row)
            ref = r[1] if r else None
            if ref is None:
                vals = [series(data, st, q)[1] for q in ("evar", "ept2", "erpt2")]
                vals = np.concatenate([v for v in vals if len(v)])
                ref = vals.min() if len(vals) else None

        for q in quantities:
            x, y = series(data, st, q)
            if len(x) == 0:
                continue
            if mode == "excitation":
                x1, y1 = series(data, 1, q)
                common = np.intersect1d(x, x1)
                if len(common) == 0:
                    continue
                y = np.array([y[x == c][0] - y1[x1 == c][0] for c in common]) * HARTREE_TO_EV
                x = common
            elif mode == "relative":
                if ref is None:
                    continue
                y = (y - ref) * 1000.0  # mHa
            # fenêtre en N_det (appliquée après calcul des références/différences)
            keep = np.ones(len(x), dtype=bool)
            if logy:
                keep &= y > 0  # points non positifs impossibles en échelle log
            if ndet_min is not None:
                keep &= x >= ndet_min
            if ndet_max is not None:
                keep &= x <= ndet_max
            x, y = x[keep], y[keep]
            if len(x) == 0:
                continue
            marker, ls = QUANTITY_STYLE[q]
            label = f"État {st}" + (f" – {QUANTITY_LABEL[q]}" if len(quantities) > 1 else "")
            ax.plot(x, y, marker=marker, ls=ls, ms=5, lw=1.3, color=color, label=label)

    ax.set_xlabel("Nombre de déterminants")
    if logx:
        ax.set_xscale("log")
    if mode == "absolute":
        ax.set_ylabel("Énergie (Ha)")
    elif mode == "relative":
        ax.set_ylabel("E − E$_{extrap}$ (mHa)")
    else:
        ax.set_ylabel("Énergie d'excitation vs état 1 (eV)")
    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title)
    ncol = 1 if len(states) * len(quantities) <= 8 else 2
    ax.legend(fontsize=8, ncol=ncol, framealpha=0.9)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200)
        print(f"Graphique enregistré : {output}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Programme principal
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Analyse d'une sortie FCI/CIPSI de Quantum Package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exemples")[1] if "Exemples" in __doc__ else None,
    )
    p.add_argument("outfile", help="fichier de sortie de Quantum Package (.out)")
    p.add_argument("-s", "--states", nargs="+", default=None, metavar="ÉTAT",
                   help="états à traiter : liste (1 3 4), plage (1-5) ou 'all' "
                        "(défaut : all pour --energies/--table, 1 pour --plot)")
    p.add_argument("-e", "--energies", action="store_true",
                   help="affiche les énergies convergées dans le shell")
    p.add_argument("-t", "--table", action="store_true",
                   help="affiche le tableau de convergence vs N_det")
    p.add_argument("-p", "--plot", action="store_true",
                   help="trace la convergence en fonction de N_det")
    p.add_argument("-q", "--quantity", nargs="+", default=["evar"],
                   choices=["evar", "ept2", "erpt2"],
                   help="grandeur(s) tracée(s) : evar (variationnelle), ept2, erpt2")
    p.add_argument("-m", "--mode", default="absolute",
                   choices=["absolute", "relative", "excitation"],
                   help="absolute : E(N_det) ; relative : E − E_extrap (QP) en mHa "
                        "(échelle log) ; excitation : E_i − E_1 en eV")
    p.add_argument("--ndet-min", type=int, default=None,
                   help="N_det minimal affiché (graphique et tableau)")
    p.add_argument("--ndet-max", type=int, default=None,
                   help="N_det maximal affiché (graphique et tableau)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--logy", action="store_true",
                   help="axe des y logarithmique (défaut : activé en mode relative ; "
                        "impossible en mode absolute, les énergies étant négatives)")
    g.add_argument("--liny", action="store_true",
                   help="axe des y linéaire (utile pour le mode relative)")
    p.add_argument("--linx", action="store_true", help="axe N_det linéaire (défaut : log)")
    p.add_argument("--extrap-row", type=int, default=1, metavar="N",
                   help="ligne N (1 = première) du tableau d'extrapolation de QP à "
                        "utiliser (défaut : 1 = seuil PT2 le plus petit)")
    p.add_argument("-o", "--output", default=None,
                   help="fichier image de sortie (défaut : <fichier>_convergence.png)")
    p.add_argument("--show", action="store_true", help="affiche la fenêtre matplotlib")
    p.add_argument("--title", default=None, help="titre du graphique")
    args = p.parse_args()

    if not os.path.isfile(args.outfile):
        sys.exit(f"Erreur : fichier introuvable : {args.outfile}")

    data, n_states_decl = parse_qp_output(args.outfile)
    if not data:
        sys.exit("Erreur : aucune donnée de convergence trouvée dans ce fichier.")

    if args.ndet_min is not None and args.ndet_max is not None \
            and args.ndet_min > args.ndet_max:
        sys.exit("Erreur : --ndet-min doit être inférieur à --ndet-max.")

    available = all_states(data)
    print(f"Fichier : {args.outfile}")
    print(f"  {len(data)} points de convergence "
          f"(N_det de {min(data)} à {max(data)}), "
          f"{n_states_decl or len(available)} états.")

    # Si aucune action n'est demandée : affichage des énergies par défaut.
    if not (args.energies or args.table or args.plot):
        args.energies = True

    states_cli = args.states
    if states_cli is None:  # option non fournie : défaut selon l'action demandée
        plot_only = args.plot and not (args.energies or args.table)
        states_cli = ["1"] if plot_only else ["all"]
    states = parse_states(states_cli, available)
    if args.extrap_row < 1:
        sys.exit("Erreur : --extrap-row doit être >= 1.")
    extrap = parse_qp_extrapolation(args.outfile)

    if args.energies:
        print_energies(data, states, extrap, args.extrap_row)
    if args.table:
        print_table(data, states, args.ndet_min, args.ndet_max)
    if args.plot:
        plot_states = states
        if args.mode == "excitation":
            plot_states = [s for s in states if s != 1]
            if not plot_states:
                sys.exit("Erreur : le mode 'excitation' nécessite au moins un état > 1.")
        logy = args.logy or (args.mode == "relative" and not args.liny)
        if logy and args.mode == "absolute":
            sys.exit("Erreur : l'échelle log en y est impossible en mode 'absolute' "
                     "(énergies négatives). Utilise --mode relative ou excitation.")
        output = args.output
        if output is None and not args.show:
            output = os.path.splitext(args.outfile)[0] + "_convergence.png"
        title = args.title or f"Convergence CIPSI – {os.path.basename(args.outfile)}"
        make_plot(data, plot_states, args.quantity, args.mode, not args.linx, logy,
                  extrap, args.extrap_row, output, args.show, title,
                  args.ndet_min, args.ndet_max)


if __name__ == "__main__":
    main()
