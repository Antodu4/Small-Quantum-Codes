#!/usr/bin/env python3
"""
Analyse d'un scan de complex scaling (QP2 / CIPSI-CS).

La sortie QP2 contient plusieurs scans de theta : chacun correspond à une étape
(itération CIPSI) de la convergence de la fonction d'onde.

 1. Extrait les énergies complexes E(état, theta) de chaque étape à partir des
    tableaux "CS energies" de la sortie QP2.
 2. Reconstruit les trajectoires E_n(theta) par continuité (les états sont triés par
    Re(E) à chaque theta, donc l'indice d'état n'est PAS une trajectoire dès qu'il
    y a des croisements). Une trajectoire est numérotée par son état à theta = 0.
 3. Calcule la vélocité  v_n(theta) = |dE_n/dtheta|  (option : |theta dE/dtheta|)
    par spline cubique et cherche ses minima : un minimum net de vitesse (point
    stationnaire de la trajectoire) = résonance (critère de Moiseyev).
    E_res = E(theta_min),  Gamma = -2 Im E_res.
 4. Compare les étapes de convergence, exporte les énergies, trace les graphiques.

Exemples d'utilisation courants :
  # Analyse complète par défaut (toutes les étapes)
  python cs_resonance.py He_cs.out

  # Tracer les trajectoires et la vélocité de l'étape 3 sur l'intervalle θ ∈ [0.150, 0.250] rad
  python cs_resonance.py He_cs.out --step 3 --plot --theta-range 0.150 0.250

  # Afficher uniquement le point à θ = 0.200 rad (ou la valeur la plus proche)
  python cs_resonance.py He_cs.out --step last --plot --theta-range 0.200

  # Masquer les points rouges des résonances et décaler/masquer la légende
  python cs_resonance.py He_cs.out --step last --plot --no-resonances --no-legend
"""
import argparse
import os
import re

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment


HARTREE_TO_EV = 27.211386245988


# --------------------------------------------------------------------------- #
# 1. Lecture
# --------------------------------------------------------------------------- #
def parse_cs_energies(path):
    """Retourne une liste d'étapes : dict(theta, E=(nθ, nstates) complexe, ndet)."""
    with open(path) as f:
        lines = f.read().splitlines()

    scans = []
    i = 0
    while i < len(lines):
        if "CS energies" in lines[i]:
            j = i + 1
            rows = []
            while j < len(lines):
                s = lines[j].split()
                if s and re.fullmatch(r"[-+]?\d\.\d+E[-+]\d+", s[0]):
                    rows.append([float(x) for x in s])
                elif rows:           # fin du tableau
                    break
                j += 1
            data = np.array(rows)
            theta = data[:, 0]
            E = data[:, 1::2] + 1j * data[:, 2::2]
            scans.append(dict(theta=theta, E=E))
            i = j
        i += 1
    if not scans:
        raise RuntimeError("Aucun tableau 'CS energies' trouvé dans le fichier.")
    ndets = [int(m.group(1)) for l in lines
             if (m := re.match(r"\*\s*Saved determinants\s+(\d+)", l))]
    for k, sc in enumerate(scans):
        sc["ndet"] = ndets[k] if len(ndets) == len(scans) else None
    return scans


# --------------------------------------------------------------------------- #
# 2. Suivi des trajectoires par continuité
# --------------------------------------------------------------------------- #
def track_trajectories(theta, E):
    """
    E[k, s] : énergies triées par Re(E) au pas k.
    Retourne T[k, n] : énergie de la trajectoire n (n = indice de l'état à theta=0).
    Prédiction linéaire (ordre 2) + affectation optimale (algorithme hongrois).
    """
    nth, ns = E.shape
    T = np.zeros_like(E)
    T[0] = E[0]
    if nth > 1:
        T[1] = E[1][_assign(T[0], E[1])]
    for k in range(2, nth):
        h1 = theta[k - 1] - theta[k - 2]
        h2 = theta[k] - theta[k - 1]
        pred = T[k - 1] + (T[k - 1] - T[k - 2]) * h2 / h1
        T[k] = E[k][_assign(pred, E[k])]
    return T


def _assign(pred, new):
    cost = np.abs(pred[:, None] - new[None, :])
    r, c = linear_sum_assignment(cost)
    idx = np.empty(len(pred), dtype=int)
    idx[r] = c
    return idx


# --------------------------------------------------------------------------- #
# 3. Vélocité et résonances
# --------------------------------------------------------------------------- #
def velocity(theta, Etraj, logderiv=False, nfine=20):
    """v(theta) = |dE/dtheta| (ou |theta dE/dtheta|) via spline cubique."""
    cs = CubicSpline(theta, Etraj)
    tf = np.linspace(theta[0], theta[-1], (len(theta) - 1) * nfine + 1)
    dE = cs(tf, 1)
    v = np.abs(dE)
    if logderiv:
        v = tf * v
    return tf, v, cs(tf)


def find_resonances(theta, T, threshold=None, logderiv=False, theta_min=0.0,
                    imag_tol=None, prominence=0.0, nfine=20):
    """
    Minima locaux de la vitesse. Tous les filtres sont optionnels.
    """
    from scipy.signal import find_peaks
    out = []
    for n in range(T.shape[1]):
        tf, v, Ef = velocity(theta, T[:, n], logderiv, nfine)
        pk, props = find_peaks(-v, prominence=prominence * np.median(v))
        for i in pk:
            Er = Ef[i]
            if tf[i] < theta_min:
                continue
            if threshold is not None and Er.real <= threshold:
                continue
            if imag_tol is not None and Er.imag > -imag_tol:
                continue
            out.append(dict(traj=n + 1, theta=tf[i], v=v[i], E=Er, Gamma=-2 * Er.imag,
                            depth=props["prominences"][list(pk).index(i)] / np.median(v),
                            mult=1))
    merged = []
    for r in sorted(out, key=lambda r: r["v"]):
        for m in merged:
            if abs(m["E"] - r["E"]) < 1e-5 and abs(m["theta"] - r["theta"]) < 1e-3:
                m["mult"] += 1
                m["traj"] = f"{m['traj']},{r['traj']}"
                break
        else:
            merged.append(r)
    return merged


def diagnostics(theta, T, logderiv, theta_min=0.0):
    """Vitesse minimale de chaque trajectoire (sans filtre) pour inspection."""
    rows = []
    for n in range(T.shape[1]):
        tf, v, Ef = velocity(theta, T[:, n], logderiv)
        m = tf >= theta_min
        i = np.argmin(np.where(m, v, np.inf))
        rows.append((n + 1, T[0, n].real, v[i], tf[i], Ef[i], np.median(v)))
    return rows


# --------------------------------------------------------------------------- #
# 4. Sorties
# --------------------------------------------------------------------------- #
def save_csv(outdir, s, theta, T, V):
    path = os.path.join(outdir, f"step{s}_trajectories.csv")
    with open(path, "w") as f:
        f.write("theta,traj,ReE,ImE,velocity\n")
        for k, th in enumerate(theta):
            for n in range(T.shape[1]):
                f.write(f"{th:.6f},{n+1},{T[k,n].real:.10f},{T[k,n].imag:.10f},{V[k,n]:.10e}\n")
    return path


def make_plots(outdir, s, theta, T, V, res, threshold, logderiv, show_resonances=True, show_legend=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylab = r"$|\theta\,dE/d\theta|$" if logderiv else r"$|dE/d\theta|$"

    # (a) plan complexe
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in range(T.shape[1]):
        ax.plot(T[:, n].real, T[:, n].imag, "-", lw=1, alpha=.7)
    if show_resonances:
        for r in res:
            ax.plot(r["E"].real, r["E"].imag, "ro", ms=7)
            ax.annotate(f"{r['traj']}", (r["E"].real, r["E"].imag), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls=":", lw=.8)
        ax.set_xlim(left=max(threshold - 0.3, T.real.min()))
    ax.set_xlabel("Re E (Ha)")
    ax.set_ylabel("Im E (Ha)")
    ax.set_title(f"Étape {s} : trajectoires E(θ)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step{s}_complex_plane.png"), dpi=150)
    plt.close(fig)

    # (b) vélocité
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in range(T.shape[1]):
        tf, v, Ef = velocity(theta, T[:, n], logderiv)
        if threshold is not None and Ef.real.max() <= threshold:
            continue
        ax.semilogy(tf, v, lw=1, label=f"traj {n+1}")
    if show_resonances:
        for r in res:
            ax.plot(r["theta"], r["v"], "ro", ms=6)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(ylab + " (Ha/rad)")
    ax.set_title(f"Étape {s} : vélocité")
    if show_legend:
        ax.legend(title="trajectoire", fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step{s}_velocity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_states(spec, n):
    """'1,3,15-17' -> [1,3,15,16,17] ; None -> tous."""
    if not spec:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [k for k in out if 1 <= k <= n]


def make_side_by_side(outdir, s, theta, T, res, threshold, logderiv, states,
                      show=False, ylog=True, xlim=None, ylim=None,
                      theta_range=None, show_resonances=True, show_legend=True):
    """Figure à 2 panneaux : trajectoires dans le plan complexe | vélocité vs theta."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sélection de la fenêtre ou d'un seul point de theta
    single_point = False
    if theta_range is not None:
        if len(theta_range) == 1:
            th_target = theta_range[0]
            idx_closest = np.argmin(np.abs(theta - th_target))
            theta_sub = theta[idx_closest:idx_closest + 1]
            T_sub = T[idx_closest:idx_closest + 1, :]
            single_point = True
            th_min, th_max = theta_sub[0], theta_sub[0]
        else:
            th_min, th_max = theta_range
            mask = (theta >= th_min) & (theta <= th_max)
            if not np.any(mask):
                raise ValueError(f"Aucune valeur de theta trouvée dans l'intervalle [{th_min}, {th_max}]")
            theta_sub = theta[mask]
            T_sub = T[mask, :]
    else:
        theta_sub = theta
        T_sub = T

    groups = []
    for n in states:
        for g in groups:
            if np.max(np.abs(T_sub[:, g[0] - 1] - T_sub[:, n - 1])) < 1e-4:
                g.append(n)
                break
        else:
            groups.append([n])

    cmap = plt.get_cmap("tab20")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for k, g in enumerate(groups):
        n = g[0]
        c = cmap(k % 20)
        lab = ",".join(map(str, g)) if len(g) <= 3 else f"{g[0]}-{g[-1]}"
        E = T_sub[:, n - 1]
        
        if single_point:
            # Affichage d'un seul point
            ax1.plot(E.real[0], E.imag[0], "o", color=c, ms=6, label=lab)
            tf, v, _ = velocity(theta, T[:, n - 1], logderiv)
            v_pt = np.interp(theta_sub[0], tf, v)
            ax2.plot(theta_sub[0], v_pt, "o", color=c, ms=6, label=lab)
        else:
            ax1.plot(E.real, E.imag, "-", color=c, lw=1.4, label=lab)
            ax1.plot(E.real[0], E.imag[0], "o", color=c, ms=4)            # début de l'intervalle
            ax1.plot(E.real[-1], E.imag[-1], ">", color=c, ms=6)          # fin de l'intervalle
            tf, v, _ = velocity(theta_sub, E, logderiv)
            ax2.plot(tf, v, "-", color=c, lw=1.4, label=lab)

    if show_resonances:
        sel = set(states)
        for r in res:
            if not (set(int(t) for t in str(r["traj"]).split(",")) & sel):
                continue
            if theta_range is not None:
                if single_point:
                    if abs(r["theta"] - theta_sub[0]) > 1e-3:
                        continue
                elif not (th_min <= r["theta"] <= th_max):
                    continue
            ax1.plot(r["E"].real, r["E"].imag, "ro", ms=5, zorder=5)
            ax2.plot(r["theta"], r["v"], "ro", ms=5, zorder=5)

    if threshold is not None:
        ax1.axvline(threshold, color="k", ls=":", lw=.8)
    ax1.axhline(0, color="gray", lw=.5)
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("Re E (Ha)")
    ax1.set_ylabel("Im E (Ha)")
    if single_point:
        ax1.set_title(f"Étape {s} : états à θ = {theta_sub[0]:.4f} rad")
    else:
        ax1.set_title(f"Étape {s} : trajectoires E(θ)   (● θ min, ▶ θ max)")
    ax1.grid(alpha=.3)

    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_ylabel((r"$|\theta\,dE/d\theta|$" if logderiv else r"$|dE/d\theta|$") + " (Ha/rad)")
    if ylog:
        ax2.set_yscale("log")
    ax2.set_title("Vélocité")
    ax2.grid(alpha=.3, which="both")

    if show_legend:
        ax2.legend(title="trajectoire", fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    fig.tight_layout()
    path = os.path.join(outdir, f"step{s}_trajectories_velocity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def select_states(spec, erange, T):
    """Sélection par numéros ('1,3,15-17') et/ou fenêtre d'énergie Re E(theta=0)."""
    st = parse_states(spec, T.shape[1])
    if erange is not None:
        lo, hi = erange
        st = [n for n in st if lo <= T[0, n - 1].real <= hi]
    return st


def group_degenerate(T, states, tol=1e-4):
    """Regroupe les trajectoires confondues -> liste de listes."""
    groups = []
    for n in states:
        for g in groups:
            if np.max(np.abs(T[:, g[0] - 1] - T[:, n - 1])) < tol:
                g.append(n)
                break
        else:
            groups.append([n])
    return groups


def export_energies(outdir, s, ndet, theta, T, E_sorted, states, order):
    """Fichier texte : une ligne par theta, colonnes Re/Im de chaque état choisi (Ha)."""
    data = T if order == "tracked" else E_sorted
    path = os.path.join(outdir, f"step{s}_energies.dat")
    cols = []
    for n in states:
        cols += [f"ReE_{n}", f"ImE_{n}"]
    with open(path, "w") as f:
        f.write(f"# Etape {s} ; {ndet} determinants ; energies en Hartree\n")
        f.write(f"# colonnes : "
                f"{'trajectoires suivies par continuite' if order == 'tracked' else 'etats tries par Re(E) (ordre QP2)'}\n")
        f.write("# " + " ".join(f"{c:>16s}" for c in ["theta"] + cols)[2:] + "\n")
        for k, th in enumerate(theta):
            row = [f"{th:16.8f}"]
            for n in states:
                z = data[k, n - 1]
                row += [f"{z.real:16.10f}", f"{z.imag:16.10f}"]
            f.write(" ".join(row) + "\n")
    return path


def make_rotation_plot(outdir, tag, runs, threshold, states, show=False,
                       theta_marks=0.1, xlim=None, ylim=None, labels=True, show_legend=True):
    """Rotation des états dans le plan complexe."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Tref = runs[-1][2]
    groups = group_degenerate(Tref, states)
    cmap = plt.get_cmap("tab20")
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(9, 7))

    for r, (name, theta, T) in enumerate(runs):
        ls = styles[r % len(styles)]
        for k, g in enumerate(groups):
            n = g[0]
            c = cmap(k % 20)
            E = T[:, n - 1]
            lab = (",".join(map(str, g)) if len(g) <= 3 else f"{g[0]}-{g[-1]}")
            ax.plot(E.real, E.imag, ls, color=c, lw=1.4,
                    label=lab if r == len(runs) - 1 else None)
            if r == len(runs) - 1:
                idx = [i for i, th in enumerate(theta)
                       if abs(th / theta_marks - round(th / theta_marks)) < 1e-6]
                ax.plot(E.real[idx], E.imag[idx], "o", color=c, ms=3.5)
                ax.plot(E.real[0], E.imag[0], "s", color=c, ms=6)
                ax.plot(E.real[-1], E.imag[-1], ">", color=c, ms=7)
                if labels:
                    ax.annotate(lab, (E.real[-1], E.imag[-1]), fontsize=7,
                                xytext=(5, -3), textcoords="offset points", color=c)

    if threshold is not None:
        ax.axvline(threshold, color="k", ls=":", lw=.8, label=f"seuil {threshold}")
    ax.axhline(0, color="gray", lw=.5)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Re E (Ha)")
    ax.set_ylabel("Im E (Ha)")
    names = ", ".join(r[0] for r in runs)
    ax.set_title(f"Rotation des états dans le plan complexe ({names})\n"
                 f"■ θ=0   ● tous les {theta_marks} rad   ▶ θ max")
    ax.grid(alpha=.3)

    if show_legend:
        if len(runs) > 1:
            from matplotlib.lines import Line2D
            h, l = ax.get_legend_handles_labels()
            for r, (name, _, _) in enumerate(runs):
                h.append(Line2D([0], [0], color="k", ls=styles[r % len(styles)]))
                l.append(name)
            ax.legend(h, l, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", title="états / étapes", borderaxespad=0.)
        else:
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", title="état", borderaxespad=0.)

    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_rotation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main():
    class Fmt(argparse.RawDescriptionHelpFormatter):
        def _get_help_string(self, action):
            h = action.help or ""
            if action.default not in (None, False, 0.0, argparse.SUPPRESS) \
                    and action.option_strings:
                h += f" [défaut : {action.default}]"
            return h

    epilog = """exemples :
  # Analyse complète de toutes les étapes (résonances + rapport de convergence)
  python cs_resonance.py He_cs.out

  # Sélectionner une fenêtre de theta [0.150, 0.250] rad
  python cs_resonance.py He_cs.out --step last --plot --states 7-20 --theta-range 0.150 0.250

  # Sélectionner UN SEUL point de theta (ex. 0.200 rad)
  python cs_resonance.py He_cs.out --step last --plot --states 7-20 --theta-range 0.200

  # Masquer les points rouges des résonances et désactiver la légende
  python cs_resonance.py He_cs.out --step last --plot --states 7-20 --no-resonances --no-legend
"""
    p = argparse.ArgumentParser(prog="cs_resonance.py", description=__doc__,
                                epilog=epilog, formatter_class=Fmt)
    p.add_argument("file", metavar="FICHIER",
                   help="sortie QP2 contenant les scans de complex scaling")
    p.add_argument("--outdir", default="cs_results", metavar="DOSSIER",
                   help="dossier de sortie (créé si besoin)")

    g = p.add_argument_group("sélection des étapes et des états")
    g.add_argument("--step", default="all", metavar="N|last|all",
                   help="étape(s) de convergence à traiter : 'all', 'last' ou un numéro")
    g.add_argument("--states", default=None, metavar="LISTE",
                   help="trajectoires à exporter/tracer, ex. '1,3,15-17'")
    g.add_argument("--erange", type=float, nargs=2, metavar=("EMIN", "EMAX"),
                   help="ne garde que les états dont Re E(theta=0) est dans [EMIN, EMAX]")

    g = p.add_argument_group("recherche de résonance (vélocité)")
    g.add_argument("--threshold", type=float, default=None, metavar="E",
                   help="seuil d'ionisation (Ha)")
    g.add_argument("--logderiv", action="store_true",
                   help="vélocité = |theta dE/dtheta| au lieu de |dE/dtheta|")
    g.add_argument("--theta-min", type=float, default=0.0, metavar="TH",
                   help="ignore les minima à theta < TH (rad)")
    g.add_argument("--prominence", type=float, default=0.0, metavar="P",
                   help="profondeur minimale d'un minimum de vitesse")
    g.add_argument("--imag-tol", type=float, default=None, metavar="G",
                   help="ne garde que les minima avec Im E < -G")
    g.add_argument("--diag", action="store_true",
                   help="affiche la vitesse minimale de chaque trajectoire")

    g = p.add_argument_group("export de données")
    g.add_argument("--export-energies", action="store_true",
                   help="écrit stepN_energies.dat")
    g.add_argument("--order", choices=["tracked", "sorted"], default="tracked",
                   help="colonnes de l'export")

    g = p.add_argument_group("graphiques")
    g.add_argument("--rotation", action="store_true",
                   help="rotation des états dans le plan complexe")
    g.add_argument("--compare", action="store_true",
                   help="superpose les étapes choisies sur un seul graphique")
    g.add_argument("--plot", action="store_true",
                   help="figure à 2 panneaux : plan complexe | vélocité(theta)")
    g.add_argument("--theta-marks", type=float, default=0.1, metavar="PAS",
                   help="pas (rad) des repères en theta sur la rotation")
    g.add_argument("--theta-range", type=float, nargs="+", metavar="THETA",
                   help="un seul theta ou intervalle à afficher, ex. --theta-range 0.200 ou 0.150 0.250")
    g.add_argument("--no-resonances", action="store_true",
                   help="masque les points rouges indiquant les résonances")
    g.add_argument("--no-legend", action="store_true",
                   help="desactive complètement l'affichage de la légende")
    g.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   help="bornes de l'axe Re E")
    g.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   help="bornes de l'axe Im E")
    g.add_argument("--no-labels", action="store_true",
                   help="pas d'étiquettes d'état")
    g.add_argument("--linear-v", action="store_true",
                   help="axe de vélocité linéaire")
    g.add_argument("--show", action="store_true",
                   help="ouvre la fenêtre matplotlib")
    g.add_argument("--no-plot", action="store_true",
                   help="désactive les figures par défaut")

    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    scans = parse_cs_energies(a.file)
    print(f"{len(scans)} étape(s) de convergence trouvée(s) ; "
          f"{len(scans[0]['theta'])} valeurs de theta, {scans[0]['E'].shape[1]} états")

    allT = [track_trajectories(sc["theta"], sc["E"]) for sc in scans]
    if len(scans) > 1:
        print("\n=== Convergence entre étapes ===")
        print("  étape  n_det   max|ΔE| vs étape précédente (Ha, sur tous θ et trajectoires)")
        for k, sc in enumerate(scans):
            nd = sc["ndet"] if sc["ndet"] is not None else "?"
            if k == 0:
                print(f"  {k+1:3d}   {nd:>6}   -")
            else:
                dE = np.abs(allT[k] - allT[k - 1])
                print(f"  {k+1:3d}   {nd:>6}   {dE.max():.2e}  "
                      f"(trajectoire {dE.max(axis=0).argmax()+1}, "
                      f"theta={sc['theta'][dE.max(axis=1).argmax()]:.3f})")

    if a.step == "all":
        todo = list(range(1, len(scans) + 1))
    elif a.step == "last":
        todo = [len(scans)]
    else:
        todo = [int(a.step)]

    for s in todo:
        sc = scans[s - 1]
        theta, E = sc["theta"], sc["E"]
        T = allT[s - 1]
        V = np.array([np.abs(np.gradient(T[:, n], theta)) for n in range(T.shape[1])]).T
        if a.logderiv:
            V = V * theta[:, None]
        res = find_resonances(theta, T, a.threshold, a.logderiv, a.theta_min,
                              imag_tol=a.imag_tol, prominence=a.prominence)
        res.sort(key=lambda r: r["v"])

        csv = save_csv(a.outdir, s, theta, T, V)
        print(f"\n=== Étape {s} ({sc['ndet']} déterminants) ===  (CSV : {csv})")
        if not res:
            print("  aucun minimum de vitesse trouvé (avec les filtres actifs)")
        else:
            print("  traj      theta     Re E (Ha)    Im E (Ha)    Re E (eV)   Gamma (eV)   v_min     mult")
            for r in res:
                print(f"  {str(r['traj']):8s}  {r['theta']:.4f}  {r['E'].real:11.6f}  "
                      f"{r['E'].imag:11.6f}  {r['E'].real*HARTREE_TO_EV:10.4f}  "
                      f"{r['Gamma']*HARTREE_TO_EV:10.4f}  {r['v']:.3e}  {r['mult']}")
        if a.diag:
            print("  -- diagnostic : vitesse minimale par trajectoire (theta >= theta-min) --")
            print("  traj  E(th=0)    v_min      theta   Re E       Im E      v_median")
            for n, e0, vm, tm, Em, vmed in diagnostics(theta, T, a.logderiv, a.theta_min):
                print(f"  {n:3d}  {e0:9.4f}  {vm:.3e}  {tm:.3f}  {Em.real:9.5f}  {Em.imag:9.5f}  {vmed:.3e}")
        st = select_states(a.states, a.erange, T)
        if a.export_energies:
            fn = export_energies(a.outdir, s, sc["ndet"], theta, T, E, st, a.order)
            print(f"  énergies exportées : {fn}  ({len(st)} états)")
        if a.rotation and not a.compare:
            png = make_rotation_plot(a.outdir, f"step{s}", [(f"étape {s}", theta, T)],
                                     a.threshold, st, show=a.show,
                                     theta_marks=a.theta_marks, xlim=a.xlim,
                                     ylim=a.ylim, labels=not a.no_labels,
                                     show_legend=not a.no_legend)
            print(f"  figure rotation : {png}")
        if a.plot:
            png = make_side_by_side(a.outdir, s, theta, T, res, a.threshold,
                                    a.logderiv, st, show=a.show, ylog=not a.linear_v,
                                    xlim=a.xlim, ylim=a.ylim,
                                    theta_range=a.theta_range,
                                    show_resonances=not a.no_resonances,
                                    show_legend=not a.no_legend)
            print(f"  figure : {png}")
        if not a.no_plot and not a.plot and not a.rotation:
            make_plots(a.outdir, s, theta, T, V, res, a.threshold, a.logderiv,
                       show_resonances=not a.no_resonances,
                       show_legend=not a.no_legend)

    if a.rotation and a.compare:
        runs = [(f"étape {k}", scans[k - 1]["theta"], allT[k - 1]) for k in todo]
        st = select_states(a.states, a.erange, runs[-1][2])
        tag = "steps_" + "-".join(str(k) for k in todo)
        png = make_rotation_plot(a.outdir, tag, runs, a.threshold, st, show=a.show,
                                 theta_marks=a.theta_marks, xlim=a.xlim,
                                 ylim=a.ylim, labels=not a.no_labels,
                                 show_legend=not a.no_legend)
        print(f"\nfigure comparaison des étapes : {png}")


if __name__ == "__main__":
    main()
