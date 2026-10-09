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
"""
import argparse
import os
import re

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
from scipy.signal import argrelmin

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
            # on avance jusqu'aux lignes de données (commencent par un theta en notation E)
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
    # nombre de déterminants de chaque étape (ligne "Saved determinants")
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
    Minima locaux de la vitesse. Tous les filtres sont optionnels (None/0 = désactivé) :
    threshold  : ne garde que Re E > threshold
    theta_min  : ignore les minima à theta < theta_min
    imag_tol   : ne garde que Im E < -imag_tol (états avec une largeur)
    prominence : profondeur minimale du minimum, en fraction de la vitesse médiane
                 (sinon c'est juste du bruit sur une courbe plate).
    Les trajectoires dégénérées (même E à 1e-5 Ha) sont regroupées (multiplicité).
    """
    from scipy.signal import find_peaks
    out = []
    for n in range(T.shape[1]):
        tf, v, Ef = velocity(theta, T[:, n], logderiv, nfine)
        # on cherche les pics de -v, avec prominence relative
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
    # regroupement des dégénérescences
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


def make_plots(outdir, s, theta, T, V, res, threshold, logderiv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylab = r"$|\theta\,dE/d\theta|$" if logderiv else r"$|dE/d\theta|$"

    # (a) plan complexe
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in range(T.shape[1]):
        ax.plot(T[:, n].real, T[:, n].imag, "-", lw=1, alpha=.7)
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

    # (b) vélocité (états au-dessus du seuil)
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in range(T.shape[1]):
        tf, v, Ef = velocity(theta, T[:, n], logderiv)
        if threshold is not None and Ef.real.max() <= threshold:
            continue
        ax.semilogy(tf, v, lw=1, label=f"traj {n+1}")
    for r in res:
        ax.plot(r["theta"], r["v"], "ro", ms=6)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(ylab + " (Ha/rad)")
    ax.set_title(f"Étape {s} : vélocité")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"step{s}_velocity.png"), dpi=150)
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
                      show=False, ylog=True):
    """Figure à 2 panneaux : trajectoires dans le plan complexe | vélocité vs theta.
    Même couleur par trajectoire sur les deux panneaux ; les états dégénérés
    (mêmes E à 1e-5 Ha à tout theta) sont tracés une seule fois."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # regroupe les trajectoires dégénérées
    groups = []
    for n in states:
        for g in groups:
            if np.max(np.abs(T[:, g[0] - 1] - T[:, n - 1])) < 1e-4:
                g.append(n)
                break
        else:
            groups.append([n])

    cmap = plt.get_cmap("tab20")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for k, g in enumerate(groups):
        n = g[0]
        c = cmap(k % 20)
        lab = ",".join(map(str, g)) if len(g) <= 3 else f"{g[0]}-{g[-1]}"
        E = T[:, n - 1]
        ax1.plot(E.real, E.imag, "-", color=c, lw=1.4, label=lab)
        ax1.plot(E.real[0], E.imag[0], "o", color=c, ms=4)            # theta = 0
        ax1.plot(E.real[-1], E.imag[-1], ">", color=c, ms=6)          # theta max
        tf, v, _ = velocity(theta, E, logderiv)
        ax2.plot(tf, v, "-", color=c, lw=1.4, label=lab)
    for r in res:
        ax1.plot(r["E"].real, r["E"].imag, "r*", ms=14, zorder=5)
        ax2.plot(r["theta"], r["v"], "r*", ms=14, zorder=5)

    if threshold is not None:
        ax1.axvline(threshold, color="k", ls=":", lw=.8)
    ax1.axhline(0, color="gray", lw=.5)
    ax1.set_xlabel("Re E (Ha)")
    ax1.set_ylabel("Im E (Ha)")
    ax1.set_title(f"Étape {s} : trajectoires E(θ)   (● θ=0, ▶ θ max)")
    ax1.grid(alpha=.3)

    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_ylabel((r"$|\theta\,dE/d\theta|$" if logderiv else r"$|dE/d\theta|$") + " (Ha/rad)")
    if ylog:
        ax2.set_yscale("log")
    ax2.set_title("Vélocité")
    ax2.grid(alpha=.3, which="both")
    ax2.legend(title="trajectoire", fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    path = os.path.join(outdir, f"step{s}_trajectories_velocity.png")
    fig.savefig(path, dpi=150)
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
    """Regroupe les trajectoires confondues (états dégénérés) -> liste de listes."""
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
    """
    Fichier texte : une ligne par theta, colonnes Re/Im de chaque état choisi (Ha).
      order='tracked' : colonnes = trajectoires (suivies par continuité,
                        numérotées par l'état à theta=0)
      order='sorted'  : colonnes = états dans l'ordre brut de QP2 (tri par Re E)
    """
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
                       theta_marks=0.1, xlim=None, ylim=None, labels=True):
    """
    Rotation des états dans le plan complexe.
    runs : liste de (nom_étape, theta, T). Plusieurs entrées => superposition
    (comparaison des étapes : style de trait différent par étape).
    Les points ◦ le long des courbes marquent les theta multiples de theta_marks.
    """
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
                # repères en theta
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
    if len(runs) > 1:
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        for r, (name, _, _) in enumerate(runs):
            h.append(Line2D([0], [0], color="k", ls=styles[r % len(styles)]))
            l.append(name)
        ax.legend(h, l, fontsize=7, ncol=2, title="états / étapes")
    else:
        ax.legend(fontsize=7, ncol=2, title="état")
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_rotation.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def main():
    class Fmt(argparse.RawDescriptionHelpFormatter):
        """Affiche (défaut : x) seulement quand le défaut est informatif."""
        def _get_help_string(self, action):
            h = action.help or ""
            if action.default not in (None, False, 0.0, argparse.SUPPRESS) \
                    and action.option_strings:
                h += f" [défaut : {action.default}]"
            return h

    epilog = """exemples :
  # analyse complète de toutes les étapes (résonances + rapport de convergence)
  python cs_resonance.py He_cs.out

  # étape convergée seulement, avec diagnostic de vitesse par trajectoire
  python cs_resonance.py He_cs.out --step last --diag

  # exporter les énergies de l'étape 3 (un fichier .dat, une ligne par theta)
  python cs_resonance.py He_cs.out --step 3 --export-energies

  # rotation dans le plan complexe des états 1, 2 et 7 à 20
  python cs_resonance.py He_cs.out --step 3 --rotation --states 1,2,7-20

  # idem avec une fenêtre d'énergie, zoom, et comparaison des 3 étapes
  python cs_resonance.py He_cs.out --rotation --compare --erange -2.1 -0.9 --xlim -2.2 -0.9

  # figure à 2 panneaux : plan complexe | vélocité
  python cs_resonance.py He_cs.out --step last --plot --states 7-20

sorties (dans --outdir) :
  stepN_trajectories.csv           trajectoires + vitesse (format long)
  stepN_energies.dat               énergies par theta (--export-energies)
  stepN_rotation.png               rotation dans le plan complexe (--rotation)
  steps_1-2-3_rotation.png         comparaison des étapes (--rotation --compare)
  stepN_trajectories_velocity.png  plan complexe | vélocité (--plot)
"""
    p = argparse.ArgumentParser(prog="cs_resonance.py", description=__doc__,
                                epilog=epilog, formatter_class=Fmt)
    p.add_argument("file", metavar="FICHIER",
                   help="sortie QP2 contenant les scans de complex scaling")
    p.add_argument("--outdir", default="cs_results", metavar="DOSSIER",
                   help="dossier de sortie (créé si besoin)")

    g = p.add_argument_group("sélection des étapes et des états")
    g.add_argument("--step", default="all", metavar="N|last|all",
                   help="étape(s) de convergence à traiter : 'all', 'last' "
                        "(étape convergée) ou un numéro (1, 2, 3...)")
    g.add_argument("--states", default=None, metavar="LISTE",
                   help="trajectoires à exporter/tracer, ex. '1,3,15-17' "
                        "(défaut : toutes)")
    g.add_argument("--erange", type=float, nargs=2, metavar=("EMIN", "EMAX"),
                   help="ne garde que les états dont Re E(theta=0) est dans "
                        "[EMIN, EMAX] (Ha) ; se combine avec --states")

    g = p.add_argument_group("recherche de résonance (vélocité)")
    g.add_argument("--threshold", type=float, default=None, metavar="E",
                   help="seuil d'ionisation (Ha), ex. -2.0 pour He+ 1s : ne garde que "
                        "les minima avec Re E > E (défaut : désactivé)")
    g.add_argument("--logderiv", action="store_true",
                   help="vélocité = |theta dE/dtheta| au lieu de |dE/dtheta|")
    g.add_argument("--theta-min", type=float, default=0.0, metavar="TH",
                   help="ignore les minima à theta < TH (rad) (défaut : 0 = désactivé)")
    g.add_argument("--prominence", type=float, default=0.0, metavar="P",
                   help="profondeur minimale d'un minimum de vitesse, en fraction "
                        "de la vitesse médiane ; ex. 0.2 (défaut : 0 = désactivé)")
    g.add_argument("--imag-tol", type=float, default=None, metavar="G",
                   help="ne garde que les minima avec Im E < -G (états ayant une "
                        "largeur), ex. 1e-4 (défaut : désactivé)")
    g.add_argument("--diag", action="store_true",
                   help="affiche la vitesse minimale de chaque trajectoire")

    g = p.add_argument_group("export de données")
    g.add_argument("--export-energies", action="store_true",
                   help="écrit stepN_energies.dat : Re/Im de l'énergie de chaque "
                        "état pour chaque theta, pour l'étape choisie")
    g.add_argument("--order", choices=["tracked", "sorted"], default="tracked",
                   help="colonnes de l'export : trajectoires suivies par continuité "
                        "('tracked') ou états dans l'ordre brut de QP2 ('sorted')")

    g = p.add_argument_group("graphiques")
    g.add_argument("--rotation", action="store_true",
                   help="rotation des états dans le plan complexe")
    g.add_argument("--compare", action="store_true",
                   help="avec --rotation : superpose les étapes choisies sur un "
                        "seul graphique")
    g.add_argument("--plot", action="store_true",
                   help="figure à 2 panneaux : plan complexe | vélocité(theta)")
    g.add_argument("--theta-marks", type=float, default=0.1, metavar="PAS",
                   help="pas (rad) des repères en theta sur la rotation")
    g.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   help="bornes de l'axe Re E de la rotation")
    g.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   help="bornes de l'axe Im E de la rotation")
    g.add_argument("--no-labels", action="store_true",
                   help="pas d'étiquettes d'état sur la rotation")
    g.add_argument("--linear-v", action="store_true",
                   help="axe de vélocité linéaire (défaut : logarithmique)")
    g.add_argument("--show", action="store_true",
                   help="ouvre aussi la fenêtre matplotlib (en plus du PNG)")
    g.add_argument("--no-plot", action="store_true",
                   help="désactive les figures par défaut (plan complexe / vélocité)")

    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    scans = parse_cs_energies(a.file)
    print(f"{len(scans)} étape(s) de convergence trouvée(s) ; "
          f"{len(scans[0]['theta'])} valeurs de theta, {scans[0]['E'].shape[1]} états")

    # trajectoires de toutes les étapes (pour le rapport de convergence)
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
                                     ylim=a.ylim, labels=not a.no_labels)
            print(f"  figure rotation : {png}")
        if a.plot:
            png = make_side_by_side(a.outdir, s, theta, T, res, a.threshold,
                                    a.logderiv, st, show=a.show, ylog=not a.linear_v)
            print(f"  figure : {png}")
        if not a.no_plot and not a.plot and not a.rotation:
            make_plots(a.outdir, s, theta, T, V, res, a.threshold, a.logderiv)

    if a.rotation and a.compare:
        runs = [(f"étape {k}", scans[k - 1]["theta"], allT[k - 1]) for k in todo]
        st = select_states(a.states, a.erange, runs[-1][2])
        tag = "steps_" + "-".join(str(k) for k in todo)
        png = make_rotation_plot(a.outdir, tag, runs, a.threshold, st, show=a.show,
                                 theta_marks=a.theta_marks, xlim=a.xlim,
                                 ylim=a.ylim, labels=not a.no_labels)
        print(f"\nfigure comparaison des étapes : {png}")


if __name__ == "__main__":
    main()
