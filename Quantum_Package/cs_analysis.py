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
    E_res = E(theta_min),  Gamma = -2 Im[E_res].
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

  # Gros plan automatique sur la résonance (meilleur minimum de vitesse)
  python cs_resonance.py He_cs.out --step last --plot --zoom
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
_FLOAT = r"[-+]?\d\.\d+E[-+]\d+"


def _read_table(lines, j):
    """Lit les lignes 'theta Re Im Re Im ...' à partir de j ; renvoie (rows, j_fin).

    Les lignes tronquées (sortie interrompue) sont ignorées : nombre de colonnes
    impair, différent de la première ligne, ou nombre illisible.
    """
    rows = []
    while j < len(lines):
        s = lines[j].split()
        if s and re.fullmatch(_FLOAT, s[0]):
            try:
                row = [float(x) for x in s]
            except ValueError:
                row = None
            if row and len(row) % 2 == 1 and (not rows or len(row) == len(rows[0])):
                rows.append(row)
        elif rows:           # fin du tableau
            break
        j += 1
    return rows, j


def _scan(rows):
    data = np.array(rows)
    return dict(theta=data[:, 0], E=data[:, 1::2] + 1j * data[:, 2::2])


def _per_theta_from_blocks(lines):
    """Calcul 'CS-CIPSI per-theta' interrompu : reconstruit le tableau final à partir
    du dernier tableau 'Energy of the states for theta' de chaque theta convergé."""
    rows, ndets, last = [], [], None
    for i, l in enumerate(lines):
        if "Energy of the states for theta" in l:
            last = i
        elif (m := re.match(r"\s*Converged theta\s*=\s*(\S+)\s+N_det\s*=\s*(\d+)", l)) and last is not None:
            E = []
            for t in lines[last + 4:]:
                s = t.split()
                if not s or not s[0].isdigit():
                    break
                E += [float(s[1]), float(s[2])]
            rows.append([float(m.group(1))] + E)
            ndets.append(int(m.group(2)))
            last = None
    n = min((len(r) for r in rows), default=0)   # même nombre d'états partout
    return [r[:n] for r in rows], ndets


def parse_cs_energies(path):
    """Retourne une liste d'étapes : dict(theta, E=(nθ, nstates) complexe, ndet, complete).

    Accepte les sorties non terminées (calcul crashé / interrompu) : on garde les
    tableaux complets, on ignore la fin tronquée et on met complete=False.
    """
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    finished = any(l.startswith("Wall time:") for l in lines[-20:])

    scans = []
    i = 0
    while i < len(lines):
        if "CS energies" in lines[i] or "CS per-theta CIPSI energies" in lines[i]:
            nth = None
            for l in lines[i + 1:i + 5]:
                if m := re.match(r"\s*Number of theta values:\s*(\d+)", l):
                    nth = int(m.group(1))
            rows, j = _read_table(lines, i + 1)
            if rows:
                sc = _scan(rows)
                sc["complete"] = nth is None or len(rows) == nth
                scans.append(sc)
            i = j
        i += 1

    ndets = [int(m.group(1)) for l in lines
             if (m := re.match(r"\*\s*Saved determinants\s+(\d+)", l))]
    # sortie tronquée : un "Saved determinants" précède chaque tableau, on aligne sur le début
    ok = len(ndets) == len(scans) or (not finished and len(ndets) > len(scans))
    for k, sc in enumerate(scans):
        sc["ndet"] = ndets[k] if ok else None

    if any("CS-CIPSI per-theta" in l for l in lines):
        rows, nd = _per_theta_from_blocks(lines)
        ndet = (nd[0] if len(set(nd)) == 1 else f"{min(nd)}-{max(nd)}") if nd else None
        if scans:
            scans[-1]["ndet"] = ndet
        elif rows:
            sc = _scan(rows)
            sc["ndet"] = ndet
            sc["complete"] = False
            scans.append(sc)
            nstep = [int(m.group(1)) for l in lines
                     if (m := re.search(r"per-theta -- step\s+\d+\s*/\s*(\d+)", l))]
            tot = f"/{nstep[0]}" if nstep else ""
            print(f"ATTENTION : calcul per-theta non terminé ; {len(rows)}{tot} valeurs de theta "
                  f"convergées reconstruites à partir des blocs 'Converged theta'.")

    if not scans:
        raise RuntimeError("Aucun tableau 'CS energies' (ni theta convergé) trouvé dans le fichier.")
    if len(scans) > 1 and not scans[-1]["complete"]:
        print(f"ATTENTION : dernier tableau 'CS energies' tronqué "
              f"({len(scans[-1]['theta'])} theta) ; étape ignorée.")
        scans.pop()
    if not finished:
        print("ATTENTION : sortie non terminée (pas de 'Wall time' final) : "
              "seules les étapes complètes disponibles sont analysées.")
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
            # Le minimum est détecté sur la spline fine, mais la résonance
            # rapportée doit appartenir à la grille theta réellement calculée.
            # On l'associe au point de grille immédiatement inférieur (convention
            # du scan : le minimum entre deux pas est affecté au pas précédent)
            # et on utilise l'énergie de trajectoire à ce point.
            if tf[i] < theta_min:
                continue
            grid_idx = max(0, int(np.searchsorted(theta, tf[i], side="right") - 1))
            theta_grid = float(theta[grid_idx])
            Er = T[grid_idx, n]
            v_grid = float(np.interp(theta_grid, tf, v))
            if theta_grid < theta_min:
                continue
            if threshold is not None and Er.real <= threshold:
                continue
            if imag_tol is not None and Er.imag > -imag_tol:
                continue
            out.append(dict(traj=n + 1, theta=theta_grid, v=v_grid, E=Er, Gamma=-2 * Er.imag,
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
    ax.set_xlabel("$Re[E]$ (Ha)")
    ax.set_ylabel("$Im[E]$ (Ha)")
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


# --------------------------------------------------------------------------- #
# Encart de zoom automatique sur le plan complexe
# --------------------------------------------------------------------------- #
def _traj_ids(r):
    """'3,4' -> {3, 4} (numéros de trajectoires d'une résonance)."""
    return {int(t) for t in str(r["traj"]).split(",")}


def zoom_box(theta, T, states, res, zoom):
    """
    Rectangle (xmin, xmax, ymin, ymax) englobant la trajectoire COMPLÈTE de la
    résonance, quels que soient les autres états.

    1. résonance : minima de vitesse visibles triés par vitesse croissante
       (restreints à --zoom-states / --zoom-theta s'ils sont donnés) ; on prend
       les --zoom-res premiers (défaut : le meilleur, v minimale) ;
    2. trajectoire complète : à partir de theta_res, on étend vers theta décroissant
       et croissant tant que la vitesse reste <= --zoom-vmax x v_min. Cela englobe
       toute la trajectoire stabilisée (depuis theta = 0 sur l'axe réel) et s'arrête
       au saut vers un autre état / à la fin de la zone stationnaire.
       Avec --zoom-theta TH1 TH2, c'est cette fenêtre qui est imposée ;
    3. le rectangle est la boîte englobante de ces points (+ E_res), avec une marge
       --zoom-margin.
    S'il n'y a aucune résonance : repli sur les trajectoires --zoom-states (ou tous
    les états tracés) sur --zoom-theta (ou tout theta).
    Retourne (boîte, résonances retenues).
    """
    cand = sorted(res or [], key=lambda r: r["v"])
    if zoom.get("states"):
        zs = set(zoom["states"])
        cand = [r for r in cand if _traj_ids(r) & zs]
    if zoom.get("theta"):
        lo, hi = min(zoom["theta"]), max(zoom["theta"])
        cand = [r for r in cand if lo <= r["theta"] <= hi]
    picked = [cand[i - 1] for i in zoom["res"] if 1 <= i <= len(cand)]

    pts = []
    if picked:
        for r in picked:
            for n in sorted(_traj_ids(r)):
                E = T[:, n - 1]
                if zoom.get("theta"):
                    m = (theta >= min(zoom["theta"])) & (theta <= max(zoom["theta"]))
                else:
                    v = np.abs(np.gradient(E, theta)) if len(theta) > 1 else np.zeros(1)
                    k0 = int(np.argmin(np.abs(theta - r["theta"])))
                    lim = zoom["vmax"] * max(v[k0], 1e-12)
                    ka = k0
                    while ka > 0 and v[ka - 1] <= lim:
                        ka -= 1
                    kb = k0
                    while kb < len(theta) - 1 and v[kb + 1] <= lim:
                        kb += 1
                    m = np.zeros(len(theta), dtype=bool)
                    m[ka:kb + 1] = True
                pts.append(E[m])
            pts.append(np.array([r["E"]]))
    else:
        traj = [n for n in (zoom.get("states") or states) if 1 <= n <= T.shape[1]]
        m = (theta >= min(zoom["theta"])) & (theta <= max(zoom["theta"])) \
            if zoom.get("theta") else np.ones(len(theta), dtype=bool)
        if not m.any():
            m[:] = True
        pts.append(T[m][:, [n - 1 for n in traj]].ravel())
    pts = np.concatenate(pts)

    x0, x1 = pts.real.min(), pts.real.max()
    y0, y1 = pts.imag.min(), pts.imag.max()
    # étendue minimale (trajectoire quasi horizontale/verticale ou point unique)
    dx, dy = x1 - x0, y1 - y0
    span = max(dx, dy)
    dx, dy = max(dx, 0.2 * span, 1e-8), max(dy, 0.2 * span, 1e-8)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    f = 1.0 + 2 * zoom["margin"]
    box = (cx - f * dx / 2, cx + f * dx / 2,
           cy - f * dy / 2, cy + f * dy / 2)
    # Bornes manuelles prioritaires : pratique lorsque la résonance est très
    # contractée sur l'axe réel et que l'encadrement automatique est trop large.
    if zoom.get("xlim") is not None:
        box = (zoom["xlim"][0], zoom["xlim"][1], box[2], box[3])
    if zoom.get("ylim") is not None:
        box = (box[0], box[1], zoom["ylim"][0], zoom["ylim"][1])
    if box[0] >= box[1] or box[2] >= box[3]:
        raise ValueError("Les bornes --zoom-xlim et --zoom-ylim doivent être croissantes.")
    return box, picked


def _picked_ids(picked):
    return set().union(*[_traj_ids(r) for r in picked]) if picked else None


def _zoom_title(picked):
    if not picked:
        return None
    return "  |  ".join(f"traj {r['traj']}, θ={r['theta']:.3f}" for r in picked)


def _inset_rect(loc, size):
    """Rectangle [x0, y0, w, h] en coordonnées d'axes, entièrement dans l'axe."""
    # Marges internes réduites pour maximiser la surface utile tout en gardant
    # l'encart et ses graduations dans le cadre du graphique principal.
    pad = 0.035
    x0 = pad if "left" in loc else 1 - size - pad
    y0 = pad if "lower" in loc else 1 - size - pad
    return [x0, y0, size, size]


def _auto_loc(ax, curves, size, source_box=None):
    """Recherche une position libre en pénalisant les courbes et la zone source."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    U, V = [], []
    # Rééchantillonnage uniforme le long des courbes : le score ne dépend ainsi
    # pas du nombre de points numériques choisi par le calcul.
    for E, _, _, _ in curves:
        E = np.asarray(E).ravel()
        if E.size == 0:
            continue
        uu = (E.real - xl[0]) / (xl[1] - xl[0])
        vv = (E.imag - yl[0]) / (yl[1] - yl[0])
        valid = np.isfinite(uu) & np.isfinite(vv)
        uu, vv = uu[valid], vv[valid]
        if not len(uu):
            continue
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(uu), np.diff(vv)))))
        if arc[-1] > 0:
            sample = np.linspace(0, arc[-1], max(80, int(arc[-1] * 500)))
            U.extend(np.interp(sample, arc, uu).tolist())
            V.extend(np.interp(sample, arc, vv).tolist())
        else:
            U.extend(uu.tolist()); V.extend(vv.tolist())
    u, v = np.asarray(U), np.asarray(V)
    pad = 0.035
    max0 = 1 - size - pad
    # Grille plus fine, en autorisant l'encart au milieu des bords et pas seulement
    # dans les quatre coins.
    grid = np.linspace(pad, max0, 25)
    best = None
    # Convertit la fenêtre agrandie en coordonnées normalisées pour éviter de
    # placer l'encart directement par-dessus la région source.
    src = None
    if source_box is not None:
        xmin, xmax, ymin, ymax = source_box
        sx0 = (xmin - xl[0]) / (xl[1] - xl[0]); sx1 = (xmax - xl[0]) / (xl[1] - xl[0])
        sy0 = (ymin - yl[0]) / (yl[1] - yl[0]); sy1 = (ymax - yl[0]) / (yl[1] - yl[0])
        src = (min(sx0, sx1), max(sx0, sx1), min(sy0, sy1), max(sy0, sy1))
    for x0 in grid:
        for y0 in grid:
            hit = ((u >= x0) & (u <= x0 + size) &
                   (v >= y0) & (v <= y0 + size))
            # Longueur échantillonnée de trajectoire masquée (score principal).
            score = float(np.count_nonzero(hit))
            if src is not None:
                sx0, sx1, sy0, sy1 = src
                overlap_x = max(0.0, min(x0 + size, sx1) - max(x0, sx0))
                overlap_y = max(0.0, min(y0 + size, sy1) - max(y0, sy0))
                # Très forte pénalité si l'encart recouvre la zone dont on fait le zoom.
                score += 1e6 * overlap_x * overlap_y
            # En cas d'égalité, préférer une position près des bords.
            score += 0.01 * (min(x0, 1 - size - x0) + min(y0, 1 - size - y0))
            if best is None or score < best[0]:
                best = (score, x0, y0)
    _, x0, y0 = best
    return f"xy:{x0:.6f}:{y0:.6f}"


def add_zoom_inset(ax, curves, box, zoom, res_pts=(), threshold=None, title=None,
                   only=None):
    """Ajoute un encart sans marqueurs-croix, un cadre source et ses connecteurs."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from matplotlib.patches import Rectangle, ConnectionPatch
    xmin, xmax, ymin, ymax = box
    loc = zoom["loc"]
    manual_xy = zoom.get("position")
    if manual_xy is not None:
        x0, y0 = manual_xy
        size = zoom["size"]
        if not (0 <= x0 <= 1-size and 0 <= y0 <= 1-size):
            raise ValueError("--zoom-position X Y doit placer le coin inférieur gauche dans le graphique : 0 <= X,Y et X+taille,Y+taille <= 1.")
        rect_axes = [x0, y0, size, size]
    else:
        if loc == "auto":
            loc = _auto_loc(ax, curves, zoom["size"], source_box=box)
        if loc.startswith("xy:"):
            _, xs, ys = loc.split(":")
            rect_axes = [float(xs), float(ys), zoom["size"], zoom["size"]]
        else:
            rect_axes = _inset_rect(loc, zoom["size"])
    axins = ax.inset_axes(rect_axes, zorder=6)

    for E, c, ls, g in curves:
        if only and not (set(g) & only):
            continue
        E = np.asarray(E)
        inside = np.where((E.real >= xmin) & (E.real <= xmax) &
                          (E.imag >= ymin) & (E.imag <= ymax))[0]
        # Tracer des segments continus ; pas de marqueurs '+' dans l'encart.
        for run in np.split(inside, np.where(np.diff(inside) > 1)[0] + 1):
            if len(run):
                axins.plot(E.real[run], E.imag[run], ls, color=c, lw=1.35,
                           solid_capstyle="round")
    for z in res_pts:
        if xmin <= z.real <= xmax and ymin <= z.imag <= ymax:
            axins.plot(z.real, z.imag, "ro", ms=5, zorder=8)
    axins.axhline(0, color="gray", lw=.5, zorder=0)
    if threshold is not None and xmin <= threshold <= xmax:
        axins.axvline(threshold, color="k", ls=":", lw=.8)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.xaxis.set_major_locator(MaxNLocator(3))
    axins.yaxis.set_major_locator(MaxNLocator(3))
    for axis, span in ((axins.xaxis, xmax - xmin), (axins.yaxis, ymax - ymin)):
        d = max(0, int(np.ceil(-np.log10(span))) + 1)
        axis.set_major_formatter(FormatStrFormatter(f"%.{d}f"))
    axins.tick_params(labelsize=7)
    axins.grid(alpha=.3)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.25")
    if title:
        axins.set_title(title, fontsize=7, pad=2)

    # Rectangle on the main plot marks the exact source region.
    source = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                       fill=False, edgecolor="0.35", linewidth=1.0,
                       linestyle="--", zorder=5)
    ax.add_patch(source)

    # Two fine connectors link the source rectangle to the inset. ConnectionPatch
    # uses data coordinates for the source and axes coordinates for the inset.
    # Select the nearest inset-side corners to reduce crossing over the plot.
    x0, y0, w, h = rect_axes
    inset_center = (x0 + w/2, y0 + h/2)
    source_center_axes = ax.transAxes.inverted().transform(
        ax.transData.transform(((xmin+xmax)/2, (ymin+ymax)/2)))
    if inset_center[0] < source_center_axes[0]:
        inset_pts = ((1, 1), (1, 0))
    else:
        inset_pts = ((0, 1), (0, 0))
    source_pts = ((xmin, ymax), (xmin, ymin)) if inset_center[0] < source_center_axes[0] else ((xmax, ymax), (xmax, ymin))
    for (xy, xyins) in zip(source_pts, inset_pts):
        con = ConnectionPatch(xyA=xy, coordsA=ax.transData,
                              xyB=xyins, coordsB=axins.transAxes,
                              color="0.4", lw=0.9, alpha=0.85,
                              clip_on=False, zorder=4)
        ax.figure.add_artist(con)
    return axins


def make_side_by_side(outdir, s, theta, T, res, threshold, logderiv, states,
                      show=False, ylog=True, xlim=None, ylim=None,
                      theta_range=None, show_resonances=True, show_legend=True,
                      zoom=None, label_resonances=False):
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
    drawn = []                       # courbes tracées (pour l'encart de zoom)
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
            drawn.append((E, c, "-", g))
            ax1.plot(E.real[0], E.imag[0], "o", color=c, ms=4)            # début de l'intervalle
            ax1.plot(E.real[-1], E.imag[-1], ">", color=c, ms=6)          # fin de l'intervalle
            tf, v, _ = velocity(theta_sub, E, logderiv)
            ax2.plot(tf, v, "-", color=c, lw=1.4, label=lab)

    # résonances visibles dans la fenêtre / les états choisis
    sel = set(states)
    res_vis = []
    for r in res:
        if not (set(int(t) for t in str(r["traj"]).split(",")) & sel):
            continue
        if theta_range is not None:
            if single_point:
                if abs(r["theta"] - theta_sub[0]) > 1e-3:
                    continue
            elif not (th_min <= r["theta"] <= th_max):
                continue
        res_vis.append(r)
    if show_resonances:
        for r in res_vis:
            ax1.plot(r["E"].real, r["E"].imag, "ro", ms=5, zorder=5)
            ax2.plot(r["theta"], r["v"], "ro", ms=5, zorder=5)


    if threshold is not None:
        ax1.axvline(threshold, color="k", ls=":", lw=.8)
    ax1.axhline(0, color="gray", lw=.5)
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("$Re[E]$ (Ha)")
    ax1.set_ylabel("$Im[E]$ (Ha)")
    if single_point:
        ax1.set_title(f"Étape {s} : états à θ = {theta_sub[0]:.4f} rad")
    else:
        ax1.set_title(f"Étape {s} : trajectoires E(θ)   (● θ min, ▶ θ max)")
    if label_resonances and res_vis:
        label_text = "\n".join(
            f"$E_R$ = ({r['E'].real:.5f}{r['E'].imag:+.5f}i) Ha\n"
            rf"$\theta_{{\mathrm{{opt}}}}$ = {r['theta']:.4f} rad"
            for r in res_vis
        )
        ax1.text(0.02, 0.025, label_text, transform=ax1.transAxes,
                 ha="left", va="bottom", fontsize=11, color="darkred",
                 zorder=8, clip_on=True)
    ax1.grid(alpha=.3)

    if zoom is not None and not single_point and drawn:
        box, picked = zoom_box(theta_sub, T_sub, states, res_vis, zoom)
        manual_zoom_box = (zoom.get("xlim") is not None or
                           zoom.get("ylim") is not None)
        # Avec des bornes manuelles, afficher toutes les trajectoires qui
        # traversent le cadre : la résonance automatiquement classée n°1 peut
        # se trouver hors de cette fenêtre et ne doit pas vider l'encart.
        picked_in_box = [
            r for r in picked
            if box[0] <= r["E"].real <= box[1]
            and box[2] <= r["E"].imag <= box[3]
        ]
        add_zoom_inset(ax1, drawn, box, zoom,
                       res_pts=[r["E"] for r in (picked_in_box if manual_zoom_box else picked)]
                               if show_resonances else (),
                       threshold=threshold,
                       title=(_zoom_title(picked_in_box if manual_zoom_box else picked)),
                       only=None if manual_zoom_box else _picked_ids(picked))

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
    """Sélection par numéros ('1,3,15-17') et/ou fenêtre d'énergie Re[E](theta=0)."""
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
                       theta_marks=0.1, xlim=None, ylim=None, labels=True, show_legend=True,
                       zoom=None, res=None, label_resonances=False):
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
    drawn = []

    for r, (name, theta, T) in enumerate(runs):
        ls = styles[r % len(styles)]
        for k, g in enumerate(groups):
            n = g[0]
            c = cmap(k % 20)
            E = T[:, n - 1]
            drawn.append((E, c, ls, g))
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

    if res and label_resonances:
        for rr in res:
            if set(_traj_ids(rr)) & set(states):
                ax.plot(rr["E"].real, rr["E"].imag, "ro", ms=5, zorder=7)
                txt = (f"E=({rr['E'].real:.5f}{rr['E'].imag:+.5f}i) Ha\n"
                       f"θ={rr['theta']:.4f} rad")
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                fx = (rr["E"].real - x0) / (x1 - x0) if x1 != x0 else 0.5
                fy = (rr["E"].imag - y0) / (y1 - y0) if y1 != y0 else 0.5
                ox = -10 if fx > 0.65 else 10
                oy = -10 if fy > 0.70 else 10
                ax.annotate(txt, (rr["E"].real, rr["E"].imag), xytext=(ox, oy),
                            textcoords="offset points", ha="right" if ox < 0 else "left",
                            va="top" if oy < 0 else "bottom", fontsize=7, color="darkred",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.88),
                            arrowprops=dict(arrowstyle="-", color="darkred", lw=0.6),
                            annotation_clip=True, zorder=8)

    if threshold is not None:
        ax.axvline(threshold, color="k", ls=":", lw=.8, label=f"seuil {threshold}")
    ax.axhline(0, color="gray", lw=.5)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("$Re[E]$ (Ha)")
    ax.set_ylabel("$Im[E]$ (Ha)")
    names = ", ".join(r[0] for r in runs)
    ax.set_title(f"Rotation des états dans le plan complexe ({names})\n"
                 f"■ θ=0   ● tous les {theta_marks} rad   ▶ θ max")
    ax.grid(alpha=.3)

    if zoom is not None and drawn:
        th_ref, T_ref = runs[-1][1], runs[-1][2]
        box, picked = zoom_box(th_ref, T_ref, states, res, zoom)
        manual_zoom_box = (zoom.get("xlim") is not None or
                           zoom.get("ylim") is not None)
        picked_in_box = [
            r for r in picked
            if box[0] <= r["E"].real <= box[1]
            and box[2] <= r["E"].imag <= box[3]
        ]
        add_zoom_inset(ax, drawn, box, zoom,
                       res_pts=[r["E"] for r in (picked_in_box if manual_zoom_box else picked)],
                       threshold=threshold,
                       title=_zoom_title(picked_in_box if manual_zoom_box else picked),
                       only=None if manual_zoom_box else _picked_ids(picked))

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
                   help="ne garde que les états dont Re[E](theta=0) est dans [EMIN, EMAX]")

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
                   help="ne garde que les minima avec Im[E] < -G")
    g.add_argument("--diag", action="store_true",
                   help="affiche la vitesse minimale de chaque trajectoire")
    g.add_argument("--scan-analysis", action="store_true",
                   help="analyse uniquement la trajectoire associée à la meilleure résonance retenue après application de --prominence et des autres filtres")

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
    g.add_argument("--label-resonances", action="store_true",
                   help="affiche près des points rouges l'énergie complexe et theta de chaque résonance")
    g.add_argument("--no-legend", action="store_true",
                   help="desactive complètement l'affichage de la légende")
    g.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   help="bornes de l'axe Re[E]")
    g.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   help="bornes de l'axe Im[E]")
    g.add_argument("--no-labels", action="store_true",
                   help="pas d'étiquettes d'état")
    g.add_argument("--linear-v", action="store_true",
                   help="axe de vélocité linéaire")
    g = p.add_argument_group("zoom automatique (encart sur le plan complexe)")
    g.add_argument("--zoom", action="store_true",
                   help="ajoute un encart en gros plan autour de la résonance (meilleur "
                        "minimum de vitesse) dans le plan complexe (avec --plot "
                        "et/ou --rotation)")
    g.add_argument("--zoom-res", default="1", metavar="LISTE",
                   help="résonance(s) à agrandir, par rang dans la liste triée par "
                        "vitesse croissante (1 = meilleur minimum), ex. '1' ou '1,3'")
    g.add_argument("--zoom-states", default=None, metavar="LISTE",
                   help="ne considère que les résonances de ces trajectoires, "
                        "ex. '34,35'")
    g.add_argument("--zoom-theta", type=float, nargs=2, metavar=("TH1", "TH2"),
                   help="impose la fenêtre en theta [TH1, TH2] définissant la "
                        "trajectoire à encadrer (défaut : automatique)")
    g.add_argument("--zoom-vmax", type=float, default=20.0, metavar="K",
                   help="la trajectoire de la résonance est suivie tant que sa "
                        "vitesse <= K x v_min (arrêt au saut vers un autre état)")
    g.add_argument("--zoom-margin", type=float, default=0.15, metavar="M",
                   help="marge autour de la zone agrandie (fraction de sa taille)")
    g.add_argument("--zoom-loc", default="auto",
                   choices=["auto", "lower left", "lower right", "upper left", "upper right"],
                   help="position de l'encart ('auto' = coin le moins chargé)")
    g.add_argument("--zoom-size", type=float, default=0.30, metavar="S",
                   help="taille de l'encart (fraction de l'axe ; défaut : 0.30)")
    g.add_argument("--zoom-position", type=float, nargs=2, metavar=("X", "Y"),
                   help="position manuelle du coin inférieur gauche de l'encart, en fractions des axes (0 à 1), ex. 0.55 0.08 ; prioritaire sur --zoom-loc")
    g.add_argument("--zoom-xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                   help="impose les bornes Re[E] du zoom (Ha), ex. -2.1 -1.8")
    g.add_argument("--zoom-ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                   help="impose les bornes Im[E] du zoom (Ha), ex. -0.2 0.02")
    g.add_argument("--show", action="store_true",
                   help="ouvre la fenêtre matplotlib")
    g.add_argument("--no-plot", action="store_true",
                   help="désactive les figures par défaut")

    a = p.parse_args()

    zoom = None
    if a.zoom:
        zoom = dict(res=parse_states(a.zoom_res, 10**6),
                    states=parse_states(a.zoom_states, 10**6) if a.zoom_states else None,
                    theta=a.zoom_theta, vmax=a.zoom_vmax, margin=a.zoom_margin,
                    loc=a.zoom_loc, size=a.zoom_size, position=a.zoom_position,
                    xlim=a.zoom_xlim, ylim=a.zoom_ylim)

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
        if not a.scan_analysis:
            if not res:
                print("  aucun minimum de vitesse trouvé (avec les filtres actifs)")
            else:
                print("  traj      theta     Re[E] (Ha)    Im[E] (Ha)    Re[E] (eV)   Gamma (eV)   v_min     mult")
                for r in res:
                    print(f"  {str(r['traj']):8s}  {r['theta']:.4f}  {r['E'].real:11.6f}  "
                          f"{r['E'].imag:11.6f}  {r['E'].real*HARTREE_TO_EV:10.4f}  "
                          f"{r['Gamma']*HARTREE_TO_EV:10.4f}  {r['v']:.3e}  {r['mult']}")
        if a.scan_analysis:
            print("  -- analyse de la trajectoire associée à la résonance retenue --")
            if not res:
                print("    aucune résonance ne satisfait les filtres, notamment --prominence ; aucune trajectoire à analyser.")
            else:
                # Une seule trajectoire est analysée : celle du minimum retenu
                # en tête du classement (plus faible vélocité parmi les minima
                # ayant passé le filtre de prominence et les autres contraintes).
                best = res[0]
                traj_id = int(str(best["traj"]).split(",")[0])
                oneT = T[:, traj_id - 1:traj_id]
                one_res = find_resonances(theta, oneT, a.threshold, a.logderiv,
                                          a.theta_min, imag_tol=a.imag_tol,
                                          prominence=a.prominence)
                print(f"    trajectoire : {traj_id}")
                print(f"    grille theta : {len(theta)} points, [{theta.min():.5f}, {theta.max():.5f}] rad")
                print(f"    filtre prominence : >= {a.prominence:g} (normalisée par la vélocité médiane)")
                print(f"    minima de cette trajectoire satisfaisant les filtres : {len(one_res)}")
                if one_res:
                    print("    theta(rad)      Re[E](Ha)       Im[E](Ha)       Gamma(eV)      v_min       prominence_norm")
                    for rr in one_res:
                        print(f"    {rr['theta']:10.5f}  {rr['E'].real:12.7f}  {rr['E'].imag:12.7f}  "
                              f"{rr['Gamma']*HARTREE_TO_EV:12.6f}  {rr['v']:.4e}  {rr['depth']:.4g}")
                else:
                    print("    aucun minimum de cette trajectoire ne passe les filtres actifs.")
                print("    résonance de référence : "
                      f"theta={best['theta']:.5f} rad, E=({best['E'].real:.7f}{best['E'].imag:+.7f}i) Ha, "
                      f"prominence_norm={best['depth']:.4g}")
        if a.diag:
            print("  -- diagnostic : vitesse minimale par trajectoire (theta >= theta-min) --")
            print("  traj  E(th=0)    v_min      theta   Re[E]       Im[E]      v_median")
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
                                     show_legend=not a.no_legend, zoom=zoom, res=res,
                                     label_resonances=a.label_resonances)
            print(f"  figure rotation : {png}")
        if a.plot:
            png = make_side_by_side(a.outdir, s, theta, T, res, a.threshold,
                                    a.logderiv, st, show=a.show, ylog=not a.linear_v,
                                    xlim=a.xlim, ylim=a.ylim,
                                    theta_range=a.theta_range,
                                    show_resonances=not a.no_resonances,
                                    show_legend=not a.no_legend, zoom=zoom,
                                    label_resonances=a.label_resonances)
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
                                 show_legend=not a.no_legend, zoom=zoom,
                                 res=find_resonances(runs[-1][1], runs[-1][2], a.threshold, a.logderiv, a.theta_min,
                                                     imag_tol=a.imag_tol, prominence=a.prominence),
                                 label_resonances=a.label_resonances)
        print(f"\nfigure comparaison des étapes : {png}")


if __name__ == "__main__":
    main()