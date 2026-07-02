#!/usr/bin/env python3
"""
state_configurations.py — Assigne à chaque état d'un EZFIO (QP2) sa
configuration électronique dominante (ex. 2s3p, 2p3s, 2p²) et, si les
énergies sont disponibles, son symbole de terme approché (S, P, D, F)
déduit de la dégénérescence.

Usage
-----
    python3 state_configurations.py EZFIO [options]

Arguments & options
-------------------
    EZFIO              répertoire EZFIO (lu seul, jamais modifié)
    --top N            nb de déterminants dominants analysés par état (déf. 8)
    --states LIST      états à analyser, ex. "24,27-29" (déf. tous)
    --core N           nb d'orbitales de cœur à ignorer (déf. auto-détection :
                       orbitales doublement occupées dans tous les
                       déterminants dominants de tous les états)
    --out FILE         sortie QP (fci) où lire le dernier bloc
                       "Energy of state" si EZFIO/fci/energy.gz est absent
    --degen-tol TOL    seuil (Ha) pour grouper les états dégénérés (déf. 1e-4)
    --details          affiche aussi les déterminants dominants de chaque état

Fonctionnement
--------------
    1. Caractère l (s/p/d/f) de chaque OM depuis ao_power + mo_coef
       (poids dominant des AO de chaque l).
    2. Étiquetage des couches par rang parmi les OM de même l :
       s -> 1s,2s,3s… ; p groupées par 3 -> 2p,3p,… ; d par 5 -> 3d,…
       (heuristique valable pour un atome avec des orbitales ordonnées).
    3. Pour chaque état : déterminants de plus grand |c|, occupations de
       valence (cœur retiré), agrégées en labels de configuration pondérés
       par |c|².
    4. Dégénérescence g des groupes d'énergie -> terme : g=1 S, 3 P, 5 D, 7 F.

Limites : heuristique pensée pour des atomes (base s/p/d non hybridée) ;
pour n_det très grand, la lecture de psi_coef.gz peut prendre ~1 min.
Aucune dépendance : stdlib uniquement (pas besoin du module ezfio).

Exemple
-------
    python3 state_configurations.py be.no.14s11p --out Be.no.fci.out
"""

import argparse
import gzip
import heapq
import os
import re
import sys

L_LETTER = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
SHELL_SIZE = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
FIRST_N = {'s': 1, 'p': 2, 'd': 3, 'f': 4}      # 1s, 2p, 3d, 4f
TERM = {1: 'S', 3: 'P', 5: 'D', 7: 'F'}
SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def read_ezfio_array(path):
    """Lit un tableau EZFIO gzippé : rang, dims, données aplaties (Fortran)."""
    with gzip.open(path, 'rt') as f:
        f.readline()
        dims = [int(x) for x in f.readline().split()]
        data = [float(line) for line in f]
    return dims, data


def read_ezfio_scalar(path, cast=int):
    with open(path) as f:
        return cast(f.read().split()[0])


def mo_labels(ez):
    """Retourne la liste des labels de couche par OM (ex. '2p')."""
    dims, pw = read_ezfio_array(os.path.join(ez, 'ao_basis', 'ao_power.gz'))
    nao = dims[0]
    ao_l = [int(pw[i] + pw[i + nao] + pw[i + 2 * nao]) for i in range(nao)]
    dims, mc = read_ezfio_array(os.path.join(ez, 'mo_basis', 'mo_coef.gz'))
    nao2, nmo = dims
    if nao2 != nao:
        sys.exit(f"Erreur : ao_power ({nao} AO) et mo_coef ({nao2} AO) incohérents.")
    labels, rank = [], {}
    for j in range(nmo):
        col = mc[j * nao:(j + 1) * nao]
        w = {}
        for c, l in zip(col, ao_l):
            w[l] = w.get(l, 0.0) + c * c
        letter = L_LETTER.get(max(w, key=w.get), '?')
        r = rank.get(letter, 0)
        rank[letter] = r + 1
        n = FIRST_N.get(letter, 1) + r // SHELL_SIZE.get(letter, 1)
        labels.append(f"{n}{letter}")
    return labels


def read_energies(ez, out_file):
    """Énergies par état : EZFIO/fci/energy.gz, sinon dernier bloc du .out."""
    path = os.path.join(ez, 'fci', 'energy.gz')
    if os.path.exists(path):
        try:
            _, e = read_ezfio_array(path)
            return e
        except Exception:
            pass
    if out_file:
        es = {}
        with open(out_file) as f:
            for line in f:
                m = re.match(r'\* Energy of state\s+(\d+)\s+(-?\d+\.\d+)', line)
                if m:
                    es[int(m.group(1))] = float(m.group(2))  # garde le dernier bloc
        if es:
            return [es[k] for k in sorted(es)]
    return None


def config_label(occ_counts, mo_lab):
    """{orbitale: occupation} -> label trié, ex. '2p3s' ou '2p²'."""
    shells = {}
    for orb, cnt in occ_counts.items():
        lab = mo_lab[orb - 1]
        shells[lab] = shells.get(lab, 0) + cnt
    key = lambda lab: (int(lab[:-1]), 'spdf'.index(lab[-1]))
    parts = []
    for lab in sorted(shells, key=key):
        cnt = shells[lab]
        parts.append(lab + (str(cnt).translate(SUP) if cnt > 1 else ''))
    return ''.join(parts) or '(cœur)'


def parse_states(spec, n_states):
    if not spec:
        return list(range(1, n_states + 1))
    out = []
    for tok in spec.split(','):
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description="Configurations dominantes et termes des états d'un EZFIO QP2.")
    p.add_argument('ezfio')
    p.add_argument('--top', type=int, default=8)
    p.add_argument('--states')
    p.add_argument('--core', type=int, default=-1)
    p.add_argument('--out')
    p.add_argument('--degen-tol', type=float, default=1e-4)
    p.add_argument('--details', action='store_true')
    a = p.parse_args()

    ez = a.ezfio.rstrip('/')
    det_dir = os.path.join(ez, 'determinants')
    n_det = read_ezfio_scalar(os.path.join(det_dir, 'n_det'))
    n_states = read_ezfio_scalar(os.path.join(det_dir, 'n_states'))
    states = parse_states(a.states, n_states)
    mo_lab = mo_labels(ez)
    print(f"EZFIO : {ez}   n_det = {n_det}   n_states = {n_states}")
    print(f"OM    : {' '.join(f'{i+1}:{l}' for i, l in enumerate(mo_lab))}\n")

    with gzip.open(os.path.join(det_dir, 'psi_det.gz'), 'rt') as f:
        f.readline()
        dims = [int(x) for x in f.readline().split()]
        n_int = dims[0]
        ints = [int(line) for line in f]
    if n_int != 1:
        sys.exit("Erreur : N_int > 1 (plus de 64 OM) non géré par ce script.")

    # top-|c| déterminants par état demandé (lecture en flux de psi_coef)
    tops = {s: [] for s in states}
    wanted = set(states)
    with gzip.open(os.path.join(det_dir, 'psi_coef.gz'), 'rt') as f:
        f.readline(); f.readline()
        for idx in range(n_states * n_det):
            st, d = divmod(idx, n_det)
            c = float(f.readline())
            if st + 1 not in wanted:
                continue
            h = tops[st + 1]
            if len(h) < a.top:
                heapq.heappush(h, (abs(c), c, d))
            elif abs(c) > h[0][0]:
                heapq.heapreplace(h, (abs(c), c, d))

    occ = lambda m: [i + 1 for i in range(64) if m >> i & 1]

    # cœur : orbitales doublement occupées dans tous les déterminants dominants
    if a.core >= 0:
        core = set(range(1, a.core + 1))
    else:
        core = None
        for s in states:
            for _, c, d in tops[s]:
                oa, ob = occ(ints[2 * d]), occ(ints[2 * d + 1])
                dbl = set(oa) & set(ob)
                core = dbl if core is None else core & dbl
        core = core or set()
    if core:
        print(f"Cœur détecté : orbitales {sorted(core)} ({', '.join(mo_lab[i-1] for i in sorted(core))})\n")

    energies = read_energies(ez, a.out)

    # groupes dégénérés -> terme
    group = {}
    if energies and len(energies) >= max(states):
        g_id, prev = 0, None
        members = {}
        for s in range(1, n_states + 1):
            if prev is None or abs(energies[s - 1] - prev) > a.degen_tol:
                g_id += 1
            members.setdefault(g_id, []).append(s)
            group[s] = g_id
            prev = energies[s - 1]
        term_of = {g: TERM.get(len(m), f'g={len(m)}') for g, m in members.items()}

    print(f"{'État':>4}  {'E (Ha)':>14}  {'Terme':>5}  Configurations dominantes (poids)")
    print('-' * 78)
    for s in states:
        weights = {}
        for _, c, d in tops[s]:
            counts = {}
            for o in occ(ints[2 * d]) + occ(ints[2 * d + 1]):
                if o not in core:
                    counts[o] = counts.get(o, 0) + 1
            lab = config_label(counts, mo_lab)
            weights[lab] = weights.get(lab, 0.0) + c * c
        best = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
        cfg = ' | '.join(f"{lab} ({w:.2f})" for lab, w in best)
        e_str = f"{energies[s-1]:14.6f}" if energies and len(energies) >= s else ' ' * 14
        t_str = term_of.get(group.get(s), '') if group else ''
        print(f"{s:>4}  {e_str}  {t_str:>5}  {cfg}")
        if a.details:
            for _, c, d in sorted(tops[s], reverse=True):
                oa = ','.join(f"{o}({mo_lab[o-1]})" for o in occ(ints[2 * d]))
                ob = ','.join(f"{o}({mo_lab[o-1]})" for o in occ(ints[2 * d + 1]))
                print(f"      c={c:+.4f}  α[{oa}]  β[{ob}]")


if __name__ == '__main__':
    main()
