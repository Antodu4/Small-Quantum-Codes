#!/usr/bin/env python3
"""
Classify each MO by its dominant AO angular-momentum character (s/p/d/f/...),
from the weight of its coefficients on each AO block.

Reads either:
  - a Molden (.mol) file, or
  - a QP2 EZFIO directory (reading ao_basis/ao_power and mo_basis/mo_coef
    directly, no Molden export needed).
The input type is auto-detected: a directory is treated as EZFIO, a file as
Molden.

Usage:
    python3 classify_mo_character.py be.24-26.nat.mol
    python3 classify_mo_character.py be.24-26.nat.mol --csv be.24-26.nat.mo_character.csv
    python3 classify_mo_character.py be.nat.14s11p
    python3 classify_mo_character.py be.nat.14s11p --out be.nat.14s11p.out --csv be.nat.14s11p.mo_character.csv
"""
import argparse
import csv
import gzip
import os
import re
import sys

N_AO_PER_SHELL = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11}
L_LABELS = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}


# ---------------------------------------------------------------- Molden ----

def parse_ao_shell_map_molden(text):
    """Return a list giving the shell-type letter for each AO index (1-based)."""
    gto_start = text.index("[GTO]")
    mo_start = text.index("[MO]")
    gto_block = text[gto_start:mo_start]

    ao_types = []
    for line in gto_block.split("\n"):
        m = re.match(r"^\s*([spdfgh])\s+\d+\s+[\d.]+", line)
        if m:
            shell = m.group(1)
            ao_types.extend([shell] * N_AO_PER_SHELL[shell])
    return ao_types


def parse_mos_molden(text):
    """Return a list of dicts: {ene, occup, coefs (1-based dict)}."""
    mo_start = text.index("[MO]")
    mo_block = text[mo_start + len("[MO]"):]

    mos = []
    cur = None
    for line in mo_block.split("\n"):
        if re.match(r"^\s*Sym=", line):
            if cur is not None:
                mos.append(cur)
            cur = {"ene": None, "occup": None, "coefs": {}}
        elif re.match(r"^\s*Ene=", line):
            cur["ene"] = float(line.split("=")[1])
        elif re.match(r"^\s*Occup=", line):
            cur["occup"] = float(line.split("=")[1])
        else:
            m = re.match(r"^\s*(\d+)\s+(-?\d+\.\d+E?[+-]?\d*)\s*$", line)
            if m and cur is not None:
                cur["coefs"][int(m.group(1))] = float(m.group(2))
    if cur is not None:
        mos.append(cur)
    return mos


def load_molden(path):
    with open(path) as f:
        text = f.read()
    return parse_ao_shell_map_molden(text), parse_mos_molden(text)


# ----------------------------------------------------------------- EZFIO ----

def _prod(seq):
    p = 1
    for x in seq:
        p *= x
    return p


def _ezfio_path(ezfio_dir, *parts):
    path = os.path.join(ezfio_dir, *parts) + ".gz"
    if not os.path.exists(path):
        path = path[:-3]
    return path


def _read_ezfio_array(path):
    """Read an EZFIO text array file (.gz or plain): rank line, shape line,
    then flattened values in Fortran (column-major) order."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        rank = int(f.readline().split()[0])
        shape = [int(x) for x in f.readline().split()]
        assert len(shape) == rank, (shape, rank)
        values = [f.readline().strip() for _ in range(_prod(shape))]
    return shape, values


def parse_ao_shell_map_ezfio(ezfio_dir):
    shape, values = _read_ezfio_array(_ezfio_path(ezfio_dir, "ao_basis", "ao_power"))
    ao_num, ndim = shape
    assert ndim == 3, "expected (ao_num, 3) for lx,ly,lz"
    # Fortran column-major: values ordered as power(1,1),power(2,1),...,power(ao_num,1),
    # power(1,2),...  i.e. lx for all AOs, then ly for all AOs, then lz for all AOs.
    lx = [int(v) for v in values[0:ao_num]]
    ly = [int(v) for v in values[ao_num:2 * ao_num]]
    lz = [int(v) for v in values[2 * ao_num:3 * ao_num]]
    return [L_LABELS.get(lx[i] + ly[i] + lz[i], f"l={lx[i] + ly[i] + lz[i]}") for i in range(ao_num)]


def parse_mo_coef_ezfio(ezfio_dir, ao_num):
    shape, values = _read_ezfio_array(_ezfio_path(ezfio_dir, "mo_basis", "mo_coef"))
    assert shape[0] == ao_num, shape
    mo_num = shape[1]
    values = [float(v.replace("D", "E").replace("d", "e")) for v in values]
    # Fortran column-major: mo_coef(ao, mo) -> column mo has ao_num consecutive values
    return [values[m * ao_num:(m + 1) * ao_num] for m in range(mo_num)]


def parse_mo_occ_ezfio(ezfio_dir, mo_num):
    path = _ezfio_path(ezfio_dir, "mo_basis", "mo_occ")
    if not os.path.exists(path):
        return [None] * mo_num
    shape, values = _read_ezfio_array(path)
    return [float(v) for v in values]


def parse_mo_occ_from_out(out_path, mo_num):
    """Read natural-orbital occupations from the 'Eigenvalues' table printed
    by save_natorb in a QP2 .out log. This is the reliable source: mo_occ in
    the EZFIO is not always updated by the natural-orbital step and can be
    left stale (e.g. only the first couple of entries set, the rest at 0)."""
    with open(out_path) as f:
        lines = f.readlines()

    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "Eigenvalues")
    except StopIteration:
        return None

    # data rows sit between the 2nd and 3rd '====' border lines after the title
    border_idxs = [i for i in range(start, len(lines)) if lines[i].strip().startswith("=====")]
    if len(border_idxs) < 3:
        return None
    data_lines = lines[border_idxs[1] + 1:border_idxs[2]]

    occ = {}
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            occ[int(parts[0])] = float(parts[1])
    if len(occ) != mo_num:
        return None
    return [occ[i] for i in range(1, mo_num + 1)]


def load_ezfio(ezfio_dir, out_path=None):
    ao_types = parse_ao_shell_map_ezfio(ezfio_dir)
    mo_coefs = parse_mo_coef_ezfio(ezfio_dir, len(ao_types))

    occ = None
    if out_path:
        occ = parse_mo_occ_from_out(out_path, len(mo_coefs))
        if occ is None:
            print(f"Warning: could not parse Eigenvalues table from {out_path}, "
                  f"falling back to EZFIO mo_occ (may be stale)", file=sys.stderr)
    if occ is None:
        occ = parse_mo_occ_ezfio(ezfio_dir, len(mo_coefs))

    mos = [{"ene": None, "occup": o, "coefs": {i + 1: c for i, c in enumerate(col)}}
           for col, o in zip(mo_coefs, occ)]
    return ao_types, mos


# ------------------------------------------------------------- classify -----

def classify(mos, ao_types):
    results = []
    for i, mo in enumerate(mos, start=1):
        weights = {}
        for ao_idx, c in mo["coefs"].items():
            shell = ao_types[ao_idx - 1]
            weights[shell] = weights.get(shell, 0.0) + c * c
        total = sum(weights.values())
        dominant = max(weights, key=weights.get)
        purity = 100.0 * weights[dominant] / total if total > 0 else 0.0
        results.append({
            "mo": i, "ene": mo["ene"], "occup": mo["occup"],
            "character": dominant, "purity_pct": purity, "weights": weights,
        })
    return results


def print_table(results):
    has_ene = any(r["ene"] is not None for r in results)
    if has_ene:
        print(f"{'OM':>3} {'Ene (Ha)':>14} {'Occup':>10} {'Caractère':>10} {'Pureté %':>9}")
        for r in results:
            print(f"{r['mo']:>3} {r['ene']:>14.6f} {r['occup']:>10.6f} {r['character']:>10} {r['purity_pct']:>9.2f}")
    else:
        print(f"{'OM':>3} {'Occup':>10} {'Caractère':>10} {'Pureté %':>9}")
        for r in results:
            occ_str = f"{r['occup']:.6f}" if r["occup"] is not None else "NA"
            print(f"{r['mo']:>3} {occ_str:>10} {r['character']:>10} {r['purity_pct']:>9.2f}")


def write_csv(path, results):
    has_ene = any(r["ene"] is not None for r in results)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if has_ene:
            w.writerow(["mo", "ene", "occup", "character", "purity_pct"])
            for r in results:
                w.writerow([r["mo"], f"{r['ene']:.8f}", f"{r['occup']:.8f}", r["character"], f"{r['purity_pct']:.2f}"])
        else:
            w.writerow(["mo", "occup", "character", "purity_pct"])
            for r in results:
                occ_str = f"{r['occup']:.8f}" if r["occup"] is not None else ""
                w.writerow([r["mo"], occ_str, r["character"], f"{r['purity_pct']:.2f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Molden (.mol) file or QP2 EZFIO directory")
    parser.add_argument("--out", metavar="FILE",
                         help="QP2 .out log with the natural-orbital 'Eigenvalues' table "
                              "(EZFIO input only; preferred source for Occup, which can be stale in the EZFIO)")
    parser.add_argument("--csv", metavar="FILE", help="also write the table to this CSV file")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        ao_types, mos = load_ezfio(args.input, args.out)
    else:
        if args.out:
            sys.exit("Error: --out is only meaningful for an EZFIO directory input.")
        ao_types, mos = load_molden(args.input)

    if len(mos) == 0:
        sys.exit(f"No MOs parsed in {args.input}")

    results = classify(mos, ao_types)
    print(f"# {args.input} -- {len(ao_types)} AOs, {len(mos)} MOs\n")
    print_table(results)

    if args.csv:
        write_csv(args.csv, results)
        print(f"\nWritten: {args.csv}")


if __name__ == "__main__":
    main()
