#!/usr/bin/env python3
"""
Extract per-state energies from a QP2 FCI output (.out) and classify states
by orbital angular momentum character (s/p/d/f/...) from their (2l+1)-fold
energy degeneracy pattern.

Usage:
    python3 extract_fci_energies.py Be.nat.fci.out
    python3 extract_fci_energies.py Be.nat.fci.out --csv Be.nat.energies.csv
    python3 extract_fci_energies.py Be.nat.fci.out --tol 5e-4
"""
import argparse
import csv
import re
import sys

L_LABELS = {1: 's', 3: 'p', 5: 'd', 7: 'f', 9: 'g', 11: 'h'}


def find_checkpoints(text):
    return [int(m.group(1)) for m in re.finditer(r"Summary at N_det =\s*(\d+)", text)]


def parse_raw_states(text, ndet):
    """Raw E and E+PT2 for each state at the given N_det checkpoint."""
    marker = f"N_det             =      {ndet:>7}"
    start = text.find(marker)
    if start == -1:
        # fall back to a looser match on the bare number
        start = text.find(f"N_det             =")
        for m in re.finditer(r"N_det             =\s*(\d+)", text):
            if int(m.group(1)) == ndet:
                start = m.start()
                break
    end = text.find("minimum PT2 Extrapolated energy", start)
    segment = text[start:end]

    states = {}
    pattern = re.compile(
        r"\* State\s+(\d+)\s*\n"
        r".*?E\s*=\s*(-?\d+\.\d+)\s*\n"
        r".*?E\+PT2\s*=\s*(-?\d+\.\d+)",
        re.S,
    )
    for m in pattern.finditer(segment):
        st, e, ept2 = int(m.group(1)), float(m.group(2)), float(m.group(3))
        states[st] = (e, ept2)
    return states


def parse_extrapolated(lines, ndet):
    """Best (smallest |PT2|) extrapolated E+PT2 for each state, read from the
    per-state extrapolation tables that follow the last checkpoint block."""
    start = None
    for i, l in enumerate(lines):
        if re.search(rf"N_det             =\s*{ndet}\b", l):
            start = i
            break
    if start is None:
        return {}

    extrap = {}
    cur_state = None
    for i in range(start, len(lines)):
        l = lines[i]
        m = re.match(r"\s*State\s+(\d+)\s*$", l)
        if m:
            cur_state = int(m.group(1))
        elif cur_state is not None and re.match(r"\s*minimum PT2", l):
            row = lines[i + 2].split()
            extrap[cur_state] = (float(row[0]), float(row[1]))
            cur_state = None
    return extrap


def group_by_degeneracy(states, extrap, tol):
    """Group consecutive states whose extrapolated energy differs by less
    than `tol` (Hartree), and label each group s/p/d/f/... by its size
    (2l+1). Falls back to the raw E+PT2 if no extrapolated value exists."""
    ordered = sorted(states)
    energies = [extrap.get(st, (None, states[st][1]))[1] for st in ordered]

    groups = []
    cur = [ordered[0]]
    for st, e, prev_e in zip(ordered[1:], energies[1:], energies[:-1]):
        if abs(e - prev_e) < tol:
            cur.append(st)
        else:
            groups.append(cur)
            cur = [st]
    groups.append(cur)
    return groups


def print_table(ordered, states, extrap):
    print(f"{'État':>4} {'E (raw)':>16} {'E+PT2 (raw)':>16} {'E extrapolée':>16}")
    for st in ordered:
        e, ept2 = states[st]
        _, eextrap = extrap.get(st, (None, ept2))
        print(f"{st:>4} {e:>16.8f} {ept2:>16.8f} {eextrap:>16.8f}")


def print_degeneracy_table(groups, states, extrap):
    print(f"\n{'États':>12} {'E moy. (Ha)':>14} {'Dégén.':>7} {'Caractère':>10}")
    for g in groups:
        es = [extrap.get(st, (None, states[st][1]))[1] for st in g]
        emean = sum(es) / len(es)
        n = len(g)
        label = L_LABELS.get(n, f'?({n})')
        st_str = f"{g[0]}" if n == 1 else f"{g[0]}-{g[-1]}"
        print(f"{st_str:>12} {emean:>14.6f} {n:>7} {label:>10}")


def write_csv(path, ordered, states, extrap, groups):
    state_to_label = {}
    for g in groups:
        label = L_LABELS.get(len(g), f'?({len(g)})')
        for st in g:
            state_to_label[st] = label

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "E_raw", "E_plus_PT2_raw", "E_extrapolated", "character"])
        for st in ordered:
            e, ept2 = states[st]
            _, eextrap = extrap.get(st, (None, ept2))
            w.writerow([st, f"{e:.10f}", f"{ept2:.10f}", f"{eextrap:.10f}", state_to_label[st]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outfile", help="QP2 FCI .out file")
    parser.add_argument("--csv", metavar="FILE", help="also write the table to this CSV file")
    parser.add_argument("--ndet", type=int, help="N_det checkpoint to use (default: last one found)")
    parser.add_argument("--tol", type=float, default=1e-3, help="degeneracy tolerance in Hartree (default: 1e-3)")
    args = parser.parse_args()

    with open(args.outfile) as f:
        text = f.read()
    lines = text.split("\n")

    checkpoints = find_checkpoints(text)
    if not checkpoints:
        sys.exit(f"No 'Summary at N_det' checkpoint found in {args.outfile}")
    ndet = args.ndet if args.ndet is not None else checkpoints[-1]

    states = parse_raw_states(text, ndet)
    if not states:
        sys.exit(f"No states parsed at N_det={ndet}")
    extrap = parse_extrapolated(lines, ndet)

    ordered = sorted(states)
    print(f"# {args.outfile} -- N_det = {ndet} -- {len(ordered)} states\n")
    print_table(ordered, states, extrap)

    groups = group_by_degeneracy(states, extrap, args.tol)
    print_degeneracy_table(groups, states, extrap)

    if args.csv:
        write_csv(args.csv, ordered, states, extrap, groups)
        print(f"\nWritten: {args.csv}")


if __name__ == "__main__":
    main()
