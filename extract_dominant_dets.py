#!/usr/bin/env python3
"""
Extract the N most important determinants for a given state from an EZFIO
database and print them in the QP2 qp_edit-compatible RST format.

Usage:
    python3 extract_dominant_dets.py <ezfio> <state> [-n N] [--renorm] [--ezfio-path PATH]

Output:
    - Summary table of the top N determinants
    - QP2 RST block ready to be pasted into qp_edit for state-following

The RST block can be pasted directly into the "Determinants ::" section when
running qp_edit on a fresh EZFIO to seed a state-following calculation.
"""

import sys
import os
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# EZFIO discovery
# ---------------------------------------------------------------------------

def find_ezfio_module():
    """Return the directory containing ezfio.py, or None."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "qp2_old/external/ezfio/Python"),
        os.path.join(os.path.dirname(__file__), "qp2_cs/external/ezfio/Python"),
        os.path.expanduser("~/qp2/external/ezfio/Python"),
        os.path.expanduser("~/quantum_package/external/ezfio/Python"),
    ]
    qp_root = os.environ.get("QP_ROOT", "")
    if qp_root:
        candidates.insert(0, os.path.join(qp_root, "external/ezfio/Python"))
    for p in candidates:
        if os.path.isfile(os.path.join(p, "ezfio.py")):
            return p
    return None


# ---------------------------------------------------------------------------
# Bitstring helpers  (matches QP2 Bitlist.ml convention)
# ---------------------------------------------------------------------------

def int64_to_bitstring(val: int) -> str:
    """
    Convert one 64-bit signed integer to a 64-char string of '+'/'-'.
    Bit i (0 = LSB = MO 1) maps to position i in the string.
    """
    # Python integers are arbitrary precision; mask to 64 bits then iterate
    val = val & 0xFFFFFFFFFFFFFFFF
    return "".join("+" if (val >> i) & 1 else "-" for i in range(64))


def det_row_to_bitstrings(alpha_ints, beta_ints):
    """
    alpha_ints / beta_ints : array-like of n_int int64 values.
    Returns (alpha_str, beta_str), each of length n_int*64.
    """
    alpha_str = "".join(int64_to_bitstring(int(v)) for v in alpha_ints)
    beta_str  = "".join(int64_to_bitstring(int(v)) for v in beta_ints)
    return alpha_str, beta_str


def bitstring_to_occ(s: str) -> list[int]:
    """Return 1-based list of occupied MO indices from a '+'/'-' string."""
    return [i + 1 for i, c in enumerate(s) if c == "+"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract top-N determinants for a given state from EZFIO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ezfio",  help="Path to the EZFIO database")
    parser.add_argument("state",  type=int, help="Target state (1-based)")
    parser.add_argument("-n", "--ndet", type=int, default=10,
                        help="Number of determinants to extract (default: 10)")
    parser.add_argument("--renorm", action="store_true",
                        help="Renormalise coefficients within the selected N dets")
    parser.add_argument("--all-states", action="store_true",
                        help="Include coefficients for all states in the RST block "
                             "(default: target state only, n_states=1)")
    parser.add_argument("--exclusive", action="store_true",
                        help="Only keep determinants for which the target state has "
                             "the strictly largest |coefficient| across all states "
                             "(excludes shared/non-dominant determinants)")
    parser.add_argument("--ezfio-path", metavar="PATH",
                        help="Path to the ezfio Python module (auto-detected otherwise)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load ezfio
    # ------------------------------------------------------------------
    ezfio_path = args.ezfio_path or find_ezfio_module()
    if ezfio_path is None:
        sys.exit("ERROR: ezfio.py not found. Use --ezfio-path to specify its directory.")
    sys.path.insert(0, ezfio_path)
    import ezfio as ezfio_mod

    e = ezfio_mod.ezfio_obj()
    e.set_file(os.path.abspath(args.ezfio))

    # ------------------------------------------------------------------
    # Read metadata
    # ------------------------------------------------------------------
    n_det    = e.get_determinants_n_det()
    n_states = e.get_determinants_n_states()
    n_int    = e.get_determinants_n_int()
    n_mo     = e.get_mo_basis_mo_num()
    n_alpha  = e.get_electrons_elec_alpha_num()
    n_beta   = e.get_electrons_elec_beta_num()

    if args.state < 1 or args.state > n_states:
        sys.exit(f"ERROR: state {args.state} is out of range [1, {n_states}].")

    n_extract = min(args.ndet, n_det)
    istate    = args.state - 1  # 0-based

    print(f"EZFIO    : {args.ezfio}")
    print(f"N_det    : {n_det}   N_states : {n_states}   N_int : {n_int}")
    print(f"N_mo     : {n_mo}   N_alpha  : {n_alpha}   N_beta : {n_beta}")
    mode = "exclusive" if args.exclusive else "top-N"
    print(f"Extracting up to {n_extract} determinants for state {args.state} [{mode}]")
    print()

    # ------------------------------------------------------------------
    # Read wavefuction arrays
    # coef shape : (n_states, n_det)   — states vary fastest in memory
    # det  shape : (n_det, 2, n_int)
    # ------------------------------------------------------------------
    coef = np.array(e.get_determinants_psi_coef())  # (n_states, n_det)
    det  = np.array(e.get_determinants_psi_det())   # (n_det, 2, n_int)

    state_coef = coef[istate, :]                    # (n_det,)

    # Rank all dets by |coef| for the target state (descending)
    sorted_all = np.argsort(np.abs(state_coef))[::-1]

    # ------------------------------------------------------------------
    # Exclusive filter (optional)
    # Keep only dets where the target state strictly dominates all others.
    # ------------------------------------------------------------------
    if args.exclusive and n_states > 1:
        other_states = [s for s in range(n_states) if s != istate]
        max_other = np.max(np.abs(coef[other_states, :]), axis=0)  # (n_det,)
        is_dominant = np.abs(state_coef) > max_other               # (n_det,) bool
        n_dominant  = int(np.sum(is_dominant))
        top_idx = np.array([int(i) for i in sorted_all if is_dominant[i]][:n_extract])
        print(f"Exclusive mode: {n_dominant} / {n_det} determinants are strictly "
              f"dominant for state {args.state}.")
        if len(top_idx) < n_extract:
            print(f"  Only {len(top_idx)} dominant determinants found "
                  f"(requested {n_extract}).")
        print()
    else:
        top_idx = sorted_all[:n_extract]

    # ------------------------------------------------------------------
    # Renormalise (optional)
    # ------------------------------------------------------------------
    selected_coef = state_coef[top_idx].copy()
    if args.renorm:
        norm = np.sqrt(np.sum(selected_coef**2))
        if norm > 0:
            selected_coef /= norm

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"{'Rank':>4}  {'Det #':>7}  {'Coefficient':>15}  {'|c|²':>8}  "
          f"{'Cumul.':>8}  Alpha occ.   Beta occ.")
    print("-" * 90)
    cumul = 0.0
    for rank, idx in enumerate(top_idx):
        c  = selected_coef[rank]
        w  = c ** 2
        cumul += state_coef[top_idx[rank]] ** 2  # cumul uses original weights
        a_str, b_str = det_row_to_bitstrings(det[idx, 0, :], det[idx, 1, :])
        a_occ = bitstring_to_occ(a_str[:n_mo])
        b_occ = bitstring_to_occ(b_str[:n_mo])
        print(f"{rank+1:4d}  {idx+1:7d}  {c:+15.8f}  {w:8.5f}  {cumul:8.5f}  "
              f"{str(a_occ):<18} {b_occ}")

    print()
    print(f"Cumulative weight of top {n_extract} determinants : "
          f"{np.sum(state_coef[top_idx]**2):.4f}")
    if args.renorm:
        print("(Coefficients have been renormalised within the selected set)")

    # ------------------------------------------------------------------
    # QP2 qp_edit RST block
    # ------------------------------------------------------------------
    # Format (from Input_determinants_by_hand.ml / to_rst):
    #
    #   Force the selected wave function ...
    #   expected_s2 = <val>
    #
    #   Number of determinants ::
    #     n_det = <N>
    #
    #   State average weights ::
    #     state_average_weight = (1.0)
    #
    #   Determinants ::
    #
    #     <coef_state1>\t<coef_state2>...
    #       <alpha bitstring>
    #       <beta bitstring>
    #
    #     ...

    if args.all_states:
        out_states  = list(range(n_states))
        out_nstates = n_states
        saw_str     = "\t".join(["1.0"] * n_states)
    else:
        out_states  = [istate]
        out_nstates = 1
        saw_str     = "1.0"

    sep  = "=" * 70
    print()
    print(sep)
    print("QP2 qp_edit RST BLOCK  (paste into the Determinants section)")
    print(sep)
    print()

    # Expected S² (singlet = 0, doublet = 0.75, …)
    s = 0.5 * abs(n_alpha - n_beta)
    expected_s2 = s * (s + 1.0)

    print(f"Force the selected wave function to be an eigenfunction of S^2.")
    print(f"If true, input the expected value of S^2 ::")
    print()
    print(f"  expected_s2 = {expected_s2:.1f}")
    print()
    print(f"Number of determinants ::")
    print()
    print(f"  n_det = {n_extract}")
    print()
    print(f"State average weights ::")
    print()
    print(f"  state_average_weight = ({saw_str})")
    print()
    print("Determinants ::")

    for rank, idx in enumerate(top_idx):
        c  = selected_coef[rank]

        # Build coefficient string (one per output state, tab-separated)
        if args.all_states:
            coefs = [coef[s_idx, idx] for s_idx in out_states]
            if args.renorm:
                # Renorm only the target state; keep others as-is
                pass
            coef_line = "\t".join(f"{v:.12f}" for v in coefs)
        else:
            coef_line = f"{c:.12f}"

        a_str, b_str = det_row_to_bitstrings(det[idx, 0, :], det[idx, 1, :])

        # Truncate to n_mo significant bits (pad or trim to match mo_num)
        # QP2 of_string reads the whole string; extra '-' beyond n_mo are fine
        # but we mirror the exact QP2 convention: n_int*64 chars.
        print()
        print(f"  {coef_line}")
        print(f"  {a_str}")
        print(f"  {b_str}")

    print()
    print(sep)
    print(f"Copy the block above (from 'Force...' to the last beta string).")
    print(f"Then run:  qp_edit <new_ezfio>  and replace the Determinants")
    print(f"section with it. Set read_wf = true before running the FCI.")
    print(sep)


if __name__ == "__main__":
    main()
