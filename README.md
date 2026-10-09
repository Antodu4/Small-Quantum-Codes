# Small Quantum Chemistry Utilities

A small collection of Python scripts for quantum chemistry workflows, focused on basis set construction, wavefunction analysis and complex-scaling resonance analysis.

---------------------------------------------------------------------------------------------------------------------------------------

## Scripts

### `basis-plus.py` — Basis set builder (extend or create)

Interactive script with two modes:

- **extend**: Downloads a basis set from the [Basis Set Exchange](https://www.basissetexchange.org/) (or loads a local file) and
appends new uncontracted Gaussian primitives to one or more orbital shells (S, P, D, F, G, H). Exponents are generated as a geometric
series α₀, α₀/k, α₀/k², …. If the provided α₀ overlaps with the existing smallest exponent, it is automatically shifted down by one
step.
- **create**: Generates a basis set from scratch (ex nihilo) for a given element using the same geometric series.

Supports output formats for **Gaussian**, **ORCA**, **Q-Chem**, **Psi4**, and **GAMESS**.

```
python3 basis-plus.py
→ prompts for mode, element, code, orbitals, n, α₀, k
```


### `basis_extender.py` — Basis set extender (extension only)

A streamlined version of `basis-plus.py` restricted to extension mode. Suitable for scripting when ex-nihilo creation is not needed.

```
python3 basis_extender.py
→ prompts for basis name, element, code, orbitals, n, α₀, k
```


### `extract_dominant_dets.py` — Dominant determinant extractor for QP2

Reads an [EZFIO](https://github.com/TREX-CoE/ezfio) database produced by [Quantum Package 2](https://github.com/QuantumPackage/qp2)
and extracts the N determinants with the largest |coefficient| for a target electronic state. Outputs:

- A ranked summary table (coefficient, weight |c|², cumulative weight, occupied MOs).
- A `qp_edit`-compatible RST block ready to be pasted into a fresh EZFIO to seed a state-following FCI or CIPSI calculation.

```
python3 extract_dominant_dets.py <ezfio> <state> [-n N] [--renorm] [--exclusive] [--all-states]
```

| Flag | Effect |
|---|---|
| `-n N` | Number of determinants to extract (default: 10) |
| `--renorm` | Renormalise coefficients within the selected set |
| `--exclusive` | Keep only determinants where the target state has the strictly largest \|c\| across all states |
| `--all-states` | Include coefficients for every state in the RST output |
| `--ezfio-path PATH` | Override auto-detection of the `ezfio.py` module |


### `extract_fci_energies.py` — Per-state energy table and degeneracy grouping from a QP2 FCI log

Parses a QP2 `fci` `.out` log and prints, for the last (or a chosen) `N_det` checkpoint, the raw / PT2-corrected / extrapolated
energy of every state. States whose extrapolated energies agree to within a tolerance are grouped as a degenerate manifold and
labelled by orbital angular-momentum character (s/p/d/f/... from the `2l+1` group size) — handy for spotting which state
indices form the p-type (or d-type, ...) block you're after in a dense multi-state spectrum.

```
python3 extract_fci_energies.py Be.nat.fci.out [--ndet N] [--tol TOL] [--csv FILE]
```

| Flag | Effect |
|---|---|
| `--ndet N` | Use the checkpoint at this `N_det` instead of the last one found |
| `--tol TOL` | Degeneracy tolerance in Hartree for grouping states (default: `1e-3`) |
| `--csv FILE` | Also write the per-state table (with assigned character) to this CSV file |


### `classify_mo_character.py` — Dominant AO angular-momentum character of each MO

Classifies every MO by its dominant AO angular-momentum character (s/p/d/f/...), from the weight of its coefficients on each
AO shell type. Reads either a Molden (`.mol`) file or a QP2 EZFIO directory directly (`ao_basis/ao_power` +
`mo_basis/mo_coef`, no Molden export needed) — the input type is auto-detected (directory → EZFIO, file → Molden).

```
python3 classify_mo_character.py be.24-26.nat.mol [--csv FILE]
python3 classify_mo_character.py be.nat.14s11p [--out be.nat.14s11p.out] [--csv FILE]
```

| Flag | Effect |
|---|---|
| `--out FILE` | EZFIO input only: read natural-orbital occupations from a QP2 `.out` log's `Eigenvalues` table (the EZFIO's own `mo_occ` can be stale after `save_natorb`) |
| `--csv FILE` | Also write the table to this CSV file |


### `set_density_weights.py` — Concentrate state-average weights on a state subset

Sets `state_average_weight` on an EZFIO so that all the density-average weight sits on a chosen subset of states and zero
elsewhere, so a subsequent `save_natorb` (or `save_natorb_no_ref`) builds natural orbitals from the density of *those* states
only — the usual first step before targeting a specific (e.g. resonance) state or block of states with `extract_states.py`.

```
python3 set_density_weights.py EZFIO SELECTION [--dry-run]
```

| Flag | Effect |
|---|---|
| `--dry-run` | Print diagnostics (computed weights) without writing |

`SELECTION` is 1-based, in the *current* wave function's state indexing, e.g. `3-5`. The EZFIO is modified **in place** —
work on a copy. Requires the QP environment to be sourced (`quantum_package.rc`) so that the `ezfio` Python module is
importable.


### `extract_states.py` — State extractor for QP2

Reduces a multi-state [EZFIO](https://github.com/TREX-CoE/ezfio) database to a chosen subset of states (`n_states` → K), keeping
the full determinant space but only the CI coefficients of the selected state(s) (renormalised by default). Works for a single
target state (`K` = 1) as well as a block of states (e.g. a resonance plus its neighbouring pseudo-continuum states), so a
(complex) Davidson run can follow the selected root(s) with `state_following` and avoid the root-drifting that occurs when
starting from only a few dominant determinants.

The EZFIO is modified **in place** — work on a copy. Also sets `state_average_weight` to `[1/K] * K` and `read_wf` to `True`;
`psi_det` and `n_det` are left untouched.

```
cp -r multi_state.ezfio target.ezfio
python3 extract_states.py target.ezfio <states> [--no-normalize] [--dry-run]
```

| Flag | Effect |
|---|---|
| `--no-normalize` | Keep the raw coefficients (no renormalisation) |
| `--dry-run` | Print diagnostics (norm, dominant determinant) without writing |

`<states>` is 1-based, matching `Energy of state N` in QP output, e.g. `30` for a single state or `42-50` / `42,47-49` for a
block. Requires the QP environment to be sourced (`quantum_package.rc`) so that the `ezfio` Python module is importable.


### `extend_n_states.py` — Grow `n_states` on an already-converted EZFIO

In QP2, `qp set determinants n_states N` never resizes `psi_coef` on disk: the provider allocates a `(N_det, N)` buffer with
whatever `N` was just set and reads the array file as-is, with no padding. If the file still holds fewer states than that
(e.g. after a single-state FCI/CIPSI run), the next `qp run fci` with `read_wf=True` crashes with `EZFIO Error in :
ezfio_read_array_do — Dimensions of data ... different from array`.

This script reads the *actual* shape of `psi_coef` straight from the array file (independent of the possibly-stale
`n_states` parameter), keeps the states already present untouched, and pads the new ones with QP2's own default guess
(state `k` → determinant `k` = 1.0). `state_average_weight` is set uniform over the new state count.

```
cp -r single_state.ezfio target.ezfio
python3 extend_n_states.py target.ezfio <n_states> [--dry-run]
```

| Flag | Effect |
|---|---|
| `--dry-run` | Print diagnostics (states already present, padding to add) without writing |

The EZFIO is modified **in place** — work on a copy. Requires the QP environment to be sourced (`quantum_package.rc`) so
that the `ezfio` Python module is importable.


### `cs_resonance.py` — Complex-scaling trajectories, velocity and resonance search from a QP2 CS log

Analyses the output of a complex-scaling (CS) θ-scan run with QP2. The log contains one θ-scan per convergence step (CIPSI
iteration); each scan is read from its `CS energies` table and kept in memory, so that steps can be compared. For every step the
script:

1. extracts the complex energies E(state, θ);
2. rebuilds the **trajectories** E_n(θ) by continuity (QP2 sorts the states by Re E at each θ, so a state index is *not* a
   trajectory as soon as levels cross): linear extrapolation + optimal assignment between successive θ. A trajectory is numbered
   by its state index at θ = 0, and degenerate trajectories are grouped on the plots;
3. computes the **velocity** v(θ) = |dE/dθ| (or |θ dE/dθ| with `--logderiv`) from a cubic spline and lists its local minima —
   a clear velocity minimum (stationary point of the trajectory) signals a resonance (Moiseyev criterion), with
   E_res = E(θ_min) and Γ = −2 Im E_res (also given in eV);
4. prints a **convergence report** between steps (number of determinants, max |ΔE| over all θ and trajectories).

```
python3 cs_resonance.py He_cs.out [--step N|last|all] [--states LIST] [--erange EMIN EMAX]
                                  [--export-energies] [--rotation [--compare]] [--plot] [...]
python3 cs_resonance.py -h        # full help with examples
```

**Selection of steps and states**

| Flag | Effect |
|---|---|
| `--step N\|last\|all` | Convergence step(s) to process: a number, `last` (converged step) or `all` (default) |
| `--states LIST` | Trajectories to export/plot, e.g. `1,3,15-17` (default: all) |
| `--erange EMIN EMAX` | Keep only states with Re E(θ=0) in [EMIN, EMAX] (Ha); combines with `--states` |

**Resonance search (velocity)** — all filters are **disabled by default**, so every local minimum of the velocity is listed;
enable the filters below to discard noise on flat (pseudo-continuum) curves.

| Flag | Effect |
|---|---|
| `--threshold E` | Keep only minima with Re E > E (e.g. `-2.0` for the He⁺ 1s ionisation threshold) |
| `--theta-min TH` | Ignore minima at θ < TH (rad) |
| `--prominence P` | Minimum depth of a velocity minimum, as a fraction of the trajectory's median velocity (e.g. `0.2`) |
| `--imag-tol G` | Keep only minima with Im E < −G (states with a finite width), e.g. `1e-4` |
| `--logderiv` | Velocity = \|θ dE/dθ\| instead of \|dE/dθ\| |
| `--diag` | Print the minimum velocity of every trajectory (no filter except `--theta-min`) |

**Data export**

| Flag | Effect |
|---|---|
| `--export-energies` | Write `stepN_energies.dat`: Re/Im E (Ha) of each selected state for every θ of the chosen step |
| `--order tracked\|sorted` | Export columns = continuity-tracked trajectories (default) or states in raw QP2 order |

**Plots**

| Flag | Effect |
|---|---|
| `--rotation` | Rotation of the states in the complex plane (■ θ=0, ● every `--theta-marks` rad, ▶ θ max) |
| `--compare` | With `--rotation`: overlay the selected steps on a single figure |
| `--plot` | Two-panel figure: complex-plane trajectories \| velocity vs θ (detected minima as red stars) |
| `--theta-marks STEP` | Spacing (rad) of the θ markers on the rotation plot (default: `0.1`) |
| `--xlim XMIN XMAX`, `--ylim YMIN YMAX` | Axis limits of the rotation plot |
| `--no-labels` | No state labels on the rotation plot |
| `--linear-v` | Linear velocity axis in `--plot` (default: logarithmic) |
| `--show` | Also open the matplotlib window (needs a display) |
| `--no-plot` | Disable the default separate figures (complex plane / velocity) |
| `--outdir DIR` | Output directory (default: `cs_results`) |

**Output files** (in `--outdir`): `stepN_trajectories.csv` (trajectories + velocity, long format), `stepN_energies.dat`,
`stepN_rotation.png`, `steps_1-2-3_rotation.png` (`--compare`), `stepN_trajectories_velocity.png` (`--plot`).

Examples:

```
python3 cs_resonance.py He_cs.out --step last --diag
python3 cs_resonance.py He_cs.out --step 3 --export-energies
python3 cs_resonance.py He_cs.out --step 3 --rotation --states 1,2,7-20
python3 cs_resonance.py He_cs.out --rotation --compare --erange -2.1 -0.9
python3 cs_resonance.py He_cs.out --step last --plot --threshold -2.0 --prominence 0.2 --imag-tol 1e-4
```

Pure text parser: no QP environment needed.

---------------------------------------------------------------------------------------------------------------------------------------

## Requirements

```
pip install requests numpy scipy matplotlib
```

The `extract_dominant_dets.py`, `set_density_weights.py`, `extract_states.py`, and `extend_n_states.py` scripts also require
the EZFIO Python module, which is bundled with QP2. For `extract_dominant_dets.py` it is auto-detected from common install
locations or via the `QP_ROOT` environment variable; for the others, source `quantum_package.rc` before running.
`extract_fci_energies.py`, `classify_mo_character.py` and `cs_resonance.py` are pure text/EZFIO-array parsers and need no QP
environment (`cs_resonance.py` additionally needs `scipy` and `matplotlib`).

---------------------------------------------------------------------------------------------------------------------------------------

## License

See [LICENSE](LICENSE).
