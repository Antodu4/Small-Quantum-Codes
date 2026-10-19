# Small Quantum Chemistry Utilities

A small collection of Python scripts for quantum chemistry workflows, focused on basis set construction, wavefunction
analysis with [Quantum Package 2](https://github.com/QuantumPackage/qp2) (QP2) and complex-scaling resonance analysis.

```
.
├── basis-plus.py                   # basis set builder (extend or create)
├── basis_extender.py               # basis set extender (extension only)
└── Quantum_Package/
    ├── qp_fci_analysis.py          # FCI/CIPSI convergence, energies, state character
    ├── cs_analysis.py              # complex-scaling trajectories and resonance search
    ├── extract_dominant_dets.py    # dominant determinants of a state → qp_edit RST block
    ├── classify_mo_character.py    # s/p/d/f... character of each MO
    └── set_density_weights.py      # state-average weights on a subset of states
```

---------------------------------------------------------------------------------------------------------------------------------------

## Basis set scripts

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

---------------------------------------------------------------------------------------------------------------------------------------

## Quantum Package scripts (`Quantum_Package/`)

### `qp_fci_analysis.py` — FCI/CIPSI convergence analysis from a QP2 log

Parses a QP2 FCI/CIPSI `.out` log and extracts, for every wave-function size (`N_det`) and every state: variational energy,
⟨S²⟩, variance, PT2, rPT2, E+PT2 and E+rPT2, as well as the FCI extrapolated energies computed by QP (last
`Extrapolated energies` block). All values are read as-is from the log; nothing is recomputed.

```
python3 qp_fci_analysis.py Be_fci.out [--energies] [--table] [--character] [--plot] [-s STATES ...] [...]
python3 qp_fci_analysis.py -h        # full help with examples
```

Without any action flag, `--energies` is assumed.

**Actions**

| Flag | Effect |
|---|---|
| `-e`, `--energies` | Print the converged (last `N_det`) and extrapolated energies of each state |
| `-t`, `--table` | Print the convergence table vs `N_det` |
| `-c`, `--character` | Group states that are degenerate (within `--degeneracy-tol`) and label them s/p/d/f/... from the `2l+1` group size — handy for atoms |
| `-p`, `--plot` | Plot the convergence of the selected states vs `N_det` |

**Options**

| Flag | Effect |
|---|---|
| `-s`, `--states` | States to process: list (`1 3 4`), range (`1-5`) or `all` (default: `all`, or `1` with `--plot` alone) |
| `--degeneracy-tol HA` | Degeneracy tolerance in Hartree for `--character` (default: `1e-3`) |
| `-q`, `--quantity` | Plotted quantity/quantities: `evar` (E), `ept2` (E+PT2), `erpt2` (E+rPT2) (default: `evar`) |
| `-m`, `--mode` | `absolute`: E(N_det); `relative`: E − E_extrap in mHa (log scale); `excitation`: excitation energies (eV) from QP's Summary tables (`evar`, `ept2` only) |
| `--ndet-min N`, `--ndet-max N` | `N_det` range shown on the plot and table |
| `--logy` / `--liny`, `--linx` | Axis scales (y is log by default in `relative` mode; `N_det` axis is log by default) |
| `--extrap-row N` | Row of QP's extrapolation table to use (default: `1`, smallest PT2 threshold) |
| `-d`, `--outdir DIR` | Output directory (default: `<file>_analysis`) |
| `-o`, `--output FILE` | Plot file name (default: `<file>_convergence_<mode>.png`) |
| `--title`, `--show` | Plot title; open the matplotlib window |

**Output files** (in `--outdir`): `<file>_convergence.csv` (all states and steps), `<file>_extrapolation.csv`,
`<file>_character.csv` (`--character`), `<file>_convergence_<mode>.png` (`--plot`).

Examples:

```
python3 qp_fci_analysis.py Be_fci_20.out --character --degeneracy-tol 5e-4
python3 qp_fci_analysis.py Be_fci_20.out --table --states 1
python3 qp_fci_analysis.py Be_fci_20.out --plot --states 1-5 --mode relative --quantity evar ept2 -o conv_be.png
python3 qp_fci_analysis.py Be_fci_20.out --plot --states 2-6 --mode excitation --show
```


### `cs_analysis.py` — Complex-scaling trajectories, velocity and resonance search from a QP2 CS log

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
python3 cs_analysis.py He_cs.out [--step N|last|all] [--states LIST] [--erange EMIN EMAX]
                                 [--export-energies] [--rotation [--compare]] [--plot [--zoom]] [...]
python3 cs_analysis.py -h        # full help with examples
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
| `--scan-analysis` | Analyse only the trajectory carrying the best resonance left after the filters above |

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
| `--plot` | Two-panel figure: complex-plane trajectories \| velocity vs θ (detected minima as red points) |
| `--theta-marks STEP` | Spacing (rad) of the θ markers on the rotation plot (default: `0.1`) |
| `--theta-range TH [TH2]` | Show a single θ value (nearest point) or a θ window, e.g. `0.200` or `0.150 0.250` |
| `--no-resonances` | Hide the red resonance markers |
| `--label-resonances` | Annotate each resonance marker with its complex energy and θ |
| `--no-legend` | Disable the legend |
| `--xlim XMIN XMAX`, `--ylim YMIN YMAX` | Axis limits of the complex plane (Re E, Im E) |
| `--no-labels` | No state labels on the rotation plot |
| `--linear-v` | Linear velocity axis in `--plot` (default: logarithmic) |
| `--show` | Also open the matplotlib window (needs a display) |
| `--no-plot` | Disable the default separate figures (complex plane / velocity) |
| `--outdir DIR` | Output directory (default: `cs_results`) |

**Automatic zoom** — inset close-up of a resonance in the complex plane (with `--plot` and/or `--rotation`)

| Flag | Effect |
|---|---|
| `--zoom` | Add the inset around the resonance (best velocity minimum) |
| `--zoom-res LIST` | Resonance(s) to zoom on, by rank in increasing velocity (`1` = best), e.g. `1,3` |
| `--zoom-states LIST` | Only consider resonances on these trajectories, e.g. `34,35` |
| `--zoom-theta TH1 TH2` | Force the θ window defining the framed trajectory (default: automatic) |
| `--zoom-vmax K` | Follow the resonance trajectory while v ≤ K × v_min (default: `20`) |
| `--zoom-margin M` | Margin around the zoomed area, as a fraction of its size (default: `0.15`) |
| `--zoom-xlim`, `--zoom-ylim` | Force the Re E / Im E bounds of the inset (Ha) |
| `--zoom-loc`, `--zoom-size`, `--zoom-position X Y` | Inset placement (`auto` = least crowded corner), size (default: `0.30`) or manual lower-left corner in axis fractions |

**Output files** (in `--outdir`): `stepN_trajectories.csv` (trajectories + velocity, long format), `stepN_complex_plane.png`,
`stepN_velocity.png`, `stepN_energies.dat` (`--export-energies`), `stepN_rotation.png` (`--rotation`),
`steps_1-2-3_rotation.png` (`--compare`), `stepN_trajectories_velocity.png` (`--plot`).

Examples:

```
python3 cs_analysis.py He_cs.out --step last --diag
python3 cs_analysis.py He_cs.out --step 3 --export-energies
python3 cs_analysis.py He_cs.out --step 3 --rotation --states 1,2,7-20
python3 cs_analysis.py He_cs.out --rotation --compare --erange -2.1 -0.9
python3 cs_analysis.py He_cs.out --step last --plot --threshold -2.0 --prominence 0.2 --imag-tol 1e-4
python3 cs_analysis.py He_cs.out --step last --plot --states 7-20 --theta-range 0.150 0.250
python3 cs_analysis.py He_cs.out --step last --plot --zoom --no-legend
```

Pure text parser: no QP environment needed.


### `extract_dominant_dets.py` — Dominant determinant extractor

Reads an [EZFIO](https://github.com/TREX-CoE/ezfio) database produced by QP2 and extracts the N determinants with the largest
|coefficient| for a target electronic state. Outputs:

- A ranked summary table (coefficient, weight |c|², cumulative weight, occupied MOs).
- A `qp_edit`-compatible RST block ready to be pasted into a fresh EZFIO to seed a state-following FCI or CIPSI calculation.

By default the script reads the truncated `psi_coef_qp_edit` / `psi_det_qp_edit` arrays (the `n_det_print_wf` dominant
determinants), which is instantaneous even for large wave functions.

```
python3 extract_dominant_dets.py <ezfio> <state> [-n N] [--full] [--renorm] [--exclusive] [--all-states]
```

| Flag | Effect |
|---|---|
| `-n N` | Number of determinants to extract (default: 10) |
| `--full` | Read the complete `psi_coef` / `psi_det` arrays (needed beyond the dominant set; much slower for large `N_det`) |
| `--renorm` | Renormalise coefficients within the selected set |
| `--exclusive` | Keep only determinants where the target state has the strictly largest \|c\| across all states |
| `--all-states` | Include coefficients for every state in the RST output |
| `--ezfio-path PATH` | Override auto-detection of the `ezfio.py` module |


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
only — the usual first step before targeting a specific (e.g. resonance) state or block of states.

```
python3 set_density_weights.py EZFIO SELECTION [--dry-run]
```

| Flag | Effect |
|---|---|
| `--dry-run` | Print diagnostics (computed weights) without writing |

`SELECTION` is 1-based, in the *current* wave function's state indexing, e.g. `3-5` or `2,4-6`. The EZFIO is modified
**in place** — work on a copy.

---------------------------------------------------------------------------------------------------------------------------------------

## Requirements

```
pip install requests numpy scipy matplotlib
```

- `basis-plus.py`, `basis_extender.py`: `requests` (Basis Set Exchange download).
- `qp_fci_analysis.py`: `numpy`, `matplotlib`. `cs_analysis.py`: `numpy`, `scipy`, `matplotlib`. Both are pure text parsers
  and need no QP environment; neither does `classify_mo_character.py`, which reads the EZFIO arrays directly.
- `extract_dominant_dets.py` and `set_density_weights.py` need the EZFIO Python module bundled with QP2. For
  `extract_dominant_dets.py` it is auto-detected from `QP_ROOT` or common install locations (or given with `--ezfio-path`);
  for `set_density_weights.py`, source `quantum_package.rc` before running.

---------------------------------------------------------------------------------------------------------------------------------------

## License

See [LICENSE](LICENSE).
