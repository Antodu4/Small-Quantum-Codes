  # Small Quantum Chemistry Utilities

  A small collection of Python scripts for quantum chemistry workflows, focused on basis set construction and wavefunction analysis.

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

  python3 basis-plus.py
  → prompts for mode, element, code, orbitals, n, α₀, k


  ### `basis_extender.py` — Basis set extender (extension only)

  A streamlined version of `basis-plus.py` restricted to extension mode. Suitable for scripting when ex-nihilo creation is not needed.

  python3 basis_extender.py
  → prompts for basis name, element, code, orbitals, n, α₀, k


  ### `extract_dominant_dets.py` — Dominant determinant extractor for QP2

  Reads an [EZFIO](https://github.com/TREX-CoE/ezfio) database produced by [Quantum Package 2](https://github.com/QuantumPackage/qp2)
  and extracts the N determinants with the largest |coefficient| for a target electronic state. Outputs:

  - A ranked summary table (coefficient, weight |c|², cumulative weight, occupied MOs).
  - A `qp_edit`-compatible RST block ready to be pasted into a fresh EZFIO to seed a state-following FCI or CIPSI calculation.

  python3 extract_dominant_dets.py <ezfio> <state> [-n N] [--renorm] [--exclusive] [--all-states]

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

  python3 extract_fci_energies.py Be.nat.fci.out [--ndet N] [--tol TOL] [--csv FILE]

  | Flag | Effect |
  |---|---|
  | `--ndet N` | Use the checkpoint at this `N_det` instead of the last one found |
  | `--tol TOL` | Degeneracy tolerance in Hartree for grouping states (default: `1e-3`) |
  | `--csv FILE` | Also write the per-state table (with assigned character) to this CSV file |


  ### `classify_mo_character.py` — Dominant AO angular-momentum character of each MO

  Classifies every MO by its dominant AO angular-momentum character (s/p/d/f/...), from the weight of its coefficients on each
  AO shell type. Reads either a Molden (`.mol`) file or a QP2 EZFIO directory directly (`ao_basis/ao_power` +
  `mo_basis/mo_coef`, no Molden export needed) — the input type is auto-detected (directory → EZFIO, file → Molden).

  python3 classify_mo_character.py be.24-26.nat.mol [--csv FILE]
  python3 classify_mo_character.py be.nat.14s11p [--out be.nat.14s11p.out] [--csv FILE]

  | Flag | Effect |
  |---|---|
  | `--out FILE` | EZFIO input only: read natural-orbital occupations from a QP2 `.out` log's `Eigenvalues` table (the EZFIO's own `mo_occ` can be stale after `save_natorb`) |
  | `--csv FILE` | Also write the table to this CSV file |


  ### `set_density_weights.py` — Concentrate state-average weights on a state subset

  Sets `state_average_weight` on an EZFIO so that all the density-average weight sits on a chosen subset of states and zero
  elsewhere, so a subsequent `save_natorb` (or `save_natorb_no_ref`) builds natural orbitals from the density of *those* states
  only — the usual first step before targeting a specific (e.g. resonance) state or block of states with `extract_states.py`.

  python3 set_density_weights.py EZFIO SELECTION [--dry-run]

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

  cp -r multi_state.ezfio target.ezfio
  python3 extract_states.py target.ezfio <states> [--no-normalize] [--dry-run]

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

  cp -r single_state.ezfio target.ezfio
  python3 extend_n_states.py target.ezfio <n_states> [--dry-run]

  | Flag | Effect |
  |---|---|
  | `--dry-run` | Print diagnostics (states already present, padding to add) without writing |

  The EZFIO is modified **in place** — work on a copy. Requires the QP environment to be sourced (`quantum_package.rc`) so
  that the `ezfio` Python module is importable.

---------------------------------------------------------------------------------------------------------------------------------------  

  ## Requirements

  pip install requests numpy

  The `extract_dominant_dets.py`, `set_density_weights.py`, `extract_states.py`, and `extend_n_states.py` scripts also require
  the EZFIO Python module, which is bundled with QP2. For `extract_dominant_dets.py` it is auto-detected from common install
  locations or via the `QP_ROOT` environment variable; for the others, source `quantum_package.rc` before running.
  `extract_fci_energies.py` and `classify_mo_character.py` are pure text/EZFIO-array parsers and need no QP environment.

---------------------------------------------------------------------------------------------------------------------------------------

  ## License

  See [LICENSE](LICENSE).

