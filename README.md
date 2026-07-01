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

---------------------------------------------------------------------------------------------------------------------------------------  

  ## Requirements

  pip install requests numpy

  The `extract_dominant_dets.py` script also requires the EZFIO Python module, which is bundled with QP2. It is auto-detected from
  common install locations or via the `QP_ROOT` environment variable.

---------------------------------------------------------------------------------------------------------------------------------------

  ## License

  See [LICENSE](LICENSE).

