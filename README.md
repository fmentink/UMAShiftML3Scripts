# UMAShiftML3Scripts
Scripts used for Accessible hybrid DFT-quality NMR crystallography via gas-phase Machine Learning Interatomic Potentials https://doi.org/10.1039/D6SC04941A

# UMA Crystal Structure Optimization 

A single-GPU version of the UMA crystal relaxation workflow. It relaxes crystal structures from CIF files using Meta's UMA machine-learning interatomic potentials (MLIPs) via FAIRChem and ASE: CIF in, relaxed CIF/XYZ out. Batch (directory) mode is supported.

## Features

- Single file or directory batch processing (with optional `--recursive` subdirectory search)
- Optimizers: LBFGS, FIRE, BFGS, GPMin (plus optional Sella)
- Optional unit-cell relaxation via `FrechetCellFilter` (only when `--cell-opt` is passed)
- `FixSymmetry` constraint applied by default (`--no-symmetry` to disable)
- Initial-force gate to skip bad starting structures (`--max-initial-force`)
- Space-group analysis before/after and volume-change reporting
- Always saves a trajectory, plus CIF, XYZ and a text summary

## Requirements

- **Hardware:** NVIDIA GPU with CUDA recommended for speed. On macOS / Apple Silicon or any CPU-only machine the script automatically falls back to CPU — expect significantly slower runs, especially for larger cells and the `uma-m` models.
- Python 3 virtual environment (see Installation).

## Installation

This repository does **not** create or manage your Python environment; you must create one yourself.

**Option A — conda:**

```bash
conda create -n uma python=3.12
conda activate uma
pip install -r requirements.txt
```


> **NVIDIA GPU users:** install the CUDA-enabled PyTorch build first (see <https://pytorch.org>) before running `pip install -r requirements.txt`.

**Optional Sella optimizer** (on PyPI since v2.5.0, released 23 Jun 2026):

```bash
pip install 'sella>=2.5.0'
```

## Usage

**Single file:**

```bash
python UMACrystalOptimizeSam.py my_structure.cif --fmax 0.01 --steps 2000
```

**Batch directory:**

```bash
python UMACrystalOptimizeSam.py ./cif_dir --recursive
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input_path` (positional) | — | CIF file or directory of CIF files |
| `--optimizer` | `LBFGS` | Optimizer to use: `{LBFGS,FIRE,BFGS,GPMin,Sella}`. Sella is optional (`pip install 'sella>=2.5.0'`), does not support symmetry constraints (always pass `--no-symmetry` with it), and needs sella ≥ 2.5.0 when combined with `--cell-opt` (cell optimization added in v2.5.0, released 23 Jun 2026). |
| `--cell-opt` | off | When passed, unit-cell parameters are also relaxed (`FrechetCellFilter`); by default only atomic positions are relaxed |
| `--fmax` | `0.01` | Force convergence threshold (eV/Å) |
| `--steps` | `1000` | Maximum optimization steps |
| `--prefix` | `<input>_UMAopt` | Output file prefix |
| `--cuda-device` | `0` | CUDA device index to use (ignored when running on CPU) |
| `--model` | `uma-s-1p1` | UMA model to use: `{uma-s-1p1, uma-s-1p2, uma-m-1p1}` |
| `--task-name` | `omol` | FAIRChem task name: `{omol, omc, omat}` |
| `--recursive` | off | In batch mode also search subdirectories for `.cif` files |
| `--no-symmetry` | off | Disable the `FixSymmetry` constraint (applied by default with `symprec=0.01`; **required when using Sella**) |
| `--max-initial-force` | — | Skip a structure if its initial max force exceeds this value (eV/Å) |

## Outputs

For each input, files are written next to it (or in the current directory):

- `<prefix>.cif` — optimized structure (space group detected from final positions via pymatgen/spglib)
- `<prefix>.xyz`
- `<prefix>.traj` — optimization trajectory (always written)
- `<prefix>_summary.txt` — parameters and results

Batch mode prints a success/failure summary at the end.

## Notes

- NVIDIA GPU with CUDA is required (no CPU fallback).
- **Sella** is optional: `pip install 'sella>=2.5.0'`. With `--cell-opt`, Sella uses its v2.5.0 cell-optimization support (`optimize_cell`). Sella does not support symmetry constraints — always pass `--no-symmetry` when using it (the script enforces this and exits with an error otherwise).
- **Cell relaxation & stress:** when `--cell-opt` is used, the calculator enables stress prediction via FAIRChem's `predict_untrained_stress` inference setting (UMA tasks are not trained with stress labels, so stress is computed by autograd); it stays disabled otherwise to save time. See <https://fair-chem.github.io/ase-calculator/#enabling-gradient-stress-or-hessian-prediction>.
- If FAIRChem/UMA model loading fails, check that your `fairchem-core` provides the pretrained UMA models (`fairchem.core.pretrained_mlip`) and consult <https://github.com/FAIR-Chem/fairchem>.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (see LICENSE).

## Citation

If you use this code or the UMA models in your work, please cite our preprint:

```bibtex
@article{10.1039/D6SC04941A,
    author = {Gunaga, Shubha and Schurko, Rob and Holmes, Sean T. and Mentink-Vigier, Frederic},
    title = {Accessible hybrid DFT-quality NMR crystallography via gas-phase Machine Learning Interatomic Potentials},
    journal = {Chemical Science},
    year = {2026},
    month = {09},
    abstract = { Nuclear magnetic resonance (NMR) crystallography is a robust method for structure determination, but its reliance on density functional theory (DFT) for geometry refinement limits its speed and accessibility. Recent machine‑learning predictors such as ShiftML3 can evaluate magnetic shieldings in seconds, but they still need high‑quality crystal geometries that are normally obtained from slow DFT optimisations. Here we demonstrate that machine learning interatomic potentials (MLIPs) can eliminate this bottleneck for organic crystals. Interestingly, models trained on gas-phase molecules at the hybrid ωB97M-V level (OMol25 dataset) — such as UMA-omol and MACE-Polar-1 — deliver structural quality rivaling hybrid periodic-DFT, without requiring large computational resources. The resulting MLIP–ShiftML3 workflow reduces computational cost by at least three orders of magnitude, making high-accuracy NMR crystallography accessible without HPC infrastructure, and opening the door to fast molecular dynamics simulations at the hybrid DFT level. Combined with sensitivity enhancement via dynamic nuclear polarization (DNP), we demonstrate the approach on L-histidine, extracting ¹³C and ¹⁵N chemical shift tensors and using ¹H chemical shifts at natural isotopic abundance and discriminating between its monoclinic and orthorhombic polymorphs. },
    issn = {2041-6520},
    doi = {10.1039/D6SC04941A},
    url = {https://doi.org/10.1039/D6SC04941A},
    eprint = {https://pubs.rsc.org/sc/article-pdf/doi/10.1039/D6SC04941A/13769183/d6sc04941a.pdf},
```
