# UMA Crystal Structure Optimization (simplified)

Single-GPU version of the UMA crystal relaxation workflow as used in https://doi.org/10.1039/D6SC04941A
Relaxes crystal structures from CIF files using Meta's UMA MLIPs via FAIRChem and ASE. Single-file or batch mode; GPU (CUDA) or CPU.
Can be executed on desktop/laptop. More info can be found here https://fair-chem.github.io/

## Features

- Single file or directory batch processing (with optional `--recursive` subdirectory search)
- Optimizers: LBFGS, FIRE, BFGS, GPMin (optional Sella)
- Optional unit-cell relaxation via `FrechetCellFilter` (only when `--cell-opt` is passed)
- `FixSymmetry` constraint applied by default (`--no-symmetry` to disable)
- Initial-force gate to skip bad starting structures (`--max-initial-force`)
- Space-group analysis before/after and volume-change reporting
- Always saves a trajectory, plus CIF, XYZ and a text summary
- GPU (CUDA) or CPU execution (`--device cpu`)

## Requirements

- **Hardware:** NVIDIA GPU with CUDA recommended for speed. On macOS / Apple Silicon or any CPU-only machine the script automatically falls back to CPU — expect significantly slower runs, especially for larger cells and the `uma-m` models. Force CPU explicitly with `--device cpu`.
- Python 3 virtual environment (see Installation).

## Installation

This repository does **not** create or manage your Python environment; you must create one yourself.

**Option A — conda:**

```bash
conda create -n uma python=3.12
conda activate uma
pip install -r requirements.txt
```

**Option B — venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
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
python UMACrystalOptimize.py my_structure.cif --fmax 0.01 --steps 2000
```

**With cell relaxation:**

```bash
python UMACrystalOptimize.py my_structure.cif --cell-opt --fmax 0.01
```

**CPU only (e.g. macOS or when GPU is busy):**

```bash
python UMACrystalOptimize.py my_structure.cif --device cpu
```

**Batch directory:**

```bash
python UMACrystalOptimize.py ./cif_dir --recursive
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input_path` (positional) | — | CIF file or directory of CIF files |
| `--optimizer` | `LBFGS` | Optimizer to use: `{LBFGS,FIRE,BFGS,GPMin,Sella}`. Sella is optional (`pip install 'sella>=2.5.0'`), does not support symmetry constraints (always pass `--no-symmetry` with it), and needs sella ≥ 2.5.0 when combined with `--cell-opt`. |
| `--cell-opt` | off | When passed, unit-cell parameters are also relaxed (`FrechetCellFilter`); by default only atomic positions are relaxed |
| `--fmax` | `0.01` | Force convergence threshold (eV/Å) |
| `--steps` | `1000` | Maximum optimization steps |
| `--prefix` | `<input>_UMAopt` | Output file prefix |
| `--device` | auto | Force device: `{cpu, cuda}`. Default: auto-detect (GPU if available). |
| `--cuda-device` | `0` | CUDA device index to use when running on GPU (ignored with `--device cpu`) |
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

- **Sella** is optional: `pip install 'sella>=2.5.0'`. With `--cell-opt`, Sella uses its v2.5.0 cell-optimization support (`optimize_cell`). Sella does not support symmetry constraints — always pass `--no-symmetry` when using it (the script enforces this and exits with an error otherwise).
- **Cell relaxation & stress:** when `--cell-opt` is used, the calculator enables stress prediction via FAIRChem's `predict_untrained_stress` inference setting (https://fair-chem.github.io/); it stays disabled otherwise to save time. See <https://fair-chem.github.io/ase-calculator/#enabling-gradient-stress-or-hessian-prediction>.
- **CPU mode:** `--device cpu` forces CPU execution regardless of GPU availability. Useful on macOS, when the GPU is occupied by another process, or for debugging.
- If FAIRChem/UMA model loading fails, check that your `fairchem-core` provides the pretrained UMA models (`fairchem.core.pretrained_mlip`) and consult <https://github.com/facebookresearch/fairchem>.
- UMA models require a Hugging Face account with accepted access to the [facebook/UMA](https://huggingface.co/facebook/UMA) repository. Run `huggingface-cli login` before first use.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (see LICENSE).

## Citation

If you use this code or the UMA models in your work, please cite:

```bibtex
@article{10.1039/D6SC04941A,
    author = {Gunaga, Shubha and Schurko, Rob and Holmes, Sean T. and Mentink-Vigier, Frederic},
    title = {Accessible hybrid DFT-quality NMR crystallography via gas-phase Machine Learning Interatomic Potentials},
    journal = {Chemical Science},
    year = {2026},
    month = {09},
    issn = {2041-6520},
    doi = {10.1039/D6SC04941A},
    url = {https://doi.org/10.1039/D6SC04941A},
}

@misc{wood2025uma,
      title={UMA: A Family of Universal Models for Atoms},
      author={Wood, Brandon M. and Dzamba, Misko and Fu, Xiang and Gao, Meng and Shuaibi, Muhammed and Barroso-Luque, Luis and Abdelmaqsoud, Kareem and Gharakhanyan, Vahe and Kitchin, John R. and Levine, Daniel S. and Michel, Kyle and Sriram, Anuroop and Cohen, Taco and Das, Abhishek and Rizvi, Ammar and Sahoo, Sushree Jagriti and Ulissi, Zachary W. and Zitnick, C. Lawrence},
      year={2025},
      eprint={2506.23971},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.23971}
}
```
