# ShiftML3 CSA Calculator

Script to run ShiftML3 by Kellner, Matthias et al. (10.1021/acs.jpclett.5c01819).
It predicts NMR magnetic shielding tensor predictions for organic crystals using the ShiftML3 deep-learning model. This script takes in a 8-model ensemble averages full tensors before extracting principal components (σ11 ≥ σ22 ≥ σ33), with per-atom standard deviations as an uncertainty estimate. No DFT required — seconds on GPU, minutes on CPU.
The magnetic shiedlings can be converted back to chemical shifts following the calibrations reported in 10.1039/D6SC04941A

## Features

- Single file or directory batch processing (with optional `--recursive`)
- 8-model ensemble: average tensors first → eigenvalues (physically meaningful)
- Per-atom output with std across the ensemble (uncertainty estimate)
- Accepts CIF and XYZ files
- GPU (CUDA) or CPU execution (`--device cpu`)
- Save a txt and a magres file

## Requirements

- **Hardware:** NVIDIA GPU with CUDA recommended for speed. Falls back to CPU automatically if no GPU is available (e.g. macOS). Force CPU explicitly with `--device cpu`.
- Python 3 virtual environment (see Installation).

## Installation

This repository does **not** create or manage your Python environment; you must create one yourself.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **NVIDIA GPU users:** install the CUDA-enabled PyTorch build first (see <https://pytorch.org>) before running `pip install -r requirements.txt`.

## Usage

**Single file:**

```bash
python ShiftML3_CSA.py my_structure.cif
```

**CPU only:**

```bash
python ShiftML3_CSA.py my_structure.cif --device cpu
```

**Batch directory (recursive):**

```bash
python ShiftML3_CSA.py ./cif_dir --recursive
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | `.` | CIF/XYZ file or directory of files |
| `--device` | auto | Force device: `{cpu, cuda}`. Default: auto-detect (GPU if available). |
| `--recursive` | off | In directory mode, also search subdirectories |
| `--prefix` | `<input>_csa` | Output file prefix |
| `--save-magres` | off | Also save output as a `.magres` file (ASE format with `ms` tensor array) |

## Outputs

For each input structure, files are written:

- `<prefix>.txt` — per-atom table with columns: atom index, element, coordinates (x/y/z), σ_iso, σ11, σ22, σ33, anisotropy, span, and ensemble std for each. Sorted by element then index.
- `<prefix>.magres` *(only with `--save-magres`)* — ASE magres file containing the averaged shielding tensor array (`ms`, units: ppm)

## Notes

- **Ensemble:** the 8 models are run independently; their tensors are averaged before eigenvalue extraction. 
- **Units:** all shifts in ppm (shielding convention: σ11 ≥ σ22 ≥ σ33).
- If model loading fails, check that `shiftml` is installed and up to date: <https://github.com/compo-molab/ShiftML>.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (see LICENSE).

## Citation

If you use this code or the ShiftML models in your work, please cite:

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

@article{10.1021/acs.jpclett.5c01819,
    author = {Kellner, Matthias and Holmes, Jacob B. and Rodriguez-Madrid, Ruben and Viscosi, Florian and Zhang, Yuxuan and Emsley, Lyndon and Ceriotti, Michele},
    title = {A Deep Learning Model for Chemical Shieldings in Molecular Organic Solids Including Anisotropy},
    journal = {The Journal of Physical Chemistry Letters},
    year = {2025},
    volume = {16},
    number = {34},
    pages = {8714--8722},
    doi = {10.1021/acs.jpclett.5c01819},
    url = {https://doi.org/10.1021/acs.jpclett.5c01819},
}
```
