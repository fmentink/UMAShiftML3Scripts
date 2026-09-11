#!/usr/bin/env python3
"""
ShiftML3 CSA Calculator — fast NMR chemical shielding tensors for organic crystals
================================================================================

Predicts full chemical shielding tensors (isotropic + anisotropy) using the
ShiftML3 deep-learning model. Runs an 8-model ensemble, averages the tensors,
then extracts principal components (σ11 ≥ σ22 ≥ σ33). Reports per-atom results
with standard deviations across the ensemble as a measure of prediction uncertainty.

Feed it a CIF or XYZ file — or a whole directory — and get back a table of
shielding parameters in seconds (GPU) to minutes (CPU), no DFT required.

Quick start:
    python ShiftML3_CSA.py my_structure.cif
    python ShiftML3_CSA.py ./cif_dir --recursive
    python ShiftML3_CSA.py my_structure.cif --device cpu

License: CC-BY-4.0 (see LICENSE file)

If you use this code or the ShiftML models in your work, please cite:
  [1] Gunaga, S.S., Schurko, R.W., Holmes, S.T., Mentink-Vigier, F. "Accessible
      hybrid DFT-quality NMR crystallography via gas-phase Machine Learning
      Interatomic Potentials." Chemical Science (2026).
      https://doi.org/10.1039/D6SC04941A

  [2] Kellner, M., Holmes, J.B., Rodriguez-Madrid, R., Viscosi, F., Zhang, Y.,
      Emsley, L., Ceriotti, M. "A Deep Learning Model for Chemical Shieldings in
      Molecular Organic Solids Including Anisotropy." J. Phys. Chem. Lett. 16(34),
      8714–8722 (2025).
      https://doi.org/10.1021/acs.jpclett.5c01819
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read


def setup_calculator(device_override=None):
    """Load ShiftML3 model once."""
    from shiftml.ase import ShiftML

    if device_override == "cpu":
        device = "cpu"
        print("✓ Using CPU (forced via --device cpu)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ No CUDA GPU found — using CPU. Runs will be slower.")

    print("Loading ShiftML3 model...")
    calculator = ShiftML("ShiftML3", device=device)
    print(f"✓ Model loaded on {device}\n")
    return calculator


def compute_csa(atoms, calculator):
    """Compute ensemble CSA tensors for a single structure.

    Returns:
        cs_iso: (n_atoms,) isotropic shifts in ppm
        sigma11, sigma22, sigma33: (n_atoms,) principal components
        std_iso, std_11, std_22, std_33: (n_atoms,) ensemble standard deviations
    """
    n_atoms = len(atoms)

    # Get ensemble tensors: shape (n_atoms, 3, 3, n_models)
    all_tensors_raw = calculator.get_cs_tensor_ensemble(atoms)
    # Transpose to (n_models, n_atoms, 3, 3)
    all_tensors = np.transpose(all_tensors_raw, (3, 0, 1, 2))
    n_models = all_tensors.shape[0]

    # Average tensors across models → (n_atoms, 3, 3)
    mean_tensor = np.mean(all_tensors, axis=0)

    # Extract eigenvalues from averaged tensor
    cs_iso = np.zeros(n_atoms)
    sigma11 = np.zeros(n_atoms)
    sigma22 = np.zeros(n_atoms)
    sigma33 = np.zeros(n_atoms)

    for i in range(n_atoms):
        cs_iso[i] = np.trace(mean_tensor[i]) / 3.0
        eigenvals = np.linalg.eigvalsh(mean_tensor[i])  # ascending order
        sigma33[i], sigma22[i], sigma11[i] = eigenvals[0], eigenvals[1], eigenvals[2]

    # Per-model eigenvalues → std across ensemble
    all_s11, all_s22, all_s33 = [], [], []
    for m in range(n_models):
        s11_m, s22_m, s33_m = np.zeros(n_atoms), np.zeros(n_atoms), np.zeros(n_atoms)
        for i in range(n_atoms):
            ev = np.linalg.eigvalsh(all_tensors[m, i])  # ascending
            s33_m[i], s22_m[i], s11_m[i] = ev[0], ev[1], ev[2]
        all_s11.append(s11_m)
        all_s22.append(s22_m)
        all_s33.append(s33_m)

    std_11 = np.std(all_s11, axis=0)
    std_22 = np.std(all_s22, axis=0)
    std_33 = np.std(all_s33, axis=0)
    # std of iso: mean of per-model traces / 3
    all_iso = [np.trace(all_tensors[m, i]) / 3.0 for m in range(n_models) for i in range(n_atoms)]
    all_iso = np.array(all_iso).reshape(n_models, n_atoms)
    std_iso = np.std(all_iso, axis=0)

    return cs_iso, sigma11, sigma22, sigma33, std_iso, std_11, std_22, std_33, mean_tensor


def process_file(input_file, calculator, output_prefix=None, save_magres=False):
    """Process a single structure file and write results."""
    input_path = Path(input_file)

    if output_prefix is None:
        output_prefix = input_path.stem + "_csa"

    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")

    # Read structure (single frame)
    atoms = read(input_file, index=0)
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    positions = atoms.get_positions()

    print(f"  Formula: {atoms.get_chemical_formula()} ({n_atoms} atoms)")
    print(f"  Computing ensemble CSA tensors (8 models)...")

    cs_iso, s11, s22, s33, std_iso, std_s11, std_s22, std_s33, mean_tensor = compute_csa(atoms, calculator)

    # Write output — sorted by element then index
    atom_indices = sorted(range(n_atoms), key=lambda i: (symbols[i], i))

    out_file = f"{output_prefix}.txt"
    with open(out_file, "w") as f:
        f.write("# ShiftML3 Ensemble CSA Results\n")
        f.write(f"# Input: {input_file}\n")
        f.write(f"# Formula: {atoms.get_chemical_formula()} ({n_atoms} atoms)\n")
        f.write("# sigma11 >= sigma22 >= sigma33 (principal components, ppm)\n")
        f.write("# std = standard deviation across 8-model ensemble\n")
        f.write("#\n")
        f.write(f"{'Atom':>5} {'Element':>8} {'X':>12} {'Y':>12} {'Z':>12} "
                f"{'sigma_iso':>10} {'sigma11':>10} {'sigma22':>10} {'sigma33':>10} "
                f"{'anisotropy':>10} {'span':>10} "
                f"{'std_iso':>8} {'std11':>8} {'std22':>8} {'std33':>8}\n")

        for i in atom_indices:
            anisotropy = s33[i] - cs_iso[i]
            span = s11[i] - s33[i]
            f.write(f"{i:5d} {symbols[i]:>8} "
                    f"{positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f} "
                    f"{cs_iso[i]:10.4f} {s11[i]:10.4f} {s22[i]:10.4f} {s33[i]:10.4f} "
                    f"{anisotropy:10.4f} {span:10.4f} "
                    f"{std_iso[i]:8.4f} {std_s11[i]:8.4f} {std_s22[i]:8.4f} {std_s33[i]:8.4f}\n")

    print(f"  ✓ Results written to: {out_file}")

    # Optionally save as .magres (ASE format with ms tensor array)
    if save_magres:
        from ase.io import write as ase_write
        atoms.arrays['ms'] = mean_tensor
        atoms.info['magres_units'] = {'ms': 'ppm'}
        magres_file = f"{output_prefix}.magres"
        ase_write(magres_file, atoms)
        print(f"  ✓ Magres file saved to: {magres_file}")

    return out_file


def find_input_files(input_path, recursive=False):
    """Find CIF/XYZ files in a directory."""
    path = Path(input_path)
    patterns = ["*.cif", "*.xyz"]
    if recursive:
        files = []
        for pat in patterns:
            files.extend(path.rglob(pat))
        return sorted(set(str(f) for f in files))
    else:
        files = []
        for pat in patterns:
            files.extend(path.glob(pat))
        return sorted(set(str(f) for f in files))


def main():
    parser = argparse.ArgumentParser(
        description="ShiftML3 Ensemble CSA Calculator — per-atom chemical shielding tensors"
    )
    parser.add_argument("input", nargs="?", default=".",
                        help="CIF/XYZ file or directory (default: current dir)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Force device: 'cpu' or 'cuda'. Default: auto-detect.")
    parser.add_argument("--recursive", action="store_true",
                        help="In directory mode, search subdirectories")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Output prefix (default: <input>_csa)")
    parser.add_argument("--save-magres", action="store_true",
                        help="Also save output as a .magres file (ASE format with ms tensor array)")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: '{args.input}' not found!")
        sys.exit(1)

    # Determine files to process
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [str(input_path)]
    elif input_path.is_dir():
        input_files = find_input_files(args.input, recursive=args.recursive)
        if not input_files:
            print(f"No CIF/XYZ files found in {args.input}")
            sys.exit(1)
    else:
        print(f"Error: '{args.input}' is neither a file nor directory")
        sys.exit(1)

    # Load model once
    calculator = setup_calculator(device_override=args.device)

    # Process files
    outputs = []
    for f in input_files:
        prefix = args.prefix if args.prefix else None
        out = process_file(f, calculator, output_prefix=prefix, save_magres=args.save_magres)
        outputs.append(out)

    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"  Processed: {len(outputs)} file(s)")
    for inp, out in zip(input_files, outputs):
        print(f"  {Path(inp).name} → {out}")

    # Citation reminder
    print(f"\n{'='*60}")
    print("If you use this work in your research, please cite:")
    print('  [1] Gunaga et al., Chem. Sci. (2026) doi:10.1039/D6SC04941A')
    print('  [2] Kellner et al., J. Phys. Chem. Lett. 16, 8714 (2025) doi:10.1021/acs.jpclett.5c01819')


if __name__ == "__main__":
    main()
