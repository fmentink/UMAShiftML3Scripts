#!/usr/bin/env python3
"""
UMA Crystal Structure Optimization — relax CIF structures with Meta's UMA MLIPs
================================================================================

High-accuracy geometry optimization for crystals using Meta's Unified Models
for Atomistic simulations (UMA), via FAIRChem and ASE. Feed it a CIF file — or
a whole directory of them — and get back relaxed, symmetrized structures.

What it does per structure:
  * Optimizes atomic positions with LBFGS, FIRE, BFGS or GPMin — or Sella
    (optional: pip install 'sella>=2.5.0'; requires --no-symmetry)
  * Optionally relaxes the unit cell too (--cell-opt); for UMA this switches
    on stress prediction automatically, and only when needed
  * Keeps the structure in its space group during relaxation (FixSymmetry;
    disable with --no-symmetry) and reports if the space group changes
  * Detects the final space group (pymatgen/spglib), then writes:
      <prefix>.cif          optimized, symmetrized structure
      <prefix>.xyz          same structure in XYZ format
      <prefix>.traj         full optimization trajectory
      <prefix>_summary.txt  parameters + results

Runs on NVIDIA GPUs (CUDA) with automatic CPU fallback (e.g. macOS).

Quick start:
    python UMACrystalOptimize.py my_structure.cif --cell-opt --fmax 0.01
    python UMACrystalOptimize.py ./cif_dir --recursive --optimizer Sella --no-symmetry

License: CC-BY-4.0 (see LICENSE file)

If you use this code or the UMA models in your work, please cite:
  [1] Gunaga, S.S., Schurko, R.W., Holmes, S.T., Mentink-Vigier, F. "Accessible
      hybrid DFT-quality NMR crystallography via gas-phase Machine Learning
      Interatomic Potentials." Chemical Science (2026).
      https://doi.org/10.1039/D6SC04941A

  [2] Wood, B.M., Dzamba, M., Fu, X., et al. "UMA: A Family of Universal
      Models for Atoms." arXiv:2506.23971 (2025).
      https://arxiv.org/abs/2506.23971
"""




import ase.io
from ase.optimize import LBFGS, FIRE, BFGS, GPMin
from ase.filters import FrechetCellFilter
from fairchem.core import FAIRChemCalculator
from ase.constraints import FixSymmetry
import numpy as np
import time
import sys
import os
import argparse
import torch
import warnings
import re
import glob
from contextlib import contextmanager
from pathlib import Path

# Import pymatgen for symmetry analysis and proper CIF output
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter, CifParser


# Sella import (optional - gracefully handle if not installed)
# Install with: pip install 'sella>=2.5.0'
try:
    from sella import Sella
    SELLA_AVAILABLE = True
except ImportError:
    SELLA_AVAILABLE = False

# Inference settings for stress prediction (needed when relaxing the cell;
# UMA tasks are not trained with stress labels, so stress comes via autograd)
try:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
except ImportError:
    InferenceSettings = None


@contextmanager
def stripped_constraints(atoms):
    """Temporarily remove constraints before passing atoms to pymatgen,
    which only understands FixAtoms/FixCartesian and warns about others."""
    saved = atoms.constraints
    atoms.set_constraint()
    try:
        yield atoms
    finally:
        atoms.set_constraint(saved)


def setup_logm_warning_filter(threshold=1.0):
    """Filter logm warnings based on error size"""
    original_showwarning = warnings.showwarning
    
    def filtered_showwarning(message, category, filename, lineno, file=None, line=None):
        message_str = str(message)
        if "logm result may be inaccurate" in message_str:
            match = re.search(r'approximate err = ([\d.e+-]+)', message_str)
            if match and float(match.group(1)) < threshold:
                return
        if "crystal system" in message_str and "is not interpreted" in message_str:
            return
        original_showwarning(message, category, filename, lineno, file, line)
    
    warnings.showwarning = filtered_showwarning


def setup_calculator(modelName="uma-s-1p1", task_name="omol", cuda_device=None, relax_cell=False, device_override=None):
           
    """Setup UMA calculator once for reuse"""
    if device_override == "cpu":
        device = "cpu"
        print("✓ Using CPU (forced via --device cpu)")
    elif torch.cuda.is_available():
        gpu_id = 0 if cuda_device is None else int(cuda_device)
        torch.cuda.set_device(gpu_id)
        device = "cuda"
        print(f"✓ Using GPU: cuda:{gpu_id}")
    else:
        # No CUDA (e.g. macOS / Apple Silicon): fall back to CPU
        device = "cpu"
        print("⚠ No CUDA GPU found - falling back to CPU (expected on macOS). Runs will be slower.")

    from fairchem.core import pretrained_mlip

    # Stress is only needed when the cell is relaxed. Enable it then via autograd
    # (UMA tasks are not trained with stress labels); skip it otherwise to save time.
    inference_settings = None
    if relax_cell:
        if InferenceSettings is not None:
            inference_settings = InferenceSettings(predict_untrained_stress={task_name})
            print(f"✓ Stress prediction enabled for task '{task_name}' (required for cell relaxation)")
        else:
            print("⚠ Could not import fairchem InferenceSettings - stress may be unavailable; cell relaxation could fail")

    if inference_settings is not None:
        predictor = pretrained_mlip.get_predict_unit(modelName, device=device, inference_settings=inference_settings)
    else:
        predictor = pretrained_mlip.get_predict_unit(modelName, device=device)
    return FAIRChemCalculator(predictor, task_name)


def analyze_symmetry(atoms):
    """Get symmetry information using pymatgen"""
    try:
        adaptor = AseAtomsAdaptor()
        with stripped_constraints(atoms) as a:
            structure = adaptor.get_structure(a)
        sga = SpacegroupAnalyzer(structure)
        
        return {
            'symbol': sga.get_space_group_symbol(),
            'number': sga.get_space_group_number(),
            'point_group': sga.get_point_group_symbol(),
            'crystal_system': sga.get_crystal_system(),
            'n_operations': len(sga.get_symmetry_operations())
        }
    except:
        return None


def get_structure_info(atoms):
    """Get structure parameters and properties"""
    cell = atoms.get_cell()
    return {
        'formula': atoms.get_chemical_formula(),
        'n_atoms': len(atoms),
        'elements': set(atoms.get_chemical_symbols()),
        'lengths': cell.lengths(),
        'angles': cell.angles(),
        'volume': atoms.get_volume(),
        'density': len(atoms)/atoms.get_volume()
    }


def save_cif_with_symmetry(atoms, filename):
    """Save CIF with symmetry detected from final atomic positions via spglib.
    Constraints are stripped before conversion to pymatgen to avoid warnings,
    as pymatgen re-detects symmetry independently from positions."""
    try:
        adaptor = AseAtomsAdaptor()
        with stripped_constraints(atoms) as a:
            structure = adaptor.get_structure(a)
        writer = CifWriter(structure, symprec=0.1, angle_tolerance=5.0, refine_struct=False)
        writer.write_file(filename)
        return True
    except:
        try:
            ase.io.write(filename, atoms, format='cif')
            return True
        except:
            return False


def print_section(title, data=None, width=50):
    """Print formatted section header with optional data"""
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}")
    if data:
        for key, value in data.items():
            print(f"{key}: {value}")


def save_summary(filename, data):
    """Save optimization summary to file"""
    with open(filename, 'w') as f:
        f.write("UMA Crystal Structure Optimization Summary\n")
        f.write("="*50 + "\n\n")
        
        for section, content in data.items():
            f.write(f"{section}:\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {content}\n")
            f.write("\n")


def optimize_crystal(cif_file, calculator, relax_cell=True, optimizer_name="LBFGS", 
                    fmax=0.02, max_steps=1000, output_prefix=None, task_name="omol",
                    fix_symmetry=True,
                    max_initial_force=None):  
    """Optimize crystal structure using provided UMA calculator"""
    
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(cif_file))[0] + "_UMAopt"

    # Validate Sella up front (optional dependency)
    if optimizer_name == "Sella":
        if not SELLA_AVAILABLE:
            print("✗ Sella is not installed.")
            print("  Install with: pip install 'sella>=2.5.0'")
            return False
        if fix_symmetry:
            print("✗ Sella does not support symmetry constraints - re-run with --no-symmetry.")
            return False


    print_section("UMA CRYSTAL STRUCTURE OPTIMIZATION", {
        'Input CIF': cif_file,
        'Optimizer': optimizer_name,
        'Relax cell': relax_cell,
        'Force convergence': f"{fmax:.6f} eV/Å",
        'Max steps': max_steps,
        'Task': task_name,
    }, 70)
    
    # Read and analyze structure
    try:
        atoms = ase.io.read(cif_file)
        initial_symmetry = analyze_symmetry(atoms)
        initial_info = get_structure_info(atoms)
        print(f"✓ Read {cif_file}: {initial_info['formula']} ({initial_info['n_atoms']} atoms)")
        if initial_symmetry:
            print(f"  Space group: {initial_symmetry['symbol']} (#{initial_symmetry['number']})")
    except Exception as e:
        print(f"✗ Error reading CIF: {e}")
        return False
    
    # Setup calculator
    try:
        if task_name != "omc":
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
        atoms.calc = calculator
        if fix_symmetry:
            atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
            print("✓ Symmetry constraint applied")
        else:
            print("✗ Symmetry constraint not applied")
    except Exception as e:
        print(f"✗ Calculator setup failed: {e}")
        return False
    
    # Initial evaluation
    try:
        start_time = time.time()
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        eval_time = time.time() - start_time
        
        max_force = np.max(np.linalg.norm(initial_forces, axis=1))
        rms_force = np.sqrt(np.mean(np.sum(initial_forces**2, axis=1)))
        
        print_section("INITIAL EVALUATION", {
            'Energy': f"{initial_energy:.6f} eV ({initial_energy/initial_info['n_atoms']:.6f} eV/atom)",
            'Max force': f"{max_force:.6f} eV/Å",
            'RMS force': f"{rms_force:.6f} eV/Å",
            'Eval time': f"{eval_time:.2f} s"
        })
        
        # Check if initial forces are too large
        if max_initial_force is not None and max_force > max_initial_force:
            print(f"\n✗ Initial max force ({max_force:.6f} eV/Å) exceeds threshold ({max_initial_force:.6f} eV/Å)")
            print(f"   Structure likely too far from equilibrium - skipping optimization")
            return False
        
    except Exception as e:
        print(f"✗ Initial evaluation failed: {e}")
        return False
    
    # Optimization
    traj_file = f"{output_prefix}.traj"
    print_section("GEOMETRY OPTIMIZATION")
    print(f"Starting {optimizer_name} optimization...")
    
    opt_start_time = time.time()

    try:
        if optimizer_name == "Sella":
            # Sella v2.5.0+ supports cell optimization via optimize_cell (requires order=0)
            sella_kwargs = {"optimize_cell": True} if relax_cell else {}
            optimizer = Sella(atoms, order=0, trajectory=traj_file, **sella_kwargs)
        else:
            optimizer_classes = {"LBFGS": LBFGS, "FIRE": FIRE, "BFGS": BFGS, "GPMin": GPMin}
            if relax_cell:
                optimizer = optimizer_classes[optimizer_name](FrechetCellFilter(atoms), trajectory=traj_file)
            else:
                 optimizer = optimizer_classes[optimizer_name](atoms, trajectory=traj_file)

        converged = optimizer.run(fmax=fmax, steps=max_steps)
        total_steps = optimizer.nsteps
        opt_time = time.time() - opt_start_time

        if converged is None:
            converged = np.max(np.linalg.norm(atoms.get_forces(), axis=1)) <= fmax

        status = "✓ Converged" if converged else "✗ Failed to converge (max steps reached)"
        print(f"{status} after {total_steps} steps ({opt_time:.1f}s, {opt_time/max(total_steps,1):.2f}s/step)")

        if not converged:
            print(f"✗ Optimization did not converge within {max_steps} steps")
            return False

    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False
    
    # Final analysis
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    final_max_force = np.max(np.linalg.norm(final_forces, axis=1))
    final_rms_force = np.sqrt(np.mean(np.sum(final_forces**2, axis=1)))
    energy_change = final_energy - initial_energy
    
    final_symmetry = analyze_symmetry(atoms)
    final_info = get_structure_info(atoms)
    
    # Report results
    print_section("FINAL RESULTS", {
        'Energy change': f"{energy_change:.6f} eV",
        'Final energy': f"{final_energy:.6f} eV ({final_energy/initial_info['n_atoms']:.6f} eV/atom)",
        'Final max force': f"{final_max_force:.6f} eV/Å",
        'Final RMS force': f"{final_rms_force:.6f} eV/Å"
    })
    
    if final_symmetry:
        if initial_symmetry and initial_symmetry['number'] == final_symmetry['number']:
            print(f"✓ Space group preserved: {final_symmetry['symbol']}")
        else:
            old_sg = initial_symmetry['symbol'] if initial_symmetry else "Unknown"
            print(f"! Space group changed: {old_sg} → {final_symmetry['symbol']}")
    
    if relax_cell:
        vol_change = ((final_info['volume'] - initial_info['volume']) / initial_info['volume']) * 100
        print(f"Volume change: {vol_change:+.2f}% ({initial_info['volume']:.1f} → {final_info['volume']:.1f} Å³)")
    
    # Save files
    print_section("SAVING RESULTS")
    
    # Save CIF
    cif_output = f"{output_prefix}.cif"
    if save_cif_with_symmetry(atoms, cif_output):
        print(f"✓ CIF saved: {cif_output}")
    else:
        print(f"✗ CIF save failed")
    
    # Save XYZ
    xyz_output = f"{output_prefix}.xyz"
    ase.io.write(xyz_output, atoms, format='xyz')
    print(f"✓ XYZ saved: {xyz_output}")
    print(f"✓ Trajectory saved: {traj_file}")
    
    # Save summary
    summary_data = {
        'Input': {
            'CIF file': cif_file,
            'Formula': initial_info['formula'],
            'Atoms': initial_info['n_atoms'],
            'Initial space group': initial_symmetry['symbol'] if initial_symmetry else 'Unknown'
        },
        'Optimization': {
            'Optimizer': optimizer_name,
            'Task': task_name, 
            'Symmetry': fix_symmetry,
            'Steps': total_steps,
            'Converged': 'Yes' if converged else 'No',
            'Time': f"{opt_time:.1f}s"
        },
        'Results': {
            'Final energy': f"{final_energy:.6f} eV",
            'Energy change': f"{energy_change:.6f} eV",
            'Final max force': f"{final_max_force:.6f} eV/Å",
            'Final space group': final_symmetry['symbol'] if final_symmetry else 'Unknown',
            'Volume change': f"{vol_change:+.2f}%" if relax_cell else 'N/A'
        }
    }

    summary_file = f"{output_prefix}_summary.txt"
    save_summary(summary_file, summary_data)
    print(f"✓ Summary saved: {summary_file}")
    
    print_section("✓ OPTIMIZATION COMPLETE", width=70)
    print(f"\n{'='*70}")
    print("If you use this work in your research, please cite:")
    print('  [1] Gunaga et al., Chem. Sci. (2026) doi:10.1039/D6SC04941A')
    print('  [2] Wood et al., arXiv:2506.23971 (2025) — UMA models')

    return True


def find_cif_files(input_path, recursive=False):
    """Find CIF files in directory"""
    path = Path(input_path)
    if recursive:
        return sorted([str(f) for f in path.rglob("*.cif")])
    else:
        return sorted([str(f) for f in path.glob("*.cif")])


def batch_optimize(input_path, calculator, **kwargs):
    """Process multiple CIF files using same calculator"""
    recursive = kwargs.get('recursive', False)
    cif_files = find_cif_files(input_path, recursive)
    
    if not cif_files:
        print(f"✗ No CIF files found in {input_path}")
        return {"total": 0, "successful": 0, "failed": 0}
    
    print_section("BATCH PROCESSING", {
        'Directory': input_path,
        'Files found': len(cif_files),
        'Recursive': recursive
    }, 70)
    
    successful = 0
    failed_files = []
    
    for i, cif_file in enumerate(cif_files, 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING {i}/{len(cif_files)}: {os.path.basename(cif_file)}")
        print(f"{'='*70}")
        
        try:
            # Generate unique output prefix
            base_name = os.path.splitext(os.path.basename(cif_file))[0]
            file_prefix = kwargs.get('output_prefix')
            if file_prefix:
                file_prefix = f"{file_prefix}_{base_name}"
            else:
                file_prefix = f"{base_name}_UMAopt"
            
            # Remove parameters not accepted by optimize_crystal
            batch_kwargs = {k: v for k, v in kwargs.items() if k not in ['output_prefix', 'recursive']}
            
            success = optimize_crystal(
                cif_file=cif_file,
                calculator=calculator,
                output_prefix=file_prefix,
                **batch_kwargs
            )
            
            if success:
                successful += 1
                print(f"✓ {os.path.basename(cif_file)} completed")
            else:
                failed_files.append(os.path.basename(cif_file))
                print(f"✗ {os.path.basename(cif_file)} failed")
                
        except Exception as e:
            failed_files.append(f"{os.path.basename(cif_file)}: {str(e)}")
            print(f"✗ {os.path.basename(cif_file)} error: {e}")
    
    # Final summary
    failed = len(failed_files)
    print_section("BATCH SUMMARY", {
        'Total files': len(cif_files),
        'Successful': successful,
        'Failed': failed,
        'Success rate': f"{successful/len(cif_files)*100:.1f}%"
    }, 70)
    
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    return {"total": len(cif_files), "successful": successful, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="UMA Crystal Structure Optimization with Batch Processing")
    parser.add_argument("input_path", help="CIF file or directory containing CIF files")
    parser.add_argument("--optimizer", choices=["LBFGS", "FIRE", "BFGS", "GPMin", "Sella"], default="LBFGS",
                        help="Optimizer to use. Sella is optional (pip install 'sella>=2.5.0'), does not support symmetry constraints (use --no-symmetry), and needs sella >= 2.5.0 with --cell-opt.")
    parser.add_argument("--cell-opt", action="store_true", help="Don't relax cell")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force threshold (eV/Å), default 0.01")
    parser.add_argument("--steps", type=int, default=1000, help="Max optimization steps")
    parser.add_argument("--prefix", type=str, help="Output prefix")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Force device: 'cpu' or 'cuda'. Default: auto-detect (GPU if available)")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device ID (when using GPU), default is autoselect")
    parser.add_argument("--model", choices=["uma-s-1p1", "uma-s-1p2", "uma-m-1p1"], default="uma-s-1p1")
    parser.add_argument("--task-name", choices=["omol", "omc", "omat"], default="omol")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--no-symmetry", action="store_true", help="Disable the FixSymmetry constraint (required when using --optimizer Sella)")
    parser.add_argument("--max-initial-force", type=float, default=None, help="Skip optimization if initial max force exceeds this value (eV/Å)")
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.input_path):
        print(f"Error: '{args.input_path}' not found!")
        sys.exit(1)
    
    if args.fmax <= 0 or args.steps <= 0:
        print("Error: fmax and steps must be positive")
        sys.exit(1)

    # Sella does not support symmetry constraints (FixSymmetry)
    if args.optimizer == "Sella" and not args.no_symmetry:
        print("Error: Sella does not support symmetry constraints.")
        print("       Add --no-symmetry when using --optimizer Sella.")
        sys.exit(1)
    
    # Setup
    setup_logm_warning_filter()
    
    try:
        calculator = setup_calculator(args.model, args.task_name, args.cuda_device, relax_cell=args.cell_opt,
                                     device_override=args.device)
        print(f"✓ Calculator initialized: {args.model} ({args.task_name})")
    except Exception as e:
        print(f"✗ Calculator setup failed: {e}")
        sys.exit(1)
    
    # Prepare optimization parameters
    opt_params = {
        'relax_cell': args.cell_opt,
        'optimizer_name': args.optimizer,
        'fmax': args.fmax,
        'max_steps': args.steps,
        'output_prefix': args.prefix,
        'task_name': args.task_name,
        'recursive': args.recursive,
        'fix_symmetry': not args.no_symmetry,
        'max_initial_force': args.max_initial_force
    }
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file mode
        if not str(input_path).lower().endswith('.cif'):
            print(f"Error: '{input_path}' is not a CIF file!")
            sys.exit(1)
        
        success = optimize_crystal(str(input_path), calculator, **{k: v for k, v in opt_params.items() if k != 'recursive'})
        if not success:
            sys.exit(1)
            
    elif input_path.is_dir():
        # Batch mode
        results = batch_optimize(str(input_path), calculator, **opt_params)
        if results["failed"] > 0:
            sys.exit(1)
    else:
        print(f"Error: '{input_path}' is neither file nor directory!")
        sys.exit(1)


if __name__ == "__main__":
    main()
