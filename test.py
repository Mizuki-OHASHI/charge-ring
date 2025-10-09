import numpy as np
import json
import os

def compare_solutions(fenics_dir, ngspy_dir):
    """Compare the two solutions at specific points"""
    
    # Load parameters
    with open(os.path.join(fenics_dir, "parameters.json")) as f:
        params_fenics = json.load(f)
    with open(os.path.join(ngspy_dir, "parameters.json")) as f:
        params_ngspy = json.load(f)
    
    print("=== Parameter Comparison ===")
    print(f"FEniCS Ef: {params_fenics['physical']['Ef']:.6f} eV")
    print(f"NGSpy Ef: {params_ngspy['physical']['Ef']:.6f} eV")
    
    # Load line profiles
    fenics_vert = np.loadtxt(os.path.join(fenics_dir, "line_profile_vertical.txt"))
    ngspy_vert = np.loadtxt(os.path.join(ngspy_dir, "line_profile_vertical.txt"))
    
    fenics_horz = np.loadtxt(os.path.join(fenics_dir, "line_profile_horizontal.txt"))
    ngspy_horz = np.loadtxt(os.path.join(ngspy_dir, "line_profile_horizontal.txt"))
    
    print("\n=== Vertical Profile Comparison (at r=0) ===")
    print(f"FEniCS max potential: {np.max(fenics_vert[:, 1]):.6f} V")
    print(f"NGSpy max potential: {np.max(ngspy_vert[:, 1]):.6f} V")
    
    print("\n=== Horizontal Profile Comparison (at SiC/SiO2 interface) ===")
    print(f"FEniCS max potential: {np.max(fenics_horz[:, 1]):.6f} V")
    print(f"NGSpy max potential: {np.max(ngspy_horz[:, 1]):.6f} V")
    
    # Interface charge comparison
    sigma_fenics = params_fenics['physical']['sigma_s']
    sigma_ngspy = params_ngspy['physical']['sigma_s']
    print("\n=== Interface Charge Density ===")
    print(f"FEniCS: {sigma_fenics:.3e} m^-2")
    print(f"NGSpy: {sigma_ngspy:.3e} m^-2")

if __name__ == "__main__":
    compare_solutions("FEniCS/example_output", "NGSpy/example_output")