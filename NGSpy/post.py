import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from ngsolve import VOL

from main import GeometricParameters, load_results


def main():
    parser = argparse.ArgumentParser(description="Post-process NGSpy results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir

    print(f"Post-processing results in {out_dir}...")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")

    # load params
    with open(os.path.join(out_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join(out_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    geom_params_input = params["geometric"]
    V_c = metadata["V_c"]
    V_tip = params["simulation"]["V_tip"]
    geom_params = GeometricParameters(**geom_params_input)
    msh, u_dimless, u_volts_np = load_results(out_dir, geom_params, V_c)

    # --- 2D Potential Plot ---
    print("Creating 2D potential plot...")

    # Get mesh vertex coordinates
    verts = np.array([v.point for v in msh.vertices])
    x, y = verts[:, 0], verts[:, 1]

    # Get triangle element vertex indices (for plotting)
    tris = []
    for el in msh.Elements(VOL):
        tris.append([v.nr for v in el.vertices])
    triangulation = Triangulation(x, y, np.array(tris))

    # Evaluate potential at vertices (in volts)
    potential_at_verts = np.array([u_dimless(msh(*v)) for v in verts]) * V_c

    # Plotting
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot potential using colormap
    tpc = ax1.tripcolor(
        triangulation,
        potential_at_verts,
        shading="gouraud",
        cmap="jet" if V_tip > 0 else "jet_r",
    )
    fig1.colorbar(tpc, ax=ax1, label="Potential (V)")

    # Overlay mesh lines
    ax1.triplot(triangulation, "k-", lw=0.1, alpha=0.5)

    # Draw boundary lines
    ax1.axhline(y=0, color="w", linestyle="-", linewidth=0.5, alpha=0.25)
    ax1.axhline(
        y=-geom_params.l_sio2, color="w", linestyle="-", linewidth=0.5, alpha=0.25
    )

    ax1.set_xlabel("r (nm)")
    ax1.set_ylabel("z (nm)")
    ax1.set_title("2D Potential Distribution with Mesh")
    ax1.set_aspect("equal")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "potential_2d_plot.png"), dpi=300)
    print(f"Saved 2D plot to {os.path.join(out_dir, 'potential_2d_plot.png')}")

    # --- Line Profile Plots ---
    print("Creating line profile plot...")

    # Vertical Profile (along center axis r=0)
    num_points = 500
    # Get geometry range
    z_min = -geom_params.l_sio2 - (geom_params.l_vac - geom_params.l_sio2)
    z_max = geom_params.l_vac
    z_coords = np.linspace(z_min, z_max, num_points)

    potential_z = []
    valid_z = []
    for z in z_coords:
        try:
            # Evaluate potential at (r=0, z)
            val = u_dimless(msh(0, z)) * V_c
            potential_z.append(val)
            valid_z.append(z)
        except Exception:
            # Ignore points outside the mesh
            continue

    # Horizontal Profile (e.g., SiO2/SiC interface z = -l_sio2)
    r_max = geom_params.region_radius
    r_coords = np.linspace(0, r_max, num_points)
    z_level = -geom_params.l_sio2

    potential_r = []
    valid_r = []
    for r in r_coords:
        try:
            # Evaluate potential at (r, z_level)
            val = u_dimless(msh(r, z_level)) * V_c
            potential_r.append(val)
            valid_r.append(r)
        except Exception:
            continue

    # Plotting (combine both profiles into one figure)
    fig2, (ax2_z, ax2_r) = plt.subplots(2, 1, figsize=(8, 10))

    # Vertical Plot
    ax2_z.plot(valid_z, potential_z, "b-")
    ax2_z.set_xlabel("z (nm)")
    ax2_z.set_ylabel("Potential (V)")
    ax2_z.set_title("Line Profile along Center Axis (r=0)")
    ax2_z.grid(True)

    # Horizontal Plot
    ax2_r.plot(valid_r, potential_r, "r-")
    ax2_r.set_xlabel("r (nm)")
    ax2_r.set_ylabel("Potential (V)")
    ax2_r.set_title(f"Line Profile at z = {z_level} nm (SiC/SiO2 Interface)")
    ax2_r.grid(True)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "potential_line_profiles.png"), dpi=150)
    print(
        f"Saved line profiles to {os.path.join(out_dir, 'potential_line_profiles.png')}"
    )


if __name__ == "__main__":
    main()
