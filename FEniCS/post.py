import argparse
import json
import os

import adios4dolfinx
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI

from main import GeometricParameters

assert MPI.COMM_WORLD.size == 1, "This script should be run with a single process."


def main():
    parser = argparse.ArgumentParser(description="Post-process FEniCS results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir

    print(f"Post-processing results in {out_dir}...")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")

    # Load mesh and solution
    with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "solution.xdmf"), "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    with open(os.path.join(out_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    L_c = params.get("L_c", 1e-9)  # [m]
    V_tip = params["simulation"]["V_tip"]  # [V]
    geom_params_input = params["geometric"]
    geom_params = GeometricParameters(**geom_params_input)

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V)
    adios4dolfinx.read_function(
        os.path.join(out_dir, "potential.bp"), u, name="potential"
    )
    print("Potential function loaded.")

    # --- 2D Potential Plot ---
    print("Creating 2D potential plot...")

    verts = msh.geometry.x
    r_coords = verts[:, 0] * L_c * 1e9  # r coordinate [nm]
    z_coords = verts[:, 1] * L_c * 1e9  # z coordinate [nm]

    # Potential at vertices: u.x.array corresponds to vertex values for Lagrange degree 1
    u_vals_on_verts = u.x.array

    # Mesh triangle connectivity: (2, 0) means connection between 2D cells (triangles) and 0D cells (vertices)
    triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)

    # Plotting range
    r_max = np.max(r_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    # Colormap settings
    cmap = "jet" if V_tip > 0 else "jet_r"

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot potential using tricontourf
    tpc = ax1.tricontourf(
        r_coords,
        z_coords,
        u_vals_on_verts,
        levels=201,
        cmap=cmap,
    )
    fig1.colorbar(tpc, ax=ax1, label="Potential (V)")

    # Overlay mesh lines
    ax1.triplot(
        r_coords,
        z_coords,
        triangles,
        color="k",
        linewidth=0.1,
        alpha=0.5,
    )

    # Draw interface lines
    ax1.axhline(y=0, color="w", linestyle="-", linewidth=0.5, alpha=0.25)
    ax1.axhline(
        y=-geom_params.l_sio2, color="w", linestyle="-", linewidth=0.5, alpha=0.25
    )

    ax1.set_xlabel("r (nm)")
    ax1.set_ylabel("z (nm)")
    ax1.set_title("2D Potential Distribution with Mesh")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, r_max)
    ax1.set_ylim(z_min, z_max)
    fig1.tight_layout()

    save_path = os.path.join(out_dir, "potential_2d_plot.png")
    fig1.savefig(save_path, dpi=300)
    plt.close(fig1)
    print(f"Saved 2D plot to {save_path}")

    # --- Line Profile Plots ---
    print("Creating line profile plot...")

    # Vertical Profile (along center axis r=0)
    num_points = 500
    # Use geometry parameters to get the correct range
    z_min_profile = -geom_params.l_sio2 - (geom_params.l_vac - geom_params.l_sio2)
    z_max_profile = geom_params.l_vac
    z_coords_profile = np.linspace(z_min_profile, z_max_profile, num_points)

    # Convert z coordinates from nm to dimensionless (inverse of L_c * 1e9)
    points_z = np.array([[0.0, z / (L_c * 1e9), 0.0] for z in z_coords_profile])

    potential_z = []
    valid_z = []

    # Use u.eval to evaluate potential at specific points
    from dolfinx import geometry

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)

    for i, z in enumerate(z_coords_profile):
        try:
            point = points_z[i : i + 1]
            # Find cells that might contain the point
            cell_candidates = geometry.compute_collisions_points(bb_tree, point)
            # Find which cell actually contains the point
            colliding_cells = geometry.compute_colliding_cells(
                msh, cell_candidates, point
            )

            if len(colliding_cells.links(0)) > 0:
                cell_id = colliding_cells.links(0)[0]
                val = u.eval(point, np.array([cell_id], dtype=np.int32))
                potential_z.append(val[0])
                valid_z.append(z)
        except Exception:
            # Ignore points outside the mesh
            continue

    # Horizontal Profile (at SiO2/SiC interface z = -l_sio2)
    z_level = -geom_params.l_sio2  # Already in nm
    r_max_profile = geom_params.region_radius  # Use geometry parameter for range
    r_coords_profile = np.linspace(0, r_max_profile, num_points)

    # Convert coordinates from nm to dimensionless
    points_r = np.array(
        [[r / (L_c * 1e9), z_level / (L_c * 1e9), 0.0] for r in r_coords_profile]
    )

    potential_r = []
    valid_r = []

    for i, r in enumerate(r_coords_profile):
        try:
            point = points_r[i : i + 1]
            # Find cells that might contain the point
            cell_candidates = geometry.compute_collisions_points(bb_tree, point)
            # Find which cell actually contains the point
            colliding_cells = geometry.compute_colliding_cells(
                msh, cell_candidates, point
            )

            if len(colliding_cells.links(0)) > 0:
                cell_id = colliding_cells.links(0)[0]
                val = u.eval(point, np.array([cell_id], dtype=np.int32))
                potential_r.append(val[0])
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
    ax2_r.set_title(f"Line Profile at z = {z_level:.1f} nm (SiC/SiO2 Interface)")
    ax2_r.grid(True)

    fig2.tight_layout()
    save_path = os.path.join(out_dir, "potential_line_profiles.png")
    fig2.savefig(save_path, dpi=150)
    plt.close(fig2)
    print(f"Saved line profiles to {save_path}")

    # Save line profile data
    np.savetxt(
        os.path.join(out_dir, "line_profile_vertical.txt"),
        np.column_stack((valid_z, potential_z)),
        header="z_nm potential_V",
    )
    np.savetxt(
        os.path.join(out_dir, "line_profile_horizontal.txt"),
        np.column_stack((valid_r, potential_r)),
        header="r_nm potential_V",
    )


if __name__ == "__main__":
    main()
