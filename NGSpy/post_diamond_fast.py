import argparse
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np

matplotlib.use("Agg")  # Use non-GUI backend for faster plotting

from matplotlib.tri import Triangulation
from ngsolve import VOL

from main_diamond_fast import GeometricParameters, PhysicalParameters, create_mesh


def exp_clamped(x, limit=100.0):
    return np.exp(np.clip(x, -limit, limit))


def load_solution_vector(out_dir: str, fes) -> ng.GridFunction:
    """
    Load solution vector from saved files without regenerating mesh.

    Args:
        out_dir: Directory containing saved results
        fes: Finite element space (must match the saved solution)

    Returns:
        GridFunction containing the loaded solution
    """
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No metadata.json found in {out_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["ndof"] != fes.ndof:
        raise RuntimeError(f"DOF mismatch: saved={meta['ndof']} current={fes.ndof}")

    u_dim_path = os.path.join(out_dir, "u_dimless.npy")
    if not os.path.isfile(u_dim_path):
        raise FileNotFoundError(f"No u_dimless.npy found in {out_dir}")

    u_vec = np.load(u_dim_path)
    if u_vec.size != fes.ndof:
        raise RuntimeError("Vector length does not match current FES")

    u = ng.GridFunction(fes, name="potential_dimless")
    u.vec.FV().NumPy()[:] = u_vec

    return u


def precompute_mesh_geometry(msh, L_c: float):
    """Precompute mesh geometry information for reuse across all V_tip.

    Args:
        msh: NGSolve mesh
        L_c: Characteristic length [m]

    Returns:
        Dictionary containing precomputed geometry data
    """
    # Get mesh vertex coordinates
    verts = np.array([v.point for v in msh.vertices])
    x, y = verts[:, 0], verts[:, 1]
    r_coords = verts[:, 0] * L_c * 1e9  # r coordinate [nm]
    z_coords = verts[:, 1] * L_c * 1e9  # z coordinate [nm]

    # Get triangle element vertex indices (for plotting)
    tris = []
    for el in msh.Elements(VOL):
        tris.append([v.nr for v in el.vertices])
    triangulation = Triangulation(x, y, np.array(tris))

    # Plotting range
    r_max = np.max(r_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    return {
        "verts": verts,
        "x": x,
        "y": y,
        "r_coords": r_coords,
        "z_coords": z_coords,
        "triangulation": triangulation,
        "r_max": r_max,
        "z_min": z_min,
        "z_max": z_max,
    }


def find_vtip_subdirs(out_dir: str, V_tip_range: str = None) -> list[tuple[str, float]]:
    """Find V_tip_±X.XXV subdirectories and extract voltage values

    Returns:
        List of (subdir_path, V_tip_value) tuples, sorted by absolute V_tip value
    """
    pattern = re.compile(r"^V_tip_([+-]?\d+\.\d{2})V$")
    vtip_dirs = []

    V_tip_min, V_tip_max = -np.inf, np.inf
    if V_tip_range:
        V_tip_min_str, V_tip_max_str = V_tip_range.split(":")
        V_tip_min = float(V_tip_min_str)
        V_tip_max = float(V_tip_max_str)

    for item in os.listdir(out_dir):
        item_path = os.path.join(out_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                V_tip = float(match.group(1))
                if V_tip_min <= V_tip <= V_tip_max:
                    vtip_dirs.append((item_path, V_tip))

    # Sort by absolute value
    vtip_dirs.sort(key=lambda x: abs(x[1]))

    return vtip_dirs


def process_single_vtip(
    subdir: str,
    V_tip: float,
    msh,
    fes,
    mesh_geometry: dict,
    geom_params: GeometricParameters,
    phys_params: PhysicalParameters,
    V_c: float,
    assume_full_ionization: bool,
    plot_acceptor_ionization: bool,
    plot_charge_density: bool,
    dpi: int,
    use_feenstra: bool,
):
    """Process a single V_tip directory and generate plots

    Args:
        subdir: Subdirectory containing V_tip results
        V_tip: Tip voltage value
        msh: Pre-generated mesh (shared across all V_tip)
        fes: Finite element space (shared across all V_tip)
        mesh_geometry: Precomputed mesh geometry data (shared across all V_tip)
        geom_params: Geometric parameters
        phys_params: Physical parameters
        V_c: Characteristic voltage
        assume_full_ionization: Whether to assume full ionization
        plot_acceptor_ionization: Whether to plot acceptor ionization
        plot_charge_density: Whether to plot charge density
        dpi: DPI for plots
        use_feenstra: Whether to use Feenstra model
    """
    print(f"\n{'=' * 60}")
    print(f"Processing V_tip = {V_tip:.3f} V")
    print(f"{'=' * 60}")

    # Load solution vector from this subdirectory (no mesh regeneration)
    u_dimless = load_solution_vector(subdir, fes)

    # Extract precomputed geometry
    verts = mesh_geometry["verts"]
    triangulation = mesh_geometry["triangulation"]
    r_max = mesh_geometry["r_max"]
    z_min = mesh_geometry["z_min"]
    z_max = mesh_geometry["z_max"]
    L_c = geom_params.L_c  # [m]

    print("Creating 2D potential profile grid (1 nm spacing)...")
    r_nm_values = np.linspace(0.0, 200.0, 201)
    z_nm_values = np.linspace(-100.0, 0.0, 101)
    potential_grid = np.full((len(z_nm_values), len(r_nm_values)), np.nan)

    r_dimless_values = (r_nm_values * 1e-9) / L_c
    z_dimless_values = (z_nm_values * 1e-9) / L_c

    for zi, z_dimless in enumerate(z_dimless_values):
        for ri, r_dimless in enumerate(r_dimless_values):
            try:
                potential_grid[zi, ri] = u_dimless(msh(r_dimless, z_dimless)) * V_c
            except Exception:
                continue

    potential_dir = os.path.join(subdir, "potential_2d_profile")
    os.makedirs(potential_dir, exist_ok=True)

    np.savetxt(
        os.path.join(potential_dir, "r.txt"),
        np.column_stack((np.arange(len(r_nm_values), dtype=int), r_nm_values)),
        fmt=["%d", "%.6f"],
        header="r_idx r_nm",
    )
    np.savetxt(
        os.path.join(potential_dir, "z.txt"),
        np.column_stack((np.arange(len(z_nm_values), dtype=int), z_nm_values)),
        fmt=["%d", "%.6f"],
        header="z_idx z_nm",
    )
    np.savetxt(
        os.path.join(potential_dir, "potential.txt"),
        potential_grid,
        fmt="%.9e",
        header=(
            "Potential (V); rows follow z_idx (z from -50 nm to 0 nm), "
            "columns follow r_idx (r from 0 nm to 100 nm)"
        ),
    )
    print(f"Saved 2D potential profile data to {potential_dir}")

    # --- 2D Potential Plot ---
    print("Creating 2D potential plot...")

    # Get mesh vertex coordinates
    verts = np.array([v.point for v in msh.vertices])
    x, y = verts[:, 0], verts[:, 1]
    r_coords = verts[:, 0] * L_c * 1e9  # r coordinate [nm]
    z_coords = verts[:, 1] * L_c * 1e9  # z coordinate [nm]

    # Get triangle element vertex indices (for plotting)
    tris = []
    for el in msh.Elements(VOL):
        tris.append([v.nr for v in el.vertices])
    triangulation = Triangulation(x, y, np.array(tris))

    # Evaluate potential at vertices (in volts)
    potential_at_verts = np.array([u_dimless(msh(*v)) for v in verts]) * V_c

    # Plotting range
    r_max = np.max(r_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

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
        y=-geom_params.diamond_thickness,
        color="w",
        linestyle="-",
        linewidth=0.5,
        alpha=0.25,
    )

    ax1.set_xlabel("r (nm)")
    ax1.set_ylabel("z (nm)")
    ax1.set_title("2D Potential Distribution with Mesh")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, r_max)
    ax1.set_ylim(z_min, z_max)
    fig1.tight_layout()
    fig1.savefig(os.path.join(subdir, "potential_2d_plot.png"), dpi=dpi * 2)
    plt.close(fig1)
    print(f"Saved 2D plot to {os.path.join(subdir, 'potential_2d_plot.png')}")

    # --- Line Profile Plots ---
    print("Creating line profile plot...")

    # Vertical Profile (along center axis r=0)
    num_points = 200
    # Get geometry range
    z_min = -geom_params.diamond_thickness
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

    # Horizontal Profile (e.g., near the diamond surface)
    r_max = geom_params.region_radius
    r_coords = np.linspace(0, r_max, num_points)
    z_level = -1.0

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
    ax2_r.set_title(f"Line Profile at z = {z_level} nm (Near Diamond Surface)")
    ax2_r.grid(True)

    fig2.tight_layout()
    fig2.savefig(os.path.join(subdir, "potential_line_profiles.png"), dpi=dpi)
    plt.close(fig2)
    print(
        f"Saved line profiles to {os.path.join(subdir, 'potential_line_profiles.png')}"
    )

    # Save line profile data
    np.savetxt(
        os.path.join(subdir, "line_profile_vertical.txt"),
        np.column_stack((valid_z, potential_z)),
        header="z_nm potential_V",
    )
    np.savetxt(
        os.path.join(subdir, "line_profile_horizontal.txt"),
        np.column_stack((valid_r, potential_r)),
        header="r_nm potential_V",
    )

    if plot_acceptor_ionization:
        print("Creating acceptor ionization line profiles...")

        def calculate_acceptor_ionization(u_dimless_val, phys_params, V_c):
            eta_a = (phys_params.Ea - phys_params.Ef) / V_c - u_dimless_val
            if assume_full_ionization:
                return phys_params.Na
            return phys_params.Na / (1 + phys_params.g_a * exp_clamped(eta_a))

        z_coords_acceptor = np.linspace(-geom_params.diamond_thickness, 0.0, num_points)
        Na_minus_z = []
        valid_z_acceptor = []

        for z in z_coords_acceptor:
            try:
                u_val = u_dimless(msh(0, z))
                Na_minus_z.append(
                    calculate_acceptor_ionization(u_val, phys_params, V_c)
                )
                valid_z_acceptor.append(z)
            except Exception:
                continue

        r_coords_acceptor = np.linspace(0, r_max, num_points)
        z_level_acceptor = -1.0
        Na_minus_r = []
        valid_r_acceptor = []

        for r in r_coords_acceptor:
            try:
                u_val = u_dimless(msh(r, z_level_acceptor))
                Na_minus_r.append(
                    calculate_acceptor_ionization(u_val, phys_params, V_c)
                )
                valid_r_acceptor.append(r)
            except Exception:
                continue

        fig3, (ax3_z, ax3_r) = plt.subplots(2, 1, figsize=(8, 10))

        ax3_z.plot(
            valid_z_acceptor,
            np.array(Na_minus_z) * 1e-6,
            label="$N_A^-$",
            color="purple",
            lw=1.5,
            ls="-",
        )
        if assume_full_ionization:
            ax3_z.axhline(
                y=phys_params.Na * 1e-6,
                color="gray",
                ls="--",
                lw=1,
                label="$N_A$ (Full Ionization)",
            )
        ax3_z.set_xlabel("z (nm)")
        ax3_z.set_ylabel("Ionized Acceptor Density (cm$^{-3}$)")
        ax3_z.set_title("Acceptor Ionization Profile along Center Axis (r=0)")
        ax3_z.legend()
        ax3_z.grid(True, alpha=0.3)
        if valid_z_acceptor:
            ax3_z.set_xlim(min(valid_z_acceptor), max(valid_z_acceptor))

        ax3_r.plot(
            valid_r_acceptor,
            np.array(Na_minus_r) * 1e-6,
            label="$N_A^-$",
            color="purple",
            lw=1.5,
            ls="-",
        )
        if assume_full_ionization:
            ax3_r.axhline(
                y=phys_params.Na * 1e-6,
                color="gray",
                ls="--",
                lw=1,
                label="$N_A$ (Full Ionization)",
            )
        ax3_r.set_xlabel("r (nm)")
        ax3_r.set_ylabel("Ionized Acceptor Density (cm$^{-3}$)")
        ax3_r.set_title(f"Acceptor Ionization Profile at z = {z_level_acceptor:.1f} nm")
        ax3_r.legend()
        ax3_r.grid(True, alpha=0.3)
        if valid_r_acceptor:
            ax3_r.set_xlim(0, max(valid_r_acceptor))

        fig3.tight_layout()
        fig3.savefig(
            os.path.join(subdir, "acceptor_ionization_line_profiles.png"), dpi=dpi
        )
        plt.close(fig3)
        print(
            f"Saved acceptor ionization profiles to {os.path.join(subdir, 'acceptor_ionization_line_profiles.png')}"
        )

        np.savetxt(
            os.path.join(subdir, "acceptor_ionization_vertical.txt"),
            np.column_stack((valid_z_acceptor, np.array(Na_minus_z) * 1e-6)),
            header="z_nm Na_minus_cm-3",
        )
        np.savetxt(
            os.path.join(subdir, "acceptor_ionization_horizontal.txt"),
            np.column_stack((valid_r_acceptor, np.array(Na_minus_r) * 1e-6)),
            header="r_nm Na_minus_cm-3",
        )
        print("Saved acceptor ionization data to text files.")

    if plot_charge_density:
        print("Creating charge density line profiles...")
        Ef_dimless = phys_params.Ef / V_c
        Ec_dimless = phys_params.Ec / V_c
        Ev_dimless = phys_params.Ev / V_c
        Ea_dimless = phys_params.Ea / V_c

        def fd_half_aymerich(eta):
            a1_aymerich = 6.316
            a2_aymerich = 12.92
            C_deg_aymerich = 0.75224956896  # 4 / (3 * sqrt(pi))

            if eta < -10.0:
                return np.exp(eta)

            eta_safe = max(eta, -4.0)
            G_inv_denominator = (
                C_deg_aymerich
                * (eta_safe**2 + a1_aymerich * eta_safe + a2_aymerich) ** 0.75
            )
            exp_neg_eta = np.exp(-eta)

            return 1.0 / (exp_neg_eta + (1.0 / G_inv_denominator))

        def calculate_charge_densities(u_dimless_val):
            if use_feenstra:
                eta_n = (Ef_dimless - Ec_dimless) + u_dimless_val
                eta_p = (Ev_dimless - Ef_dimless) - u_dimless_val
                n = phys_params.Nc * fd_half_aymerich(eta_n)
                p = phys_params.Nv * fd_half_aymerich(eta_p)
            else:
                n = phys_params.n0 * exp_clamped(u_dimless_val)
                p = phys_params.p0 * exp_clamped(-u_dimless_val)

            if assume_full_ionization:
                Na_minus = phys_params.Na
            else:
                eta_a = (Ea_dimless - Ef_dimless) - u_dimless_val
                Na_minus = phys_params.Na / (1 + phys_params.g_a * exp_clamped(eta_a))

            return n, p, Na_minus

        z_coords_charge = np.linspace(-geom_params.diamond_thickness, 0.0, num_points)
        n_z = []
        p_z = []
        Na_minus_z = []
        valid_z_charge = []

        for z in z_coords_charge:
            try:
                u_val = u_dimless(msh(0, z))
                n, p, Na_minus = calculate_charge_densities(u_val)
                n_z.append(n)
                p_z.append(p)
                Na_minus_z.append(Na_minus)
                valid_z_charge.append(z)
            except Exception:
                continue

        r_coords_charge = np.linspace(0, r_max, num_points)
        z_level_charge = -1.0
        n_r = []
        p_r = []
        Na_minus_r = []
        valid_r_charge = []

        for r in r_coords_charge:
            try:
                u_val = u_dimless(msh(r, z_level_charge))
                n, p, Na_minus = calculate_charge_densities(u_val)
                n_r.append(n)
                p_r.append(p)
                Na_minus_r.append(Na_minus)
                valid_r_charge.append(r)
            except Exception:
                continue

        fig4, (ax4_z, ax4_r) = plt.subplots(2, 1, figsize=(8, 10))

        ax4_z.semilogy(
            valid_z_charge,
            np.array(n_z) * 1e-6,
            label="$n$ (Electron)",
            color="blue",
            lw=1.5,
            ls="-",
        )
        ax4_z.semilogy(
            valid_z_charge,
            np.array(p_z) * 1e-6,
            label="$p$ (Hole)",
            color="red",
            lw=1.0,
            ls="--",
        )
        ax4_z.semilogy(
            valid_z_charge,
            np.array(Na_minus_z) * 1e-6,
            label="$N_A^-$ (Ionized Acceptor)",
            color="purple",
            lw=1.5,
            ls="-",
        )
        ax4_z.set_xlabel("z (nm)")
        ax4_z.set_ylabel("Charge Density (cm$^{-3}$)")
        ionization_status = (
            "Full Ionization" if assume_full_ionization else "Partial Ionization"
        )
        ax4_z.set_title(
            f"Charge Density Profile along Center Axis (r=0)\n({ionization_status})"
        )
        ax4_z.legend()
        ax4_z.grid(True, which="both", alpha=0.3)
        if valid_z_charge:
            ax4_z.set_xlim(min(valid_z_charge), max(valid_z_charge))

        ax4_r.semilogy(
            valid_r_charge,
            np.array(n_r) * 1e-6,
            label="$n$ (Electron)",
            color="blue",
            lw=1.5,
            ls="-",
        )
        ax4_r.semilogy(
            valid_r_charge,
            np.array(p_r) * 1e-6,
            label="$p$ (Hole)",
            color="red",
            lw=1.0,
            ls="--",
        )
        ax4_r.semilogy(
            valid_r_charge,
            np.array(Na_minus_r) * 1e-6,
            label="$N_A^-$ (Ionized Acceptor)",
            color="purple",
            lw=1.5,
            ls="-",
        )
        ax4_r.set_xlabel("r (nm)")
        ax4_r.set_ylabel("Charge Density (cm$^{-3}$)")
        ax4_r.set_title(
            f"Charge Density Profile at z = {z_level_charge:.1f} nm\n({ionization_status})"
        )
        ax4_r.legend()
        ax4_r.grid(True, which="both", alpha=0.3)
        if valid_r_charge:
            ax4_r.set_xlim(0, max(valid_r_charge))

        fig4.tight_layout()
        fig4.savefig(os.path.join(subdir, "charge_density_line_profiles.png"), dpi=dpi)
        plt.close(fig4)
        print(
            f"Saved charge density profiles to {os.path.join(subdir, 'charge_density_line_profiles.png')}"
        )

        np.savetxt(
            os.path.join(subdir, "charge_density_vertical.txt"),
            np.column_stack(
                (
                    valid_z_charge,
                    np.array(n_z) * 1e-6,
                    np.array(p_z) * 1e-6,
                    np.array(Na_minus_z) * 1e-6,
                )
            ),
            header="z_nm n_cm-3 p_cm-3 Na_minus_cm-3",
        )
        np.savetxt(
            os.path.join(subdir, "charge_density_horizontal.txt"),
            np.column_stack(
                (
                    valid_r_charge,
                    np.array(n_r) * 1e-6,
                    np.array(p_r) * 1e-6,
                    np.array(Na_minus_r) * 1e-6,
                )
            ),
            header="r_nm n_cm-3 p_cm-3 Na_minus_cm-3",
        )
        print("Saved charge density data to text files.")


def create_comparison_plots(vtip_dirs: list[tuple[str, float]], out_dir: str):
    """Create comparison plots overlaying all V_tip line profiles"""
    print("\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)

    comparison_dir = os.path.join(out_dir, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)

    # Collect data from all V_tip directories
    all_data_vertical = []
    all_data_horizontal = []

    for subdir, V_tip in vtip_dirs:
        vert_file = os.path.join(subdir, "line_profile_vertical.txt")
        horiz_file = os.path.join(subdir, "line_profile_horizontal.txt")

        if os.path.exists(vert_file):
            data = np.loadtxt(vert_file)
            all_data_vertical.append((V_tip, data[:, 0], data[:, 1]))

        if os.path.exists(horiz_file):
            data = np.loadtxt(horiz_file)
            all_data_horizontal.append((V_tip, data[:, 0], data[:, 1]))

    if not all_data_vertical and not all_data_horizontal:
        print("No line profile data found for comparison.")
        return

    # Plot vertical comparison
    if all_data_vertical:
        fig_v, ax_v = plt.subplots(figsize=(8, 6))
        for V_tip, z, potential in all_data_vertical:
            ax_v.plot(z, potential, label=f"V_tip = {V_tip:.2f} V")
        ax_v.set_xlabel("z (nm)")
        ax_v.set_ylabel("Potential (V)")
        ax_v.set_title("Vertical Line Profiles (r=0) - All V_tip")
        ax_v.legend()
        ax_v.grid(True)
        fig_v.tight_layout()
        fig_v.savefig(os.path.join(comparison_dir, "vertical_comparison.png"), dpi=150)
        plt.close(fig_v)
        print(f"Saved vertical comparison to {comparison_dir}")

    # Plot horizontal comparison
    if all_data_horizontal:
        fig_h, ax_h = plt.subplots(figsize=(8, 6))
        for V_tip, r, potential in all_data_horizontal:
            ax_h.plot(r, potential, label=f"V_tip = {V_tip:.2f} V")
        ax_h.set_xlabel("r (nm)")
        ax_h.set_ylabel("Potential (V)")
        ax_h.set_title("Horizontal Line Profiles - All V_tip")
        ax_h.legend()
        ax_h.grid(True)
        fig_h.tight_layout()
        fig_h.savefig(
            os.path.join(comparison_dir, "horizontal_comparison.png"), dpi=150
        )
        plt.close(fig_h)
        print(f"Saved horizontal comparison to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Post-process NGSpy results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--plot_acceptor_ionization",
        action="store_true",
        help="Plot acceptor ionization profile",
    )
    parser.add_argument(
        "--plot_charge_density",
        action="store_true",
        help="Plot charge density profile",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="Skip comparison plots",
    )
    parser.add_argument(
        "--V_tip_range", type=str, default=None, help="V_tip range to process (min:max)"
    )
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir

    print(f"Post-processing results in {out_dir}...")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")

    # Load global parameters
    params_file = os.path.join(out_dir, "parameters.json")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"No parameters.json found in {out_dir}")

    with open(params_file, "r") as f:
        params = json.load(f)

    geom_params_input = params["geometric"]
    phys_params_input = params["physical"]
    assume_full_ionization = params["simulation"].get("assume_full_ionization", False)
    geom_params = GeometricParameters(**geom_params_input)
    phys_params = PhysicalParameters(**phys_params_input)
    phys_params.update_equilibrium_densities()

    # Get V_c from any metadata.json (should be the same for all)
    vtip_dirs = find_vtip_subdirs(out_dir, V_tip_range=args.V_tip_range)

    if not vtip_dirs:
        raise FileNotFoundError(f"No V_tip_±X.XXV subdirectories found in {out_dir}")

    print(f"Found {len(vtip_dirs)} V_tip directories:")
    for subdir, V_tip in vtip_dirs:
        print(f"  {os.path.basename(subdir)}: V_tip = {V_tip:.2f} V")

    # Get V_c and use_feenstra from first subdirectory
    first_metadata = os.path.join(vtip_dirs[0][0], "metadata.json")
    with open(first_metadata, "r") as f:
        metadata = json.load(f)
    V_c = metadata["V_c"]
    use_feenstra = metadata.get("Feenstra", True)

    # Create mesh once (shared by all V_tip)
    print("\nCreating mesh (once for all V_tip)...")
    msh = create_mesh(geom_params)
    fes = ng.H1(msh, order=1)
    print(f"Mesh created: {fes.ndof} DOFs")

    # Precompute mesh geometry (once for all V_tip)
    print("Precomputing mesh geometry...")
    mesh_geometry = precompute_mesh_geometry(msh, geom_params.L_c)
    print(f"Mesh geometry precomputed: {len(mesh_geometry['verts'])} vertices")

    # Process each V_tip directory
    for subdir, V_tip in vtip_dirs:
        process_single_vtip(
            subdir,
            V_tip,
            msh,
            fes,
            mesh_geometry,
            geom_params,
            phys_params,
            V_c,
            assume_full_ionization,
            args.plot_acceptor_ionization,
            args.plot_charge_density,
            args.dpi,
            use_feenstra,
        )

    # Create comparison plots
    if not args.no_comparison:
        create_comparison_plots(vtip_dirs, out_dir)

    print("\n" + "=" * 60)
    print("Post-processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
