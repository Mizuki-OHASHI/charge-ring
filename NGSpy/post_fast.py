import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from ngsolve import VOL

from main_fast import GeometricParameters, PhysicalParameters, load_results


def find_vtip_subdirs(out_dir: str) -> list[tuple[str, float]]:
    """Find V_tip_±X.XXV subdirectories and extract voltage values

    Returns:
        List of (subdir_path, V_tip_value) tuples, sorted by absolute V_tip value
    """
    pattern = re.compile(r"^V_tip_([+-]?\d+\.\d{2})V$")
    vtip_dirs = []

    for item in os.listdir(out_dir):
        item_path = os.path.join(out_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                V_tip = float(match.group(1))
                vtip_dirs.append((item_path, V_tip))

    # Sort by absolute value
    vtip_dirs.sort(key=lambda x: abs(x[1]))

    return vtip_dirs


def process_single_vtip(
    subdir: str,
    V_tip: float,
    geom_params: GeometricParameters,
    phys_params: PhysicalParameters,
    V_c: float,
    assume_full_ionization: bool,
    plot_donor_ionization: bool,
    plot_charge_density: bool,
    plot_mesh: bool,
):
    """Process a single V_tip directory and generate plots"""
    print(f"\n{'=' * 60}")
    print(f"Processing V_tip = {V_tip:.3f} V")
    print(f"{'=' * 60}")

    # Load results from this subdirectory
    msh, u_dimless, _ = load_results(subdir, geom_params, V_c)
    L_c = geom_params.L_c  # [m]
    parser = argparse.ArgumentParser(description="Post-process NGSpy results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--plot_donor_ionization",
        action="store_true",
        help="Plot donor ionization profile",
    )
    parser.add_argument(
        "--plot_charge_density",
        action="store_true",
        help="Plot charge density profile",
    )
    parser.add_argument(
        "--plot_mesh", action="store_true", help="Plot mesh (for publication)"
    )
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir
    plot_donor_ionization = args.plot_donor_ionization
    plot_charge_density = args.plot_charge_density
    plot_mesh = args.plot_mesh
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
        y=-geom_params.l_sio2, color="w", linestyle="-", linewidth=0.5, alpha=0.25
    )

    ax1.set_xlabel("r (nm)")
    ax1.set_ylabel("z (nm)")
    ax1.set_title("2D Potential Distribution with Mesh")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, r_max)
    ax1.set_ylim(z_min, z_max)
    fig1.tight_layout()
    fig1.savefig(os.path.join(subdir, "potential_2d_plot.png"), dpi=300)
    plt.close(fig1)
    print(f"Saved 2D plot to {os.path.join(subdir, 'potential_2d_plot.png')}")

    # --- plot mesh only (for publication) ---
    if plot_mesh:
        figwidth_mm = 160
        figwidth_in = figwidth_mm / 25.4
        rc_params = {
            "font.size": 10,  # 目盛りや凡例の基本サイズ
            "axes.titlesize": 11.5,  # 軸のタイトル
            "axes.labelsize": 11.5,  # 軸ラベル (x, y)
            "xtick.labelsize": 10,  # x軸目盛り
            "ytick.labelsize": 10,  # y軸目盛り
            "legend.fontsize": 10,  # 凡例
            # "figure.titlesize": 14,  # fig.suptitle()
            # PDF保存時にフォントを埋め込む設定
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        rc_params_stashed = plt.rcParams.copy()
        plt.rcParams.update(rc_params)
        fig, axes = plt.subplots(1, 2, figsize=(figwidth_in, figwidth_in * 0.4))
        axes[0].triplot(triangulation, "k-", lw=0.5)
        axes[0].axhline(y=0, color="white", linestyle="--", linewidth=0.5)
        axes[0].axhline(
            y=-geom_params.l_sio2, color="white", linestyle="--", linewidth=0.5
        )
        # zoom region in full mesh
        axes[0].add_patch(
            plt.Rectangle(
                (0, -20), 50, 40, linewidth=1, edgecolor="red", facecolor="none"
            )
        )
        axes[0].set_title("Full Mesh")
        axes[0].set_xlabel("r / nm")
        axes[0].set_ylabel("z / nm")
        axes[0].set_aspect("equal")
        axes[0].set_xlim(0, r_max)
        axes[0].set_ylim(z_min, z_max)
        # zoomed mesh
        axes[1].triplot(triangulation, "k-", lw=0.5)
        axes[1].axhline(y=0, color="blue", linestyle="--", linewidth=1.5)
        axes[1].axhline(
            y=-geom_params.l_sio2, color="blue", linestyle="--", linewidth=1.5
        )
        axes[1].set_title("Zoomed Mesh (red box)")
        axes[1].set_xlabel("r / nm")
        axes[1].set_ylabel("z / nm")
        axes[1].set_aspect("equal")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("red")
        axes[1].text(
            10, 15, "tip", color="black", fontsize=10, ha="center", va="center"
        )
        axes[1].text(
            30, 5, "vacuum", color="black", fontsize=10, ha="center", va="center"
        )
        axes[1].text(
            30, -2.5, "SiO$_2$", color="black", fontsize=10, ha="center", va="center"
        )
        axes[1].text(
            30, -12.5, "SiC", color="black", fontsize=10, ha="center", va="center"
        )
        axes[1].set_xlim(0, 50)
        axes[1].set_ylim(-20, 20)
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, "mesh_plot.pdf"))
        plt.close(fig)
        print(f"Saved mesh plot to {os.path.join(subdir, 'mesh_plot.pdf')}")
        plt.rcParams.update(rc_params_stashed)

    # --- Line Profile Plots ---
    print("Creating line profile plot...")

    # Vertical Profile (along center axis r=0)
    num_points = 200
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
    fig2.savefig(os.path.join(subdir, "potential_line_profiles.png"), dpi=150)
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

    if plot_donor_ionization:
        # plot lineprofiles of donor ionization
        # calcurate Nd_h^+ and Nd_c^+ from u_dimless
        print("Creating donor ionization line profiles...")

        # Helper function to calculate ionized donor densities
        def calculate_donor_ionization(u_dimless_val, phys_params, V_c):
            """
            Calculate ionized donor densities from dimensionless potential.

            Parameters:
            - u_dimless_val: dimensionless potential value
            - phys_params: PhysicalParameters object
            - V_c: characteristic voltage (thermal voltage)

            Returns:
            - Ndp_h, Ndp_c, Ndp_total (all in m^-3)
            """
            u_volt = u_dimless_val * V_c  # Convert to volts
            kTeV = phys_params.kTeV

            # Calculate ionized donor densities using Fermi-Dirac statistics
            # Ndp = Nd / (1 + 2 * exp((Ef - Ed + e*u) / kT))
            # In energy: Ef - Ed + e*u (where e*u is the potential energy shift)
            Ndp_h = phys_params.Nd_h / (
                1 + 2 * np.exp((phys_params.Ef - phys_params.Edh + u_volt) / kTeV)
            )
            Ndp_c = phys_params.Nd_c / (
                1 + 2 * np.exp((phys_params.Ef - phys_params.Edc + u_volt) / kTeV)
            )
            Ndp_total = Ndp_h + Ndp_c

            return Ndp_h, Ndp_c, Ndp_total

        # --- Vertical Profile (along center axis r=0) ---
        z_coords_donor = np.linspace(z_min, -geom_params.l_sio2, num_points)
        Ndp_h_z = []
        Ndp_c_z = []
        Ndp_total_z = []
        valid_z_donor = []

        for z in z_coords_donor:
            try:
                # Evaluate potential at (r=0, z)
                u_val = u_dimless(msh(0, z))
                Ndp_h, Ndp_c, Ndp_total = calculate_donor_ionization(
                    u_val, phys_params, V_c
                )
                Ndp_h_z.append(Ndp_h)
                Ndp_c_z.append(Ndp_c)
                Ndp_total_z.append(Ndp_total)
                valid_z_donor.append(z)
            except Exception:
                continue

        # --- Horizontal Profile (at z = -l_sio2) ---
        r_coords_donor = np.linspace(0, r_max, num_points)
        z_level_donor = -geom_params.l_sio2
        Ndp_h_r = []
        Ndp_c_r = []
        Ndp_total_r = []
        valid_r_donor = []

        for r in r_coords_donor:
            try:
                # Evaluate potential at (r, z_level_donor)
                u_val = u_dimless(msh(r, z_level_donor))
                Ndp_h, Ndp_c, Ndp_total = calculate_donor_ionization(
                    u_val, phys_params, V_c
                )
                Ndp_h_r.append(Ndp_h)
                Ndp_c_r.append(Ndp_c)
                Ndp_total_r.append(Ndp_total)
                valid_r_donor.append(r)
            except Exception:
                continue

        # --- Plotting ---
        fig3, (ax3_z, ax3_r) = plt.subplots(2, 1, figsize=(8, 10))

        # Vertical Plot
        ax3_z.plot(
            valid_z_donor,
            np.array(Ndp_h_z) * 1e-6,
            label="$N_{D,h}^+$ (Hex)",
            color="blue",
            lw=1,
            ls="-",
        )
        ax3_z.plot(
            valid_z_donor,
            np.array(Ndp_c_z) * 1e-6,
            label="$N_{D,c}^+$ (Cubic)",
            color="red",
            lw=1,
            ls="-",
        )
        ax3_z.plot(
            valid_z_donor,
            np.array(Ndp_total_z) * 1e-6,
            label="$N_D^+$ (Total)",
            color="purple",
            lw=1.5,
            ls="-",
        )
        ax3_z.axhline(
            y=phys_params.Nd_h * 1e-6,
            color="blue",
            ls="--",
            lw=1,
            label="$N_{D,h}$ (Complete Ionization (Hex))",
        )
        ax3_z.axhline(
            y=phys_params.Nd_c * 1e-6,
            color="red",
            ls="--",
            lw=1,
            label="$N_{D,c}$ (Complete Ionization (Cubic))",
        )
        ax3_z.axhline(
            y=phys_params.Nd * 1e-6,
            color="purple",
            ls="--",
            lw=1.5,
            label="$N_D$ (Complete Ionization)",
        )
        ax3_z.set_xlabel("z (nm)")
        ax3_z.set_ylabel("Ionized Donor Density (cm$^{-3}$)")
        ax3_z.set_title("Donor Ionization Profile along Center Axis (r=0)")
        ax3_z.legend()
        ax3_z.grid(True, alpha=0.3)
        ax3_z.set_xlim(min(valid_z_donor), max(valid_z_donor))

        # Horizontal Plot
        ax3_r.plot(
            valid_r_donor,
            np.array(Ndp_h_r) * 1e-6,
            label="$N_{D,h}^+$ (Hex)",
            color="blue",
            lw=1,
            ls="-",
        )
        ax3_r.plot(
            valid_r_donor,
            np.array(Ndp_c_r) * 1e-6,
            label="$N_{D,c}^+$ (Cubic)",
            color="red",
            lw=1,
            ls="-",
        )
        ax3_r.plot(
            valid_r_donor,
            np.array(Ndp_total_r) * 1e-6,
            label="$N_D^+$ (Total)",
            color="purple",
            lw=1.5,
            ls="-",
        )
        ax3_r.axhline(
            y=phys_params.Nd_h * 1e-6,
            color="blue",
            ls="--",
            lw=1,
            label="$N_{D,h}$ (Complete Ionization (Hex))",
        )
        ax3_r.axhline(
            y=phys_params.Nd_c * 1e-6,
            color="red",
            ls="--",
            lw=1,
            label="$N_{D,c}$ (Complete Ionization (Cubic))",
        )
        ax3_r.axhline(
            y=phys_params.Nd * 1e-6,
            color="purple",
            ls="--",
            lw=1.5,
            label="$N_D$ (Complete Ionization)",
        )
        ax3_r.set_xlabel("r (nm)")
        ax3_r.set_ylabel("Ionized Donor Density (cm$^{-3}$)")
        ax3_r.set_title(
            f"Donor Ionization Profile at z = {z_level_donor:.1f} nm (SiC/SiO2 Interface)"
        )
        ax3_r.legend()
        ax3_r.grid(True, alpha=0.3)
        ax3_r.set_xlim(0, max(valid_r_donor))

        fig3.tight_layout()
        fig3.savefig(
            os.path.join(subdir, "donor_ionization_line_profiles.png"), dpi=150
        )
        plt.close(fig3)
        print(
            f"Saved donor ionization profiles to {os.path.join(subdir, 'donor_ionization_line_profiles.png')}"
        )

        # Save line profile data
        np.savetxt(
            os.path.join(subdir, "donor_ionization_vertical.txt"),
            np.column_stack(
                (
                    valid_z_donor,
                    np.array(Ndp_h_z) * 1e-6,
                    np.array(Ndp_c_z) * 1e-6,
                    np.array(Ndp_total_z) * 1e-6,
                )
            ),
            header="z_nm Ndp_h_cm-3 Ndp_c_cm-3 Ndp_total_cm-3",
        )
        np.savetxt(
            os.path.join(subdir, "donor_ionization_horizontal.txt"),
            np.column_stack(
                (
                    valid_r_donor,
                    np.array(Ndp_h_r) * 1e-6,
                    np.array(Ndp_c_r) * 1e-6,
                    np.array(Ndp_total_r) * 1e-6,
                )
            ),
            header="r_nm Ndp_h_cm-3 Ndp_c_cm-3 Ndp_total_cm-3",
        )
        print("Saved donor ionization data to text files.")

    if plot_charge_density:
        # Plot line profiles of charge density components (Nd+, n, p)
        print("Creating charge density line profiles...")

        # Helper function to calculate charge densities
        def calculate_charge_densities(
            u_dimless_val, phys_params, V_c, assume_full_ionization
        ):
            """
            Calculate charge densities from dimensionless potential.

            Parameters:
            - u_dimless_val: dimensionless potential value
            - phys_params: PhysicalParameters object
            - V_c: characteristic voltage (thermal voltage)
            - assume_full_ionization: whether to assume complete ionization

            Returns:
            - n, p, Ndp (all in m^-3)
            """
            u_volt = u_dimless_val * V_c  # Convert to volts
            kTeV = phys_params.kTeV

            # Calculate electron density using Fermi-Dirac statistics
            # n = Nc * F_{1/2}((Ef - Ec + e*u) / kT)
            eta_n = (phys_params.Ef - phys_params.Ec + u_volt) / kTeV

            # Use fermi_dirac_integral approximation from main.py
            # Aymerich-Humet approximation constants for F_{1/2}
            a1_aymerich = 6.316
            a2_aymerich = 12.92
            C_deg_aymerich = 0.75224956896  # 4 / (3 * sqrt(pi))

            if eta_n < -10.0:
                # Non-degenerate region (Boltzmann approximation)
                n_integral_approx = np.exp(eta_n)
            else:
                # Aymerich-Humet approximation (valid across the full range)
                eta_n_safe = max(eta_n, -4.0)
                G_inv_denominator = (
                    C_deg_aymerich
                    * (eta_n_safe**2 + a1_aymerich * eta_n_safe + a2_aymerich) ** 0.75
                )
                exp_neg_eta_n = np.exp(-eta_n)
                n_integral_approx = 1.0 / (exp_neg_eta_n + (1.0 / G_inv_denominator))

            n = phys_params.Nc * n_integral_approx

            # Calculate hole density using Fermi-Dirac statistics
            # p = Nv * F_{1/2}((Ev - Ef - e*u) / kT)
            eta_p = (phys_params.Ev - phys_params.Ef - u_volt) / kTeV
            if eta_p < -10.0:
                # 非縮退領域 (ボルツマン近似)
                # F_1/2(eta_p) \approx exp(eta_p)
                p_integral_approx = np.exp(eta_p)
            else:
                # Aymerich-Humet 近似 (全領域で有効)
                # F_1/2(eta_p) \approx 1.0 / (exp(-eta_p) + G(eta_p)^-1)
                eta_p_safe = max(eta_p, -4.0)
                G_inv_denominator = (
                    C_deg_aymerich
                    * (eta_p_safe**2 + a1_aymerich * eta_p_safe + a2_aymerich) ** 0.75
                )
                exp_neg_eta_p = np.exp(-eta_p)
                p_integral_approx = 1.0 / (exp_neg_eta_p + (1.0 / G_inv_denominator))

            p = phys_params.Nv * p_integral_approx

            # Calculate ionized donor density
            if assume_full_ionization:
                Ndp = phys_params.Nd  # Fully ionized
            else:
                # Ionized donor densities for each site
                Ndp_h = phys_params.Nd_h / (
                    1 + 2 * np.exp((phys_params.Ef - phys_params.Edh + u_volt) / kTeV)
                )
                Ndp_c = phys_params.Nd_c / (
                    1 + 2 * np.exp((phys_params.Ef - phys_params.Edc + u_volt) / kTeV)
                )
                Ndp = Ndp_h + Ndp_c

            return n, p, Ndp

        # --- Vertical Profile (along center axis r=0) ---
        z_coords_charge = np.linspace(z_min, -geom_params.l_sio2, num_points)
        n_z = []
        p_z = []
        Ndp_z = []
        valid_z_charge = []

        for z in z_coords_charge:
            try:
                # Evaluate potential at (r=0, z)
                u_val = u_dimless(msh(0, z))
                n, p, Ndp = calculate_charge_densities(
                    u_val, phys_params, V_c, assume_full_ionization
                )
                n_z.append(n)
                p_z.append(p)
                Ndp_z.append(Ndp)
                valid_z_charge.append(z)
            except Exception:
                continue

        # --- Horizontal Profile (at z = -l_sio2) ---
        r_coords_charge = np.linspace(0, r_max, num_points)
        z_level_charge = -geom_params.l_sio2
        n_r = []
        p_r = []
        Ndp_r = []
        valid_r_charge = []

        for r in r_coords_charge:
            try:
                # Evaluate potential at (r, z_level_charge)
                u_val = u_dimless(msh(r, z_level_charge))
                n, p, Ndp = calculate_charge_densities(
                    u_val, phys_params, V_c, assume_full_ionization
                )
                n_r.append(n)
                p_r.append(p)
                Ndp_r.append(Ndp)
                valid_r_charge.append(r)
            except Exception:
                continue

        # --- Plotting ---
        fig4, (ax4_z, ax4_r) = plt.subplots(2, 1, figsize=(8, 10))

        # Vertical Plot
        ax4_z.semilogy(
            valid_z_charge,
            np.array(n_z) * 1e-6,
            label="$n$ (Electron)",
            color="blue",
            lw=1.5,
            ls="-",
        )
        # ax4_z.semilogy(
        #     valid_z_charge,
        #     np.array(p_z) * 1e-6,
        #     label="$p$ (Hole)",
        #     color="red",
        #     lw=1.5,
        #     ls="-",
        # )
        ax4_z.semilogy(
            valid_z_charge,
            np.array(Ndp_z) * 1e-6,
            label="$N_D^+$ (Ionized Donor)",
            color="red",
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
        ax4_z.set_xlim(min(valid_z_charge), max(valid_z_charge))

        # Horizontal Plot
        ax4_r.semilogy(
            valid_r_charge,
            np.array(n_r) * 1e-6,
            label="$n$ (Electron)",
            color="blue",
            lw=1.5,
            ls="-",
        )
        # ax4_r.semilogy(
        #     valid_r_charge,
        #     np.array(p_r) * 1e-6,
        #     label="$p$ (Hole)",
        #     color="red",
        #     lw=1.5,
        #     ls="-",
        # )
        ax4_r.semilogy(
            valid_r_charge,
            np.array(Ndp_r) * 1e-6,
            label="$N_D^+$ (Ionized Donor)",
            color="red",
            lw=1.5,
            ls="-",
        )
        ax4_r.set_xlabel("r (nm)")
        ax4_r.set_ylabel("Charge Density (cm$^{-3}$)")
        ax4_r.set_title(
            f"Charge Density Profile at z = {z_level_charge:.1f} nm (SiC/SiO2 Interface)\n({ionization_status})"
        )
        ax4_r.legend()
        ax4_r.grid(True, which="both", alpha=0.3)
        ax4_r.set_xlim(0, max(valid_r_charge))

        fig4.tight_layout()
        fig4.savefig(os.path.join(subdir, "charge_density_line_profiles.png"), dpi=150)
        plt.close(fig4)
        print(
            f"Saved charge density profiles to {os.path.join(subdir, 'charge_density_line_profiles.png')}"
        )

        # Save line profile data
        np.savetxt(
            os.path.join(subdir, "charge_density_vertical.txt"),
            np.column_stack(
                (
                    valid_z_charge,
                    np.array(n_z) * 1e-6,
                    np.array(p_z) * 1e-6,
                    np.array(Ndp_z) * 1e-6,
                )
            ),
            header="z_nm n_cm-3 p_cm-3 Ndp_cm-3",
        )
        np.savetxt(
            os.path.join(subdir, "charge_density_horizontal.txt"),
            np.column_stack(
                (
                    valid_r_charge,
                    np.array(n_r) * 1e-6,
                    np.array(p_r) * 1e-6,
                    np.array(Ndp_r) * 1e-6,
                )
            ),
            header="r_nm n_cm-3 p_cm-3 Ndp_cm-3",
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
        print("No line profile data found for comparison plots.")
        return

    # Plot vertical comparison
    if all_data_vertical:
        fig_v, ax_v = plt.subplots(figsize=(10, 6))
        for V_tip, z, potential in all_data_vertical:
            ax_v.plot(z, potential, label=f"V_tip = {V_tip:+.2f} V", lw=1.5)
        ax_v.set_xlabel("z (nm)")
        ax_v.set_ylabel("Potential (V)")
        ax_v.set_title("Vertical Line Profiles Comparison (r=0)")
        ax_v.legend()
        ax_v.grid(True, alpha=0.3)
        fig_v.tight_layout()
        fig_v.savefig(
            os.path.join(comparison_dir, "potential_vertical_comparison.png"), dpi=150
        )
        plt.close(fig_v)
        print(
            f"Saved vertical comparison to {comparison_dir}/potential_vertical_comparison.png"
        )

    # Plot horizontal comparison
    if all_data_horizontal:
        fig_h, ax_h = plt.subplots(figsize=(10, 6))
        for V_tip, r, potential in all_data_horizontal:
            ax_h.plot(r, potential, label=f"V_tip = {V_tip:+.2f} V", lw=1.5)
        ax_h.set_xlabel("r (nm)")
        ax_h.set_ylabel("Potential (V)")
        ax_h.set_title("Horizontal Line Profiles Comparison (SiC/SiO2 Interface)")
        ax_h.legend()
        ax_h.grid(True, alpha=0.3)
        fig_h.tight_layout()
        fig_h.savefig(
            os.path.join(comparison_dir, "potential_horizontal_comparison.png"), dpi=150
        )
        plt.close(fig_h)
        print(
            f"Saved horizontal comparison to {comparison_dir}/potential_horizontal_comparison.png"
        )


def main():
    parser = argparse.ArgumentParser(description="Post-process NGSpy results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--plot_donor_ionization",
        action="store_true",
        help="Plot donor ionization profile",
    )
    parser.add_argument(
        "--plot_charge_density",
        action="store_true",
        help="Plot charge density profile",
    )
    parser.add_argument(
        "--plot_mesh", action="store_true", help="Plot mesh (for publication)"
    )
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="Skip comparison plots",
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

    # Get V_c from any metadata.json (should be the same for all)
    vtip_dirs = find_vtip_subdirs(out_dir)

    if not vtip_dirs:
        raise FileNotFoundError(f"No V_tip_±X.XXV subdirectories found in {out_dir}")

    print(f"Found {len(vtip_dirs)} V_tip directories:")
    for subdir, V_tip in vtip_dirs:
        print(f"  - {os.path.basename(subdir)} (V_tip = {V_tip:+.3f} V)")

    # Get V_c from first subdirectory
    first_metadata = os.path.join(vtip_dirs[0][0], "metadata.json")
    with open(first_metadata, "r") as f:
        metadata = json.load(f)
    V_c = metadata["V_c"]

    # Process each V_tip directory
    for subdir, V_tip in vtip_dirs:
        process_single_vtip(
            subdir=subdir,
            V_tip=V_tip,
            geom_params=geom_params,
            phys_params=phys_params,
            V_c=V_c,
            assume_full_ionization=assume_full_ionization,
            plot_donor_ionization=args.plot_donor_ionization,
            plot_charge_density=args.plot_charge_density,
            plot_mesh=args.plot_mesh,
        )

    # Create comparison plots
    if not args.no_comparison:
        create_comparison_plots(vtip_dirs, out_dir)

    print("\n" + "=" * 60)
    print("Post-processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
