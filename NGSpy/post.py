import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from ngsolve import VOL

from main import GeometricParameters, PhysicalParameters, load_results


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
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir
    plot_donor_ionization = args.plot_donor_ionization
    plot_charge_density = args.plot_charge_density

    print(f"Post-processing results in {out_dir}...")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")

    # load params
    with open(os.path.join(out_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join(out_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    geom_params_input = params["geometric"]
    phys_params_input = params["physical"]
    V_c = metadata["V_c"]
    V_tip = params["simulation"]["V_tip"]
    assume_full_ionization = params["simulation"].get("assume_full_ionization", False)
    geom_params = GeometricParameters(**geom_params_input)
    phys_params = PhysicalParameters(**phys_params_input)
    L_c = geom_params.L_c  # [m]
    msh, u_dimless, _ = load_results(out_dir, geom_params, V_c)

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
    fig1.savefig(os.path.join(out_dir, "potential_2d_plot.png"), dpi=300)
    print(f"Saved 2D plot to {os.path.join(out_dir, 'potential_2d_plot.png')}")

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
    fig2.savefig(os.path.join(out_dir, "potential_line_profiles.png"), dpi=150)
    print(
        f"Saved line profiles to {os.path.join(out_dir, 'potential_line_profiles.png')}"
    )

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
            os.path.join(out_dir, "donor_ionization_line_profiles.png"), dpi=150
        )
        print(
            f"Saved donor ionization profiles to {os.path.join(out_dir, 'donor_ionization_line_profiles.png')}"
        )

        # Save line profile data
        np.savetxt(
            os.path.join(out_dir, "donor_ionization_vertical.txt"),
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
            os.path.join(out_dir, "donor_ionization_horizontal.txt"),
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
        def calculate_charge_densities(u_dimless_val, phys_params, V_c, assume_full_ionization):
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
            if eta_n > 25:
                n = phys_params.Nc * (2 / np.sqrt(np.pi)) * (
                    (2 / 3) * eta_n**1.5 + (np.pi**2 / 12) * eta_n**(-0.5)
                )
            else:
                n = phys_params.Nc * np.exp(eta_n) / (1 + 0.27 * np.exp(eta_n))

            # Calculate hole density using Fermi-Dirac statistics
            # p = Nv * F_{1/2}((Ev - Ef - e*u) / kT)
            eta_p = (phys_params.Ev - phys_params.Ef - u_volt) / kTeV
            if eta_p > 25:
                p = phys_params.Nv * (2 / np.sqrt(np.pi)) * (
                    (2 / 3) * eta_p**1.5 + (np.pi**2 / 12) * eta_p**(-0.5)
                )
            else:
                p = phys_params.Nv * np.exp(eta_p) / (1 + 0.27 * np.exp(eta_p))

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
        ionization_status = "Full Ionization" if assume_full_ionization else "Partial Ionization"
        ax4_z.set_title(f"Charge Density Profile along Center Axis (r=0)\n({ionization_status})")
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
        fig4.savefig(
            os.path.join(out_dir, "charge_density_line_profiles.png"), dpi=150
        )
        print(
            f"Saved charge density profiles to {os.path.join(out_dir, 'charge_density_line_profiles.png')}"
        )

        # Save line profile data
        np.savetxt(
            os.path.join(out_dir, "charge_density_vertical.txt"),
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
            os.path.join(out_dir, "charge_density_horizontal.txt"),
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


if __name__ == "__main__":
    main()
