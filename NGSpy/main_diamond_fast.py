import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from logging import INFO, basicConfig, getLogger

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
import scipy.constants as const
from netgen.geom2d import SplineGeometry
from ngsolve.solvers import Newton
from scipy.optimize import brentq

from fermi_dirac_integral import (
    F_half_aymerich_humet_ng as fermi_dirac_half,
    F_half_aymerich_humet_np as fermi_dirac_integral,
)


logger = getLogger(__name__)


@dataclass
class PhysicalParameters:
    """Store physical parameters for the diamond simulation"""

    T: float = 300.0  # Temperature [K]
    Na: float = 1.3e23  # Acceptor concentration [m^-3]
    sigma_s: float = 0.0  # Surface charge density at diamond/vacuum interface [m^-2]
    Eg: float = 5.5  # Bandgap [eV]
    Ea_offset: float = 0.37  # Acceptor level offset [eV from Ev]
    ni: float = 1e-21  # Intrinsic carrier concentration [m^-3]
    eps_diamond: float = 5.7
    eps_vac: float = 1.0
    g_a: float = 4.0  # Acceptor degeneracy factor
    m_e_longitudinal_ratio: float = 1.56  # Electron longitudinal mass / m0
    m_e_transverse_ratio: float = 0.28  # Electron transverse mass / m0
    m_h_heavy_ratio: float = 0.67  # Heavy hole mass / m0
    m_h_light_ratio: float = 0.26  # Light hole mass / m0

    # Derived quantities
    kTeV: float = 0.0
    Nc: float = 0.0
    Nv: float = 0.0
    Ec: float = 0.0
    Ev: float = 0.0
    Ea: float = 0.0
    Ef: float = 0.0
    n0: float = 0.0
    p0: float = 0.0
    Na_minus0: float = 0.0
    m_e_eff: float = 0.0
    m_h_eff: float = 0.0

    def __post_init__(self):
        """Initialize and compute physical constants"""
        self.kTeV = const.k * self.T / const.e

        m0 = const.m_e
        ml = self.m_e_longitudinal_ratio * m0
        mt = self.m_e_transverse_ratio * m0
        self.m_e_eff = (ml * mt**2) ** (1 / 3)
        self.m_h_eff = (
            self.m_h_heavy_ratio ** (3 / 2) + self.m_h_light_ratio ** (3 / 2)
        ) ** (2 / 3) * m0

        self.Nc = (
            6 * 2 * (2 * np.pi * self.m_e_eff * const.k * self.T / (const.h**2)) ** 1.5
        )
        # TODO: `6 *`: ダイヤモンドの伝導帯は6つの等価な谷を持つため

        self.Nv = (
            2 * (2 * np.pi * self.m_h_eff * const.k * self.T / (const.h**2)) ** 1.5
        )
        self.Ev = 0.0
        self.Ec = self.Ev + self.Eg
        self.Ea = self.Ev + self.Ea_offset

    def update_equilibrium_densities(self):
        """Update equilibrium carrier and ionized acceptor densities"""
        self.p0 = self.Nv * np.exp((self.Ev - self.Ef) / self.kTeV)
        self.n0 = self.Nc * np.exp((self.Ef - self.Ec) / self.kTeV)
        self.Na_minus0 = self.Na / (
            1 + self.g_a * np.exp((self.Ea - self.Ef) / self.kTeV)
        )


@dataclass
class GeometricParameters:
    """Store geometric parameters related to the simulation"""

    L_c: float = 1e-9  # Characteristic length [1 nm]
    diamond_thickness: float = 200.0  # Diamond layer thickness [nm]
    virtual_layer_thickness: float = 5.0  # Virtual layer thickness [nm]
    tip_radius: float = 45.0  # Tip curvature radius [nm]
    tip_height: float = 8.0  # Distance from tip to sample [nm]
    l_vac: float = 200.0  # Vacuum layer thickness [nm]
    region_radius: float = 500.0  # Calculation region radius [nm]
    n_tip_arc_points: int = 7  # Number of intermediate points in tip arc (odd)

    def __post_init__(self):
        assert self.n_tip_arc_points % 2 == 1, "n_tip_arc_points should be odd"


# def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
#     """Fermi-Dirac integral of half-integer order (j=1/2) approximation"""
#     return np.piecewise(
#         x,
#         [x > 25],
#         [
#             lambda x: (2 / np.sqrt(np.pi))
#             * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
#             lambda x: np.exp(x) / (1 + 0.27 * np.exp(x)),
#         ],
#     )


# def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
#     """
#     Fermi-Dirac integral of order 1/2 (j=1/2) using the approximation
#     by Aymerich-Humet et al. (1981).

#     This approximation is continuous and accurate across all regimes,
#     transitioning smoothly from the Boltzmann limit (exp(x) for x << 0)
#     to the degenerate limit (Sommerfeld expansion, first-order term).
#     """

#     # Aymerich-Humet et al. (1981) approximation parameters
#     a1 = 6.316
#     a2 = 12.92

#     # Pre-factor for the degenerate limit (4 / (3 * sqrt(pi)))
#     C_deg = 0.75224956896

#     result = np.piecewise(
#         x,
#         [x < -10.0],  # Non-degenerate regime
#         [
#             lambda x: np.exp(x),
#             lambda x: 1.0
#             / (np.exp(-x) + (C_deg * (x**2 + a1 * x + a2) ** 0.75) ** (-1.0)),
#         ],
#     )

#     return result


def find_fermi_level(
    params: PhysicalParameters, out_dir: str, plot: bool = False
) -> float:
    """Calculate the Fermi level from the charge neutrality condition"""

    def charge_neutrality_eq(Ef: float) -> float:
        p = params.Nv * fermi_dirac_integral((params.Ev - Ef) / params.kTeV)
        n = params.Nc * fermi_dirac_integral((Ef - params.Ec) / params.kTeV)
        Na_minus = params.Na / (1 + params.g_a * np.exp((params.Ea - Ef) / params.kTeV))
        return np.log(p) - np.log(n + Na_minus)

    search_min = params.Ev - 1.0
    search_max = params.Ec + 1.0
    Ef, res = brentq(charge_neutrality_eq, search_min, search_max, full_output=True)
    if not res.converged:
        raise ValueError("Root finding for Fermi level did not converge")

    if plot:
        _plot_fermi_level_determination(params, Ef, out_dir)

    return Ef


def _plot_fermi_level_determination(
    params: PhysicalParameters, Ef: float, out_dir: str
):
    """Fermi level determination process plot helper function"""

    ee = np.linspace(params.Ev - 0.1, params.Ec + 0.1, 500)
    p = params.Nv * fermi_dirac_integral((params.Ev - ee) / params.kTeV)
    n = params.Nc * fermi_dirac_integral((ee - params.Ec) / params.kTeV)
    Na_minus = params.Na / (1 + params.g_a * np.exp((params.Ea - ee) / params.kTeV))

    plt.figure(figsize=(8, 6))
    plt.plot(ee, p, label="Positive Charges ($p$)", color="blue", lw=2)
    plt.plot(
        ee,
        n + Na_minus,
        label="Negative Charges ($n + N_A^-$)",
        color="red",
        lw=2,
    )
    plt.plot(
        ee,
        Na_minus,
        label="$N_A^-$",
        color="purple",
        lw=1,
        ls="--",
        alpha=0.7,
    )
    plt.yscale("log")
    plt.title("Charge Concentrations vs. Fermi Level")
    plt.xlabel("Energy Level (E) / eV")
    plt.ylabel("Concentration / m$^{-3}$")
    plt.axvline(Ef, color="red", ls="-.", lw=1.5, label=f"Ef = {Ef:.2f} eV")
    plt.axvline(params.Ec, color="gray", ls=":", label="$E_c$")
    plt.axvline(params.Ea, color="purple", ls="--", lw=1, label="$E_a$")
    plt.axvline(params.Ev, color="gray", ls=":", label="$E_v$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(out_dir, "fermi_level_determination.png"), dpi=150)
    plt.close()


def create_mesh(geom: GeometricParameters):
    """Use netgen to generate the geometry and mesh for the simulation region"""

    L_c = geom.L_c
    R_dimless = geom.region_radius * 1e-9 / L_c
    diamond_depth_dimless = geom.diamond_thickness * 1e-9 / L_c
    virtual_layer_thickness_dimless = geom.virtual_layer_thickness * 1e-9 / L_c
    vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_arc_angle = 75 * np.pi / 180
    n_middle_points = geom.n_tip_arc_points

    geo = SplineGeometry()

    # Axis points
    p1 = geo.AppendPoint(0, -diamond_depth_dimless)
    p1m = geo.AppendPoint(0, -virtual_layer_thickness_dimless)
    origin = geo.AppendPoint(0, 0)

    # Tip geometry on axis
    tip1 = geo.AppendPoint(0, tip_z_dimless)
    tip2 = geo.AppendPoint(
        tip_radius_dimless * np.sin(tip_arc_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.cos(tip_arc_angle)),
    )
    tipMlst = [
        geo.AppendPoint(
            tip_radius_dimless * np.sin(mid_angle),
            tip_z_dimless + tip_radius_dimless * (1 - np.cos(mid_angle)),
        )
        for mid_angle in np.linspace(0, tip_arc_angle, n_middle_points + 2)[1:-1]
    ]
    tip3 = geo.AppendPoint(
        tip_radius_dimless * np.sin(tip_arc_angle)
        + (
            vac_depth_dimless
            - tip_z_dimless
            - tip_radius_dimless * (1 - np.cos(tip_arc_angle))
        )
        / np.tan(tip_arc_angle),
        vac_depth_dimless,
    )

    # Far-field points
    q1 = geo.AppendPoint(R_dimless, -diamond_depth_dimless)
    q1m = geo.AppendPoint(R_dimless, -virtual_layer_thickness_dimless)
    q2 = geo.AppendPoint(R_dimless, 0)
    q3 = geo.AppendPoint(R_dimless, vac_depth_dimless)

    # Diamond domain (material 1)
    geo.Append(["line", p1, p1m], bc="axis", leftdomain=0, rightdomain=1, maxh=5)
    geo.Append(
        ["line", p1m, q1m],
        bc="diamond/diamond-virtual",
        leftdomain=3,
        rightdomain=1,
        maxh=0.5,
    )
    geo.Append(["line", q1m, q1], bc="far-field", leftdomain=0, rightdomain=1)
    geo.Append(["line", q1, p1], bc="ground", leftdomain=0, rightdomain=1)

    # Diamond virtual layer near surface (material 3)
    geo.Append(["line", p1m, origin], bc="axis", leftdomain=0, rightdomain=3, maxh=0.5)
    geo.Append(
        ["line", origin, q2],
        bc="diamond-virtual/vacuum",
        leftdomain=2,
        rightdomain=3,
        maxh=0.5,
    )
    geo.Append(["line", q2, q1m], bc="far-field", leftdomain=0, rightdomain=3)

    # Vacuum domain (material 2)
    geo.Append(["line", origin, tip1], bc="axis", leftdomain=0, rightdomain=2, maxh=0.5)
    for i in range(0, len(tipMlst), 2):
        points = [
            tip1 if i == 0 else tipMlst[i - 1],
            tipMlst[i],
            tip2 if i == len(tipMlst) - 1 else tipMlst[i + 1],
        ]
        geo.Append(
            ["spline3", *points], bc="tip", leftdomain=0, rightdomain=2, maxh=0.5
        )
    geo.Append(["line", tip2, tip3], bc="tip", leftdomain=0, rightdomain=2)
    geo.Append(["line", tip3, q3], bc="top", leftdomain=0, rightdomain=2)
    geo.Append(["line", q3, q2], bc="far-field", leftdomain=0, rightdomain=2)

    geo.SetMaterial(1, "diamond")
    geo.SetMaterial(2, "vac")
    geo.SetMaterial(3, "diamond-virtual")

    logger.info("Geometry defined with:")
    logger.info(f"  - Domain radius: {R_dimless:.2f}")
    logger.info(f"  - Diamond depth: {diamond_depth_dimless:.2f}")
    logger.info(f"  - Vacuum height: {vac_depth_dimless:.2f}")
    logger.info(f"  - Tip radius: {tip_radius_dimless:.2f}")
    logger.info(f"  - Tip height: {tip_z_dimless:.2f}")
    logger.info(f"  - Virtual layer thickness: {virtual_layer_thickness_dimless:.2f}")

    logger.info("Checking geometry integrity...")
    logger.info(
        f"  - Number of points defined: {len([p1, origin, tip1, *tipMlst, tip2, tip3, q1, q2, q3])}"
    )

    logger.info("Starting mesh generation...")
    logger.info("  (This may take a while for complex geometries...)")

    # geo.SetDomainMaxH(1, 5.0)
    # geo.SetDomainMaxH(2, 2.0)
    geo.SetDomainMaxH(3, 0.5)

    ngmesh = geo.GenerateMesh(maxh=10, grading=0.2)
    logger.info("Mesh generation completed")

    mesh = ng.Mesh(ngmesh)

    logger.info(f"Mesh generated with {mesh.ne} elements and {mesh.nv} vertices")

    return mesh


def run_fem_simulation(
    phys: PhysicalParameters,
    geom: GeometricParameters,
    V_tip_values: list[float],
    Feenstra: bool,
    out_dir: str,
    assume_full_ionization: bool,
    maxerr=1e-11,
):
    """Run the FEM simulation using NGSolve for multiple V_tip values

    Optimizations:
    - Mesh is created once and reused for all V_tip values
    - V_tip values are sorted by absolute value for smooth convergence
    - For i > 0, use previous solution as warm start and try direct Newton
    - Fall back to partial homotopy if direct Newton fails
    """

    # Sort and validate voltages
    V_tip_sorted = sort_and_validate_voltages(V_tip_values)
    logger.info(f"Processing V_tip values in order: {V_tip_sorted}")

    # Create mesh once
    logger.info("Creating mesh (this will be reused for all V_tip values)...")
    msh = create_mesh(geom)
    L_c = geom.L_c
    V_c = const.k * phys.T / const.e  # Thermal voltage [V]

    # Function space and test/trial functions
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")

    # Define relative permittivity
    epsilon_r = ng.CoefficientFunction(
        [
            phys.eps_diamond,  # 1: diamond
            phys.eps_vac,  # 2: vacuum
            phys.eps_diamond,  # 3: diamond-virtual
        ]
    )

    # Homotopy parameters
    homotopy_charge = ng.Parameter(0.0)
    homotopy_sigma = ng.Parameter(0.0)

    # Define weak form (once, will be reused)
    a = _setup_weak_form(
        fes,
        epsilon_r,
        phys,
        V_c,
        L_c,
        homotopy_charge,
        homotopy_sigma,
        geom,
        msh,
        Feenstra,
        assume_full_ionization,
    )

    # Loop through V_tip values
    for i, V_tip in enumerate(V_tip_sorted):
        logger.info(f"{'=' * 60}")
        logger.info(f"Solving for V_tip = {V_tip:.3f} V ({i + 1}/{len(V_tip_sorted)})")
        logger.info(f"{'=' * 60}")

        # Set boundary conditions
        u.Set(0, definedon=msh.Boundaries("ground"))  # Dirichlet BC at ground
        u.Set(V_tip / V_c, definedon=msh.Boundaries("tip"))  # Dirichlet BC at tip

        if i == 0:
            # First voltage: full procedure (linear warm-start + homotopy)
            logger.info("First V_tip: performing full initialization...")
            _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh)
            solve_with_homotopy(
                a, u, fes, msh, homotopy_charge, homotopy_sigma, maxerr=maxerr
            )
        else:
            # Subsequent voltages: try direct Newton with fallback
            logger.info("Using previous solution as warm start...")
            solve_with_direct_newton(
                a, u, fes, msh, homotopy_charge, homotopy_sigma, maxerr=maxerr
            )

        # Save results in subdirectory
        out_subdir = os.path.join(out_dir, f"V_tip_{V_tip:+.2f}V")
        save_results(msh, u, epsilon_r, V_c, Feenstra, out_subdir)
        logger.info(f"Results saved to {out_subdir}")


def _setup_weak_form(
    fes,
    epsilon_r,
    phys,
    V_c,
    L_c,
    homotopy_charge,
    homotopy_sigma,
    geom,
    msh,
    Feenstra,
    assume_full_ionization,
):
    """Define the weak form of the Poisson equation with nonlinear charge density"""

    uh, vh = fes.TnT()
    r = ng.x

    # Coefficient for charge density calculation
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    Ef_dimless = phys.Ef / V_c
    Ec_dimless = phys.Ec / V_c
    Ev_dimless = phys.Ev / V_c
    Ea_dimless = phys.Ea / V_c

    lambda_ff = 1 / (geom.region_radius * 1e-9 / L_c)
    sigma_s_target = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)

    logger.info(
        json.dumps(
            {
                "L_c": L_c,
                "V_c": V_c,
                "C0": C0,
                "Ef_dimless": Ef_dimless,
                "Ec_dimless": Ec_dimless,
                "Ev_dimless": Ev_dimless,
                "Ea_dimless": Ea_dimless,
                "lambda_ff": lambda_ff,
                "sigma_s_target": sigma_s_target,
                "Feenstra": Feenstra,
            },
            indent=2,
        )
    )

    clip_potential = 120.0
    clip_exp = 40.0

    def clamp(val, bound):
        return ng.IfPos(val - bound, bound, ng.IfPos(-bound - val, -bound, val))

    def safe_exp(x):
        x_clip = clamp(x, clip_exp)
        return ng.exp(x_clip)

    # def fermi_dirac_half(x):
    #     x_clip = clamp(x, clip_exp)
    #     high = (2 / np.sqrt(np.pi)) * (
    #         (2 / 3) * x_clip**1.5 + (np.pi**2 / 12) * x_clip ** (-0.5)
    #     )
    #     low = safe_exp(x_clip) / (1 + 0.27 * safe_exp(x_clip))
    #     return ng.IfPos(x_clip - 25.0, high, low)

    # def fermi_dirac_half(x):
    #     """
    #     Fermi-Dirac integral of order 1/2 (j=1/2) using the approximation
    #     by Aymerich-Humet et al. (1981), implemented for NGSolve.
    #     """

    #     # Aymerich-Humet 近似 (F_1/2) のための定数
    #     a1 = 6.316
    #     a2 = 12.92
    #     C_deg = 0.75224956896  # 4 / (3 * np.sqrt(np.pi))

    #     # 1. 非縮退領域 (x < -10.0) の近似: F_1/2(x) \approx exp(x)
    #     #    safe_exp は x を clamp(x, clip_exp) してから ng.exp() する
    #     boltzmann_approx = safe_exp(x)

    #     # 2. 全領域の近似: F_1/2(x) \approx [exp(-x) + G(x)^-1]^-1

    #     # 2a. exp(-x) の項
    #     #     safe_exp(-x) は -x を clamp する (x を [-clip_exp, clip_exp] にクランプ)
    #     exp_neg_x = safe_exp(-x)

    #     # 2b. G(x)^-1 の項
    #     #     多項式 (x^2 + a1*x + a2) は x >= -4.0 で定義されているため
    #     #     x_safe = max(x, -4.0) を ng.IfPos で実装
    #     x_safe = ng.IfPos(x - (-4.0), x, -4.0)

    #     G_inv_denominator = C_deg * (x_safe**2 + a1 * x_safe + a2) ** 0.75

    #     # G(x)^-1 を計算 (多項式は x >= -4 で常に正なのでゼロ除算の心配はない)
    #     G_inv = G_inv_denominator ** (-1.0)

    #     # 2c. 全領域の近似式を結合
    #     full_approx = (exp_neg_x + G_inv) ** (-1.0)

    #     # 3. x = -10.0 を境に、非縮退近似と全領域近似を切り替える
    #     #    これにより、x が非常に小さい負の値のときの数値的安定性を確保する
    #     return ng.IfPos(x - (-10.0), full_approx, boltzmann_approx)

    u_clip = clamp(uh, clip_potential)

    # Nonlinear charge density terms (within the diamond region)
    if Feenstra:
        n_term = C0 * phys.Nc * fermi_dirac_half((Ef_dimless - Ec_dimless) + u_clip)
        p_term = C0 * phys.Nv * fermi_dirac_half((Ev_dimless - Ef_dimless) - u_clip)
        if assume_full_ionization:
            Na_term = C0 * phys.Na
        else:
            Na_term = (
                C0
                * phys.Na
                / (1 + phys.g_a * safe_exp((Ea_dimless - Ef_dimless) - u_clip))
            )
    else:
        n_term = C0 * phys.n0 * safe_exp(u_clip)
        p_term = C0 * phys.p0 * safe_exp(-u_clip)
        if assume_full_ionization:
            Na_term = C0 * phys.Na
        else:
            Na_term = (
                C0
                * phys.Na
                / (1 + phys.g_a * safe_exp((Ea_dimless - Ef_dimless) - u_clip))
            )
    rho_dimless = homotopy_charge * (p_term - n_term - Na_term)

    sigma_s_dimless = homotopy_sigma * sigma_s_target

    a = ng.BilinearForm(fes, symmetric=False)
    a += epsilon_r * ng.grad(uh) * ng.grad(vh) * r * ng.dx
    a += epsilon_r * lambda_ff * uh * vh * r * ng.ds("far-field")
    a += -rho_dimless * vh * r * ng.dx(definedon=msh.Materials("diamond"))
    a += -sigma_s_dimless * vh * r * ng.ds("diamond-virtual/vacuum")

    return a


def _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh):
    """Prepare initial guess for nonlinear solver by solving linear problem"""
    logger.info("Performing warm-start with linear Poisson equation...")

    # Define trial and test functions
    w = fes.TrialFunction()
    v = fes.TestFunction()

    # Radial coordinate
    r = ng.x

    # Far-field Robin boundary condition parameter
    lambda_ff = 1 / geom.region_radius

    # Set up bilinear and linear forms
    a_lin = ng.BilinearForm(fes)
    a_lin += epsilon_r * ng.grad(w) * ng.grad(v) * r * ng.dx
    a_lin += epsilon_r * lambda_ff * w * v * r * ng.ds("far-field")

    # Define linear form (right-hand side is zero)
    f_lin = ng.LinearForm(fes)

    # Initialize solution vector
    u.vec[:] = 0.0
    uh = ng.GridFunction(fes)

    # Warm-up by gradually increasing V_tip
    voltages_warmup = np.linspace(0.0, V_tip, 5)[1:]
    for v_val in voltages_warmup:
        # Set boundary conditions (initialization)
        uh = ng.GridFunction(fes)
        uh.Set(0, definedon=msh.Boundaries("ground"))
        uh.Set(v_val / V_c, definedon=msh.Boundaries("tip"))

        # Set up bilinear and linear forms
        a_lin.Assemble()
        f_lin.Assemble()

        # Modify right-hand side to account for Dirichlet BCs
        r = f_lin.vec.CreateVector()
        r.data = f_lin.vec - a_lin.mat * uh.vec

        # Determine free degrees of freedom
        freedofs = fes.FreeDofs()
        freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
        freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))

        # Dirichlet boundary conditions
        uh.vec.data += a_lin.mat.Inverse(freedofs, inverse="sparsecholesky") * r

        # Copy results to u
        u.vec.data = uh.vec

        logger.info(f"  [Linear Warm-up] Solved at V_tip = {v_val:.2f} V")


def solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma, maxerr=1e-11):
    """Solve the nonlinear Poisson equation using homotopy method

    Stages:
      1. Increase homotopy_charge from 0 to 1 (space charge)
      2. Increase homotopy_sigma from 0 to 1 (interface charge)
    """

    homotopy_sigma.Set(0.0)
    _solve_homotopy_stage(
        a, u, fes, msh, homotopy_charge, "Space Charge", maxerr=maxerr
    )

    homotopy_charge.Set(1.0)
    _solve_homotopy_stage(
        a, u, fes, msh, homotopy_sigma, "Interface Charge", maxerr=maxerr
    )


def solve_with_direct_newton(
    a, u, fes, msh, homotopy_charge, homotopy_sigma, fallback_theta=0.8, maxerr=1e-11
):
    """Solve directly with Newton method (homotopy already at 1.0)

    If Newton fails to converge, fall back to partial homotopy from fallback_theta to 1.0
    """
    logger.info("--- Attempting direct Newton solve (homotopy = 1.0) ---")

    homotopy_charge.Set(1.0)
    homotopy_sigma.Set(1.0)

    freedofs = fes.FreeDofs()
    freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
    freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))

    newton_kwargs = dict(
        freedofs=freedofs,
        maxit=100,
        maxerr=maxerr,
        inverse="sparsecholesky",
        dampfactor=0.7,
        printing=False,
    )

    a.Assemble()

    try:
        converged, iter = Newton(a, u, **newton_kwargs)
        if converged < 0:
            raise RuntimeError("Newton solver did not converge")
        logger.info(f"  [Direct Newton] Converged in {iter} iterations.")
        return
    except Exception as exc:
        logger.warning(f"  [Direct Newton] Failed: {exc}")
        logger.info(
            f"  Falling back to partial homotopy from θ={fallback_theta:.2f} to 1.0"
        )

        # Fallback: run homotopy from fallback_theta to 1.0 for both parameters
        backup = ng.GridFunction(fes)
        backup.vec.data = u.vec

        # First do charge homotopy
        homotopy_sigma.Set(fallback_theta)
        theta = fallback_theta
        step = 0.1
        min_step = 1e-4

        while theta < 1.0 - 1e-12:
            trial = min(1.0, theta + step)
            homotopy_sigma.Set(trial)
            a.Assemble()

            try:
                converged, iter = Newton(a, u, **newton_kwargs)
                if converged < 0:
                    raise RuntimeError("Newton solver did not converge")
                theta = trial
                backup.vec.data = u.vec
                logger.info(
                    f"  [Fallback Homotopy: θ={theta:.3f}] Converged in {iter} iterations."
                )
                if step < 0.5:
                    step *= 1.5
            except Exception:
                u.vec.data = backup.vec
                step *= 0.5
                logger.warning(
                    f"  [Fallback Homotopy: θ→{trial:.3f}] Failed. Reducing step to {step:.4f}."
                )
                if step < min_step:
                    raise RuntimeError(
                        "Fallback homotopy failed: step size became too small."
                    )


def _solve_homotopy_stage(
    a, u, fes, msh, homotopy_param, stage_name: str, maxerr=1e-11
):
    logger.info(f"--- Starting Homotopy Stage: {stage_name} ---")

    theta = 0.0
    step = 0.1
    min_step = 1e-4

    backup = ng.GridFunction(fes)
    backup.vec.data = u.vec

    freedofs = fes.FreeDofs()
    freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
    freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))

    newton_kwargs = dict(
        freedofs=freedofs,
        maxit=100,
        maxerr=maxerr,
        inverse="sparsecholesky",
        dampfactor=0.7,
        printing=False,
    )

    while theta < 1.0 - 1e-12:
        trial = min(1.0, theta + step)
        homotopy_param.Set(trial)
        a.Assemble()

        try:
            converged, iter = Newton(a, u, **newton_kwargs)
            if converged < 0:
                raise RuntimeError("Newton solver did not converge")
            theta = trial
            backup.vec.data = u.vec
            logger.info(
                f"  [{stage_name} Homotopy: θ={theta:.3f}] Converged in {iter} Newton iterations."
            )
            if step < 0.5:
                step *= 1.5
        except Exception as exc:
            u.vec.data = backup.vec
            step *= 0.5
            logger.warning(
                f"  [{stage_name} Homotopy: θ→{trial:.3f}] Failed ({exc}). Reducing step to {step:.4f}."
            )
            if step < min_step:
                raise RuntimeError(
                    f"Homotopy stage '{stage_name}' failed: step size became too small."
                )


def save_results(msh, u, epsilon_r, V_c, Feenstra: bool, out_dir: str):
    """Save simulation results to disk

    Saved files:
      mesh.vol            Netgen mesh
      solution.vtu        VTK (for Paraview etc.)
      u_dimless.npy       Dimensionless potential DOF vector
      u_volts.npy         Potential DOF vector in [V]
      epsilon_r.json      Relative permittivity for each material
      metadata.json       Metadata including characteristic scales
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save mesh in Netgen format
    msh.ngmesh.Save(os.path.join(out_dir, "mesh.vol"))

    u_np = u.vec.FV().NumPy()
    np.save(os.path.join(out_dir, "u_dimless.npy"), u_np)
    np.save(os.path.join(out_dir, "u_volts.npy"), u_np * V_c)

    # Save relative permittivity for each material in the order of material index
    mat_names = list(msh.GetMaterials())
    eps_map = (
        {
            name: val
            for name, val in zip(mat_names, [float(v) for v in epsilon_r.components])
        }
        if hasattr(epsilon_r, "components")
        else {name: None for name in mat_names}
    )
    with open(os.path.join(out_dir, "epsilon_r.json"), "w") as f:
        json.dump(eps_map, f, indent=2)

    # Save metadata
    meta = {
        "V_c": V_c,
        "Feenstra": Feenstra,
        "ndof": u.space.ndof,
        "fes": "H1",
        "order": u.space.globalorder,
        "materials": mat_names,
        "boundaries": list(msh.GetBoundaries()),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save VTK for visualization
    vtk = ng.VTKOutput(
        ma=msh,
        coefs=[u],
        names=["potential_dimless"],
        filename=os.path.join(out_dir, "solution"),
        subdivision=0,
    )
    vtk.Do()

    logger.info(f"Saved results to {out_dir}")


def load_results(out_dir: str, geom: GeometricParameters, V_c: float):
    """
    Reload saved simulation results.

    Steps:
      1. Recreate the mesh with create_mesh(geom) (assuming parameters are unchanged)
      2. Check that the number of degrees of freedom (ndof) matches
      3. Assign the DOF vector to the solution
    Returns: (mesh, u, u_volts_numpy)

    Usage:
    >>> # After saving
    >>> save_results(msh, u, epsilon_r, V_c, "out")
    >>> # In a separate process/post-processing script
    >>> geom = GeometricParameters()
    >>> msh2, u_loaded, uV = load_results("out", geom, V_c)
    """
    # Recreate mesh
    msh = create_mesh(geom)
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")

    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("No metadata.json found in the specified directory")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["ndof"] != fes.ndof:
        raise RuntimeError(f"DOF mismatch: saved={meta['ndof']} current={fes.ndof}")

    u_dim_path = os.path.join(out_dir, "u_dimless.npy")
    if not os.path.isfile(u_dim_path):
        raise FileNotFoundError("No u_dimless.npy found in the specified directory")

    u_vec = np.load(u_dim_path)
    if u_vec.size != fes.ndof:
        raise RuntimeError("Vector length does not match current FES")

    u.vec.FV().NumPy()[:] = u_vec

    # Convert to volts
    u_volts = u_vec * V_c

    logger.info("Loaded solution from disk")
    return msh, u, u_volts


def parse_range_input(input_str: str) -> list[float]:
    if ":" in input_str:
        parts = input_str.split(":")
        if len(parts) != 3:
            raise ValueError("Range input must be in the format min:max:step")
        min_val = float(parts[0])
        max_val = float(parts[1])
        step = float(parts[2])
        return list(np.arange(min_val, max_val + step, step))
    else:
        return [float(input_str)]


def sort_and_validate_voltages(V_tip_values: list[float]) -> list[float]:
    """Sort voltages by absolute value and validate they don't cross zero"""
    if len(V_tip_values) == 0:
        raise ValueError("V_tip_values cannot be empty")

    # Check if range crosses zero
    has_positive = any(v > 0 for v in V_tip_values)
    has_negative = any(v < 0 for v in V_tip_values)

    if has_positive and has_negative:
        raise ValueError(
            f"V_tip range crosses zero (values: {V_tip_values}). "
            "Please specify separate ranges for positive and negative voltages."
        )

    # Sort by absolute value
    return sorted(V_tip_values, key=abs)


def main():
    parser = argparse.ArgumentParser(
        description="2D Axisymmetric Poisson Solver for a Tip-on-Semiconductor System."
    )
    parser.add_argument(
        "--V_tip",
        type=str,
        default="2.0",
        help="Tip voltages (single value or range min:max:step).",
    )
    parser.add_argument(
        "--tip_radius", type=float, default=45.0, help="Tip radius in nm."
    )
    parser.add_argument(
        "--tip_height", type=float, default=8.0, help="Tip-sample distance in nm."
    )
    parser.add_argument(
        "--diamond_thickness",
        type=float,
        default=200.0,
        help="Thickness of the diamond layer in nm.",
    )
    parser.add_argument(
        "--sigma_s",
        type=float,
        default=0.0,
        help="Surface charge density at the diamond/vacuum interface in cm^-2.",
    )
    parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin.")
    parser.add_argument(
        "--Na",
        type=float,
        default=6e16,
        help="Acceptor concentration (boron) in cm^-3.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="out", help="Output directory for results."
    )
    parser.add_argument(
        "--plot_fermi",
        action="store_true",
        help="Plot the Fermi level determination process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Feenstra", "Boltzmann", "F", "B"],
        default="Feenstra",
        help="Choose the carrier statistics model.",
    )
    parser.add_argument(
        "--assume_full_ionization",
        action="store_true",
        help="Assume complete ionization of donors (overrides model to Boltzmann).",
    )
    parser.add_argument(
        "--maxerr",
        type=float,
        default=1e-11,
        help="Maximum error tolerance for Newton solver.",
    )
    args, _ = parser.parse_known_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    basicConfig(
        level=INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
        filename=os.path.join(args.out_dir, "main.log"),
    )

    # Also log errors to the file via the root logger's exception hook
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = log_exception

    print("Logging to", os.path.join(args.out_dir, "main.log"))
    start = datetime.now()
    logger.info(f"Started simulation at {start}")

    # Initialize physical parameters (convert units to m^-3, m^-2)
    phys_params = PhysicalParameters(
        T=args.T,
        Na=args.Na * 1e6,  # cm^-3 -> m^-3
        sigma_s=args.sigma_s * 1e4,  # cm^-2 -> m^-2
    )

    # Calculate Fermi level
    Ef = find_fermi_level(phys_params, args.out_dir, plot=args.plot_fermi)
    phys_params.Ef = Ef
    phys_params.update_equilibrium_densities()

    # Initialize geometric parameters
    geom_params = GeometricParameters(
        diamond_thickness=args.diamond_thickness,
        tip_radius=args.tip_radius,
        tip_height=args.tip_height,
    )

    V_tip_values = parse_range_input(args.V_tip)

    # Save parameters to JSON file
    all_params = {
        "physical": asdict(phys_params),
        "geometric": asdict(geom_params),
        "simulation": {
            "V_tips": V_tip_values,
            "model": args.model,
            "assume_full_ionization": args.assume_full_ionization,
        },
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "parameters.json"), "w") as f:
        json.dump(all_params, f, indent=2)

    # Run FEM simulation
    run_fem_simulation(
        phys=phys_params,
        geom=geom_params,
        V_tip_values=V_tip_values,
        out_dir=args.out_dir,
        Feenstra=(args.model[0].upper() == "F"),
        assume_full_ionization=args.assume_full_ionization,
        maxerr=args.maxerr,
    )

    end = datetime.now()
    logger.info(f"Finished simulation at {end}, duration: {end - start}")


if __name__ == "__main__":
    main()
