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


logger = getLogger(__name__)


@dataclass
class PhysicalParameters:
    """Store physical parameters related to the simulation"""

    T: float = 300.0  # Temperature [K]
    Nd: float = 1e22  # Donor concentration [m^-3]
    Na: float = 0.0  # Acceptor concentration [m^-3]
    sigma_s: float = 1e15  # Surface charge density at SiC/SiO2 interface [m^-2]
    m_de: float = 0.42 * const.m_e
    m_dh: float = 1.0 * const.m_e
    Eg: float = 3.26  # Bandgap [eV]
    Ed_offset_hex: float = 0.124  # Donor level offset for hexagonal site [eV from Ec]
    Ed_offset_cub: float = 0.066  # Donor level offset for cubic site [eV from Ec]
    Ed_ratio_c_to_h: float = 1.88  # Ratio of cubic to hexagonal site
    Ea_offset: float = 0.2  # Acceptor level offset [eV from Ev]
    ni: float = 8.2e15  # Intrinsic carrier concentration [m^-3]
    eps_sic: float = 9.7
    eps_sio2: float = 3.9
    eps_vac: float = 1.0

    # Physical quantities derived from calculations
    n0: float = 0.0
    p0: float = 0.0
    Nc: float = 0.0
    Nv: float = 0.0
    Ec: float = 0.0
    Ev: float = 0.0
    Edh: float = 0.0  # Donor level (hex site)
    Edc: float = 0.0  # Donor level (cub site)
    Nd_h: float = 0.0  # Donor concentration at hexagonal sites [m^-3]
    Nd_c: float = 0.0  # Donor concentration at cubic sites [m^-3]
    kTeV: float = 0.0
    Ef: float = 0.0

    def __post_init__(self):
        """Initialize and compute physical constants"""
        self.kTeV = const.k * self.T / const.e
        # Distribute donor concentration based on site ratio
        total_ratio = 1.0 + self.Ed_ratio_c_to_h
        self.Nd_h = self.Nd * 1.0 / total_ratio
        self.Nd_c = self.Nd * self.Ed_ratio_c_to_h / total_ratio
        self.n0 = (self.Nd + np.sqrt(self.Nd**2 + 4 * self.ni**2)) / 2
        self.p0 = self.ni**2 / self.n0
        self.Nc = 2 * (2 * np.pi * self.m_de * const.k * self.T / (const.h**2)) ** 1.5
        self.Nv = 2 * (2 * np.pi * self.m_dh * const.k * self.T / (const.h**2)) ** 1.5
        self.Ev = 0
        self.Ec = self.Ev + self.Eg
        self.Edh = self.Ec - self.Ed_offset_hex
        self.Edc = self.Ec - self.Ed_offset_cub


@dataclass
class GeometricParameters:
    """Store geometric parameters related to the simulation"""

    L_c: float = 1e-9  # Characteristic length [1 nm]
    l_sio2: float = 5.0  # SiO2 layer thickness [nm]
    tip_radius: float = 45.0  # Tip curvature radius [nm]
    tip_height: float = 8.0  # Distance from tip to sample [nm]
    l_vac: float = 200.0  # Vacuum layer thickness [nm]
    region_radius: float = 500.0  # Calculation region radius [nm]
    n_tip_arc_points: int = 7  # Number of intermediate points in tip arc (odd)

    def __post_init__(self):
        assert self.n_tip_arc_points % 2 == 1, "n_tip_arc_points should be odd"


def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
    """Fermi-Dirac integral of half-integer order (j=1/2) approximation"""
    return np.piecewise(
        x,
        [x > 25],
        [
            lambda x: (2 / np.sqrt(np.pi))
            * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
            lambda x: np.exp(x) / (1 + 0.27 * np.exp(x)),
        ],
    )


def find_fermi_level(
    params: PhysicalParameters, out_dir: str, plot: bool = False
) -> float:
    """Calculate the Fermi level from the charge neutrality condition"""

    def charge_neutrality_eq(Ef: float) -> float:
        p = params.Nv * fermi_dirac_integral((params.Ev - Ef) / params.kTeV)
        n = params.Nc * fermi_dirac_integral((Ef - params.Ec) / params.kTeV)
        # Ionized donor density for each site
        Ndp_h = params.Nd_h / (1 + 2 * np.exp((Ef - params.Edh) / params.kTeV))
        Ndp_c = params.Nd_c / (1 + 2 * np.exp((Ef - params.Edc) / params.kTeV))
        Ndp = Ndp_h + Ndp_c
        # solve p + Ndp = n --> log(p + Ndp) = log(n)
        # log scale to avoid overflow and improve numerical stability
        return np.log(p + Ndp) - np.log(n)

    search_min = params.Ev + params.kTeV
    search_max = params.Ec - params.kTeV
    Ef, res = brentq(charge_neutrality_eq, search_min, search_max, full_output=True)
    if not res.converged:
        raise ValueError("Root finding for Fermi level did not converge")

    _plot_fermi_level_determination(params, Ef, out_dir)

    return Ef


def _plot_fermi_level_determination(
    params: PhysicalParameters, Ef: float, out_dir: str
):
    """Fermi level determination process plot helper function"""

    ee = np.linspace(params.Ev - 0.1, params.Ec + 0.1, 500)
    p = params.Nv * fermi_dirac_integral((params.Ev - ee) / params.kTeV)
    n = params.Nc * fermi_dirac_integral((ee - params.Ec) / params.kTeV)
    # Ionized donor density for each site
    Ndp_h = params.Nd_h / (1 + 2 * np.exp((ee - params.Edh) / params.kTeV))
    Ndp_c = params.Nd_c / (1 + 2 * np.exp((ee - params.Edc) / params.kTeV))
    Ndp = Ndp_h + Ndp_c

    plt.figure(figsize=(8, 6))
    plt.plot(ee, p + Ndp, label="Positive Charges ($p + N_D^+$)", color="blue", lw=2)
    plt.plot(ee, n, label="Negative Charges ($n$)", color="red", lw=2)
    plt.plot(
        ee,
        Ndp_h,
        label="$N_{D,h}^+$ (Hexagonal)",
        color="cyan",
        lw=1,
        ls="--",
        alpha=0.7,
    )
    plt.plot(
        ee, Ndp_c, label="$N_{D,c}^+$ (Cubic)", color="purple", lw=1, ls="--", alpha=0.7
    )
    plt.yscale("log")
    plt.title("Charge Concentrations vs. Fermi Level")
    plt.xlabel("Energy Level (E) / eV")
    plt.ylabel("Concentration / m$^{-3}$")
    plt.axvline(Ef, color="red", ls="-.", lw=1.5, label=f"Ef = {Ef:.2f} eV")
    plt.axvline(params.Ec, color="gray", ls=":", label="$E_c$")
    plt.axvline(params.Edh, color="cyan", ls="--", lw=1, label="$E_{d,h}$ (Hex)")
    plt.axvline(params.Edc, color="purple", ls="--", lw=1, label="$E_{d,c}$ (Cub)")
    plt.axvline(params.Ev, color="gray", ls=":", label="$E_v$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(out_dir, "fermi_level_determination.png"), dpi=150)
    plt.close()


def create_mesh(geom: GeometricParameters):
    """Use netgen to generate the geometry and mesh for the simulation region"""

    # Non-dimensionalization of geometry (representative length L_c)
    L_c = geom.L_c
    R_dimless = geom.region_radius * 1e-9 / L_c
    sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    sic_depth_dimless = (geom.l_vac - geom.l_sio2) * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_arc_angle = 75 * np.pi / 180  # Tip arc center angle
    n_middle_points = geom.n_tip_arc_points
    # Construct 2D axisymmetric geometry with SplineGeometry
    geo = SplineGeometry()

    # Define points (on center axis r=0)
    p1 = geo.AppendPoint(0, -sic_depth_dimless - sio2_depth_dimless)  # SiC bottom
    p2 = geo.AppendPoint(0, -sio2_depth_dimless)  # SiC/SiO2 interface
    origin = geo.AppendPoint(0, 0)  # SiO2/vacuum interface (origin)

    # Tip of the probe
    tip1 = geo.AppendPoint(0, tip_z_dimless)  # Tip bottom (arc start point)

    # Tip arc end point
    tip2 = geo.AppendPoint(
        tip_radius_dimless * np.sin(tip_arc_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.cos(tip_arc_angle)),
    )

    # Tip arc middle points (for spline3)
    tipMlst = [
        geo.AppendPoint(
            tip_radius_dimless * np.sin(mid_angle),
            tip_z_dimless + tip_radius_dimless * (1 - np.cos(mid_angle)),
        )
        for mid_angle in np.linspace(0, tip_arc_angle, n_middle_points + 2)[1:-1]
    ]

    # Tip cone end point
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

    # Far field boundary points
    q1 = geo.AppendPoint(
        R_dimless, -sic_depth_dimless - sio2_depth_dimless
    )  # SiC bottom right
    q2 = geo.AppendPoint(R_dimless, -sio2_depth_dimless)  # SiC/SiO2 interface right
    q3 = geo.AppendPoint(R_dimless, 0)  # SiO2/vacuum interface right
    q4 = geo.AppendPoint(R_dimless, vac_depth_dimless)  # Vacuum layer top

    # Boundary definitions (strictly follow the pattern in ref.py)
    # Important: Define all boundaries once, and specify shared boundaries with leftdomain/rightdomain

    # Bottom rectangle (SiC, domain=1): p1 → p2 → q2 → q1 → p1
    geo.Append(["line", p1, p2], bc="axis", leftdomain=0, rightdomain=1, maxh=5)
    geo.Append(["line", p2, q2], bc="sic/sio2", leftdomain=2, rightdomain=1, maxh=1)
    geo.Append(["line", q2, q1], bc="far-field", leftdomain=0, rightdomain=1)
    geo.Append(["line", q1, p1], bc="ground", leftdomain=0, rightdomain=1)

    # Middle rectangle (SiO2, domain=2): p2 → origin → q3 → q2
    # p2→q2 is already defined
    geo.Append(["line", origin, p2], bc="axis", leftdomain=2, rightdomain=0, maxh=0.5)
    geo.Append(["line", q2, q3], bc="far-field", leftdomain=2, rightdomain=0)
    geo.Append(["line", q3, origin], bc="sio2/vacuum", leftdomain=2, rightdomain=3)

    # Top with tip (vacuum, domain=3): origin → tip1 → tip2 → tip3 → q4 → q3
    geo.Append(["line", origin, tip1], bc="axis", leftdomain=0, rightdomain=3, maxh=0.5)
    for i in range(0, len(tipMlst), 2):
        points = [
            tip1 if i == 0 else tipMlst[i - 1],
            tipMlst[i],
            tip2 if i == len(tipMlst) - 1 else tipMlst[i + 1],
        ]
        geo.Append(
            ["spline3", *points], bc="tip", leftdomain=0, rightdomain=3, maxh=0.5
        )
    geo.Append(["line", tip2, tip3], bc="tip", leftdomain=0, rightdomain=3)
    geo.Append(["line", tip3, q4], bc="top", leftdomain=0, rightdomain=3)
    geo.Append(["line", q4, q3], bc="far-field", leftdomain=0, rightdomain=3)
    # O→q3 is already defined

    # Define materials
    geo.SetMaterial(1, "sic")
    geo.SetMaterial(2, "sio2")
    geo.SetMaterial(3, "vac")

    # Debug output of geometry information
    logger.info("Geometry defined with:")
    logger.info(f"  - Domain radius: {R_dimless:.2f}")
    logger.info(f"  - SiC depth: {sic_depth_dimless:.2f}")
    logger.info(f"  - SiO2 thickness: {sio2_depth_dimless:.2f}")
    logger.info(f"  - Vacuum height: {vac_depth_dimless:.2f}")
    logger.info(f"  - Tip radius: {tip_radius_dimless:.2f}")
    logger.info(f"  - Tip height: {tip_z_dimless:.2f}")

    # Check number of points and segments
    logger.info("Checking geometry integrity...")
    logger.info(
        f"  - Number of points defined: {len([p1, p2, origin, tip1, *tipMlst, tip2, tip3, q1, q2, q3, q4])}"
    )

    logger.info("Starting mesh generation...")
    logger.info("  (This may take a while for complex geometries...)")

    # Mesh size control

    # Set mesh size for each domain
    # geo.SetDomainMaxH(1, 5.0)  # SiC
    geo.SetDomainMaxH(2, 2)  # SiO2
    # geo.SetDomainMaxH(3, 1.0)  # Vacuum

    # Mesh generation (global maximum element size)
    # The area around the tip tends to become finer automatically
    ngmesh = geo.GenerateMesh(maxh=10, grading=0.2)
    logger.info("Mesh generation completed")

    # Convert to NGSolve mesh object
    mesh = ng.Mesh(ngmesh)

    logger.info(f"Mesh generated with {mesh.ne} elements and {mesh.nv} vertices")

    return mesh


def run_fem_simulation(
    phys: PhysicalParameters,
    geom: GeometricParameters,
    V_tip: float,
    Feenstra: bool,
    out_dir: str,
    assume_full_ionization: bool,
):
    """Run the FEM simulation using NGSolve"""

    msh = create_mesh(geom)
    L_c = geom.L_c
    V_c = const.k * phys.T / const.e  # Thermal voltage [V]

    # Function space and test/trial functions
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")

    # Define relative permittivity
    epsilon_r = ng.CoefficientFunction([phys.eps_sic, phys.eps_sio2, phys.eps_vac])

    # Homotopy parameters
    homotopy_charge = ng.Parameter(0.0)
    homotopy_sigma = ng.Parameter(0.0)

    # Define weak form
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

    # Set boundary conditions
    u.Set(0, definedon=msh.Boundaries("ground"))  # Dirichlet BC at ground
    u.Set(V_tip / V_c, definedon=msh.Boundaries("tip"))  # Dirichlet BC at tip

    # Compute initial solution with linear problem (Laplace equation)
    _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh)

    # Solve nonlinear problem with homotopy method
    solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma)

    save_results(msh, u, epsilon_r, V_c, Feenstra, out_dir)


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
    Edh_dimless = phys.Edh / V_c
    Edc_dimless = phys.Edc / V_c

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
                "Edh_dimless": Edh_dimless,
                "Edc_dimless": Edc_dimless,
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

    def fermi_dirac_half(x):
        x_clip = clamp(x, clip_exp)
        high = (2 / np.sqrt(np.pi)) * (
            (2 / 3) * x_clip**1.5 + (np.pi**2 / 12) * x_clip ** (-0.5)
        )
        low = safe_exp(x_clip) / (1 + 0.27 * safe_exp(x_clip))
        return ng.IfPos(x_clip - 25.0, high, low)

    u_clip = clamp(uh, clip_potential)

    # Nonlinear charge density terms (at SiC region)
    if Feenstra:
        n_term = C0 * phys.Nc * fermi_dirac_half((Ef_dimless - Ec_dimless) + u_clip)
        p_term = C0 * phys.Nv * fermi_dirac_half((Ev_dimless - Ef_dimless) - u_clip)
        if assume_full_ionization:
            # Boltzmann approximation (complete ionization)
            Ndp_term = C0 * phys.Nd  # Fully ionized
        else:
            # Ionized donor density for each site
            Ndp_h_term = (
                C0 * phys.Nd_h / (1 + 2 * safe_exp((Ef_dimless - Edh_dimless) + u_clip))
            )
            Ndp_c_term = (
                C0 * phys.Nd_c / (1 + 2 * safe_exp((Ef_dimless - Edc_dimless) + u_clip))
            )
            Ndp_term = Ndp_h_term + Ndp_c_term
    else:  # Boltzmann approximation (complete ionization)
        n_term = C0 * phys.n0 * safe_exp(u_clip)
        p_term = C0 * phys.p0 * safe_exp(-u_clip)
        Ndp_term = C0 * phys.Nd  # Fully ionized
    rho_dimless = homotopy_charge * (p_term + Ndp_term - n_term)

    sigma_s_dimless = homotopy_sigma * sigma_s_target

    a = ng.BilinearForm(fes, symmetric=False)
    a += epsilon_r * ng.grad(uh) * ng.grad(vh) * r * ng.dx
    a += epsilon_r * lambda_ff * uh * vh * r * ng.ds("far-field")
    a += -rho_dimless * vh * r * ng.dx(definedon=msh.Materials("sic"))
    a += -sigma_s_dimless * vh * r * ng.ds("sic/sio2")

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


def solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma):
    """Solve the nonlinear Poisson equation using homotopy method

    Stages:
      1. Increase homotopy_charge from 0 to 1 (space charge)
      2. Increase homotopy_sigma from 0 to 1 (interface charge)
    """

    homotopy_sigma.Set(0.0)
    _solve_homotopy_stage(a, u, fes, msh, homotopy_charge, "Space Charge")

    homotopy_charge.Set(1.0)
    _solve_homotopy_stage(a, u, fes, msh, homotopy_sigma, "Interface Charge")


def _solve_homotopy_stage(a, u, fes, msh, homotopy_param, stage_name: str):
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
        maxerr=1e-11,
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


def main():
    parser = argparse.ArgumentParser(
        description="2D Axisymmetric Poisson Solver for a Tip-on-Semiconductor System."
    )
    parser.add_argument(
        "--V_tip", type=float, default=2.0, help="Tip voltage in Volts."
    )
    parser.add_argument(
        "--tip_radius", type=float, default=45.0, help="Tip radius in nm."
    )
    parser.add_argument(
        "--tip_height", type=float, default=8.0, help="Tip-sample distance in nm."
    )
    parser.add_argument(
        "--l_sio2", type=float, default=5.0, help="Thickness of SiO2 layer in nm."
    )
    parser.add_argument(
        "--Nd", type=float, default=1e16, help="Donor concentration in cm^-3."
    )
    parser.add_argument(
        "--sigma_s",
        type=float,
        default=1e12,
        help="Surface charge density at SiC/SiO2 interface in cm^-2.",
    )
    parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin.")
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

    if args.model.lower().startswith("b") and args.assume_full_ionization:
        logger.warning(
            "Boltzmann model always assumes full ionization; --assume_full_ionization has no effect."
        )

    # Initialize physical parameters (convert units to m^-3, m^-2)
    phys_params = PhysicalParameters(
        T=args.T,
        Nd=args.Nd * 1e6,  # cm^-3 -> m^-3
        sigma_s=args.sigma_s * 1e4,  # cm^-2 -> m^-2
    )

    # Calculate Fermi level
    Ef = find_fermi_level(phys_params, args.out_dir, plot=args.plot_fermi)
    phys_params.Ef = Ef

    # Initialize geometric parameters
    geom_params = GeometricParameters(
        l_sio2=args.l_sio2, tip_radius=args.tip_radius, tip_height=args.tip_height
    )

    # Save parameters to JSON file
    all_params = {
        "physical": asdict(phys_params),
        "geometric": asdict(geom_params),
        "simulation": {
            "V_tip": args.V_tip,
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
        V_tip=args.V_tip,
        out_dir=args.out_dir,
        Feenstra=(args.model[0].upper() == "F"),
        assume_full_ionization=args.assume_full_ionization,
    )

    end = datetime.now()
    logger.info(f"Finished simulation at {end}, duration: {end - start}")


if __name__ == "__main__":
    main()
