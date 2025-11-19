import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from logging import INFO, basicConfig, getLogger

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

# NOTE:
# 1D planar model for SiC | SiO2 | metal structure
# Based on main_fast.py but simplified to 1D geometry

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
class GeometricParameters1D:
    """Store geometric parameters for 1D planar structure"""

    L_c: float = 1e-9  # Characteristic length [1 nm]
    d_sic: float = 100.0  # SiC layer thickness [nm]
    d_sio2: float = 5.0  # SiO2 layer thickness [nm]


def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
    """
    Fermi-Dirac integral of order 1/2 (j=1/2) using the approximation
    by Aymerich-Humet et al. (1981).

    This approximation is continuous and accurate across all regimes,
    transitioning smoothly from the Boltzmann limit (exp(x) for x << 0)
    to the degenerate limit (Sommerfeld expansion, first-order term).
    """

    # Aymerich-Humet et al. (1981) approximation parameters
    a1 = 6.316
    a2 = 12.92

    # Pre-factor for the degenerate limit (4 / (3 * sqrt(pi)))
    C_deg = 0.75224956896

    result = np.piecewise(
        x,
        [x < -10.0],  # Non-degenerate regime
        [
            lambda x: np.exp(x),
            lambda x: 1.0
            / (np.exp(-x) + (C_deg * (x**2 + a1 * x + a2) ** 0.75) ** (-1.0)),
        ],
    )

    return result


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


def create_grid_1d(geom: GeometricParameters1D, n_sic: int = 200, n_sio2: int = 50):
    """
    Create 1D grid for SiC | SiO2 structure
    
    Coordinate system:
        x ∈ [-d_sic, 0]: SiC region
        x ∈ [0, d_sio2]: SiO2 region
        x = 0: SiC/SiO2 interface
    
    Returns:
        x: grid points (dimensionless)
        mask_sic: boolean array indicating SiC region
        mask_sio2: boolean array indicating SiO2 region
    """
    L_c = geom.L_c
    d_sic_dimless = geom.d_sic * 1e-9 / L_c
    d_sio2_dimless = geom.d_sio2 * 1e-9 / L_c
    
    # Create grid with higher density near interface
    x_sic = -d_sic_dimless * (1 - np.linspace(0, 1, n_sic)**2)  # Denser near 0
    x_sio2 = d_sio2_dimless * np.linspace(0, 1, n_sio2)**2  # Denser near 0
    
    # Combine grids (remove duplicate at interface)
    x = np.concatenate([x_sic, x_sio2[1:]])
    
    # Create material masks
    mask_sic = x < 0
    mask_sio2 = x >= 0
    
    logger.info(f"Created 1D grid: {len(x)} points")
    logger.info(f"  SiC region: {n_sic} points, x ∈ [{-d_sic_dimless:.2f}, 0]")
    logger.info(f"  SiO2 region: {n_sio2} points, x ∈ [0, {d_sio2_dimless:.2f}]")
    
    return x, mask_sic, mask_sio2


def compute_charge_density(
    phi: np.ndarray,
    phys: PhysicalParameters,
    V_c: float,
    homotopy_charge: float,
    Feenstra: bool,
    assume_full_ionization: bool,
) -> np.ndarray:
    """
    Compute charge density in SiC region
    
    Args:
        phi: dimensionless potential
        phys: physical parameters
        V_c: characteristic voltage [V]
        homotopy_charge: homotopy parameter for space charge (0 to 1)
        Feenstra: if True, use Fermi-Dirac statistics; else Boltzmann
        assume_full_ionization: if True, assume complete ionization
    
    Returns:
        rho: charge density [dimensionless]
    """
    L_c = 1e-9  # nm
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    Ef_dimless = phys.Ef / V_c
    Ec_dimless = phys.Ec / V_c
    Ev_dimless = phys.Ev / V_c
    Edh_dimless = phys.Edh / V_c
    Edc_dimless = phys.Edc / V_c
    
    # Clip potential to avoid overflow
    phi_clip = np.clip(phi, -120, 120)
    
    if Feenstra:
        # Fermi-Dirac statistics
        n = C0 * phys.Nc * fermi_dirac_integral((Ef_dimless - Ec_dimless) + phi_clip)
        p = C0 * phys.Nv * fermi_dirac_integral((Ev_dimless - Ef_dimless) - phi_clip)
        
        if assume_full_ionization:
            Ndp = C0 * phys.Nd
        else:
            # Ionized donor density for each site
            exp_arg_h = np.clip((Ef_dimless - Edh_dimless) + phi_clip, -40, 40)
            exp_arg_c = np.clip((Ef_dimless - Edc_dimless) + phi_clip, -40, 40)
            Ndp_h = C0 * phys.Nd_h / (1 + 2 * np.exp(exp_arg_h))
            Ndp_c = C0 * phys.Nd_c / (1 + 2 * np.exp(exp_arg_c))
            Ndp = Ndp_h + Ndp_c
    else:
        # Boltzmann approximation
        exp_arg_pos = np.clip(phi_clip, -40, 40)
        exp_arg_neg = np.clip(-phi_clip, -40, 40)
        n = C0 * phys.n0 * np.exp(exp_arg_pos)
        p = C0 * phys.p0 * np.exp(exp_arg_neg)
        Ndp = C0 * phys.Nd
    
    rho = homotopy_charge * (p + Ndp - n)
    
    return rho


def setup_bvp_system(
    phys: PhysicalParameters,
    geom: GeometricParameters1D,
    V_tip: float,
    homotopy_charge: float,
    homotopy_sigma: float,
    Feenstra: bool,
    assume_full_ionization: bool,
):
    """
    Set up the boundary value problem for scipy.solve_bvp
    
    The system is formulated as:
        y[0] = φ (potential)
        y[1] = dφ/dx (electric field)
    
    dy/dx = f(x, y) where:
        dy[0]/dx = y[1]
        dy[1]/dx = -ρ(y[0])/ε in SiC
        dy[1]/dx = 0 in SiO2
    """
    V_c = const.k * phys.T / const.e
    L_c = geom.L_c
    d_sic_dimless = geom.d_sic * 1e-9 / L_c
    d_sio2_dimless = geom.d_sio2 * 1e-9 / L_c
    
    # Surface charge density (dimensionless)
    sigma_s_dimless = homotopy_sigma * (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)
    
    def ode_func(x, y):
        """
        ODE system: dy/dx = f(x, y)
        y[0] = φ, y[1] = dφ/dx
        """
        phi = y[0]
        dphidx = y[1]
        
        # Determine which region we're in
        if np.isscalar(x):
            in_sic = x < 0
        else:
            in_sic = x < 0
        
        # Compute d²φ/dx²
        if np.isscalar(x):
            if in_sic:
                # SiC region: Poisson-Boltzmann
                rho = compute_charge_density(
                    np.array([phi]), phys, V_c, homotopy_charge, 
                    Feenstra, assume_full_ionization
                )[0]
                d2phidx2 = -rho / phys.eps_sic
            else:
                # SiO2 region: Laplace equation
                d2phidx2 = 0.0
        else:
            d2phidx2 = np.zeros_like(x)
            # SiC region
            rho_sic = compute_charge_density(
                phi[in_sic], phys, V_c, homotopy_charge,
                Feenstra, assume_full_ionization
            )
            d2phidx2[in_sic] = -rho_sic / phys.eps_sic
            # SiO2 region: d2phidx2 = 0 (already initialized)
        
        return np.vstack([dphidx, d2phidx2])
    
    def bc_func(ya, yb):
        """
        Boundary conditions:
        ya: y at x = -d_sic (ground)
        yb: y at x = d_sio2 (metal electrode)
        """
        # Left boundary: φ(-d_sic) = 0
        bc_left = ya[0] - 0.0
        
        # Right boundary: φ(d_sio2) = V_tip / V_c
        bc_right = yb[0] - V_tip / V_c
        
        return np.array([bc_left, bc_right])
    
    def bc_func_with_interface(ya, yb, y_interface_left, y_interface_right):
        """
        Extended boundary conditions including interface condition
        This is a more complex formulation if needed
        """
        # Standard BCs
        bc_left = ya[0] - 0.0
        bc_right = yb[0] - V_tip / V_c
        
        # Interface condition: ε_sic * E_left - ε_sio2 * E_right = -e*σ_s / ε_0
        # In dimensionless form: ε_sic * dφ/dx|₀₋ - ε_sio2 * dφ/dx|₀₊ = -σ_s_dimless
        bc_interface = (phys.eps_sic * y_interface_left[1] - 
                       phys.eps_sio2 * y_interface_right[1] + 
                       sigma_s_dimless)
        
        return np.array([bc_left, bc_right, bc_interface])
    
    return ode_func, bc_func, V_c, sigma_s_dimless


def solve_linear_initial_guess(
    geom: GeometricParameters1D,
    phys: PhysicalParameters,
    V_tip: float,
    x: np.ndarray,
    mask_sic: np.ndarray,
    mask_sio2: np.ndarray,
) -> np.ndarray:
    """
    Solve linear Poisson equation (ρ = 0) for initial guess
    
    Solution for ρ = 0:
        SiC: d²φ/dx² = 0 → φ = ax + b
        SiO2: d²φ/dx² = 0 → φ = cx + d
    
    With interface condition (continuity + flux jump with σ_s):
        φ(0-) = φ(0+)
        ε_sic * dφ/dx|₀₋ - ε_sio2 * dφ/dx|₀₊ = -e*σ_s/ε_0
    """
    L_c = geom.L_c
    V_c = const.k * phys.T / const.e
    d_sic_dimless = geom.d_sic * 1e-9 / L_c
    d_sio2_dimless = geom.d_sio2 * 1e-9 / L_c
    V_tip_dimless = V_tip / V_c
    
    sigma_s_dimless = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)
    
    # Solve for linear case:
    # φ_sic(x) = a1*x + b1 for x ∈ [-d_sic, 0]
    # φ_sio2(x) = a2*x + b2 for x ∈ [0, d_sio2]
    
    # Boundary conditions:
    # φ(-d_sic) = 0 → -a1*d_sic + b1 = 0 → b1 = a1*d_sic
    # φ(d_sio2) = V_tip → a2*d_sio2 + b2 = V_tip
    
    # Interface conditions:
    # φ(0-) = φ(0+) → b1 = b2
    # ε_sic*a1 - ε_sio2*a2 = -σ_s
    
    # From these equations:
    # b1 = b2 and b1 = a1*d_sic
    # So: a1*d_sic = b2
    # From right BC: a2*d_sio2 + b2 = V_tip → a2*d_sio2 + a1*d_sic = V_tip
    # From interface: ε_sic*a1 - ε_sio2*a2 = -σ_s
    
    # Solve the 2x2 system:
    # [d_sio2, d_sic    ] [a2]   [V_tip  ]
    # [-ε_sio2, ε_sic   ] [a1] = [-σ_s   ]
    
    A = np.array([
        [d_sio2_dimless, d_sic_dimless],
        [-phys.eps_sio2, phys.eps_sic]
    ])
    b = np.array([V_tip_dimless, -sigma_s_dimless])
    
    a2, a1 = np.linalg.solve(A, b)
    b1 = a1 * d_sic_dimless
    b2 = b1
    
    # Construct solution
    phi_init = np.zeros_like(x)
    phi_init[mask_sic] = a1 * x[mask_sic] + b1
    phi_init[mask_sio2] = a2 * x[mask_sio2] + b2
    
    logger.info(f"Linear initial guess: φ_interface = {b1:.4f}, E_sic = {a1:.4f}, E_sio2 = {a2:.4f}")
    
    return phi_init


def solve_with_homotopy_1d(
    phys: PhysicalParameters,
    geom: GeometricParameters1D,
    V_tip: float,
    x: np.ndarray,
    mask_sic: np.ndarray,
    mask_sio2: np.ndarray,
    Feenstra: bool,
    assume_full_ionization: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve nonlinear Poisson equation using homotopy method
    
    Stage 1: Increase space charge from 0 to 1
    Stage 2: Increase interface charge from 0 to 1
    
    Returns:
        phi: potential solution (dimensionless)
        E: electric field -dφ/dx (dimensionless)
    """
    V_c = const.k * phys.T / const.e
    
    # Initial guess from linear solution
    phi_init = solve_linear_initial_guess(geom, phys, V_tip, x, mask_sic, mask_sio2)
    E_init = -np.gradient(phi_init, x)
    y_init = np.vstack([phi_init, E_init])
    
    # Stage 1: Space charge homotopy (σ_s = 0)
    logger.info("--- Stage 1: Space charge homotopy ---")
    theta = 0.0
    step = 0.1
    min_step = 1e-3
    
    current_y = y_init
    
    while theta < 1.0 - 1e-12:
        trial = min(1.0, theta + step)
        
        ode_func, bc_func, _, _ = setup_bvp_system(
            phys, geom, V_tip, trial, 0.0, Feenstra, assume_full_ionization
        )
        
        try:
            sol = solve_bvp(ode_func, bc_func, x, current_y, max_nodes=5000, tol=1e-6)
            
            if not sol.success:
                raise RuntimeError(sol.message)
            
            theta = trial
            current_y = sol.sol(x)
            logger.info(f"  [Space charge: θ={theta:.3f}] Converged. RMS residual: {sol.rms_residuals.max():.2e}")
            
            if step < 0.5:
                step *= 1.5
                
        except Exception as exc:
            step *= 0.5
            logger.warning(f"  [Space charge: θ→{trial:.3f}] Failed ({exc}). Reducing step to {step:.4f}.")
            if step < min_step:
                raise RuntimeError("Space charge homotopy failed: step size too small")
    
    # Stage 2: Interface charge homotopy (space charge = 1)
    logger.info("--- Stage 2: Interface charge homotopy ---")
    theta = 0.0
    step = 0.1
    
    while theta < 1.0 - 1e-12:
        trial = min(1.0, theta + step)
        
        ode_func, bc_func, _, _ = setup_bvp_system(
            phys, geom, V_tip, 1.0, trial, Feenstra, assume_full_ionization
        )
        
        try:
            sol = solve_bvp(ode_func, bc_func, x, current_y, max_nodes=5000, tol=1e-6)
            
            if not sol.success:
                raise RuntimeError(sol.message)
            
            theta = trial
            current_y = sol.sol(x)
            logger.info(f"  [Interface charge: θ={theta:.3f}] Converged. RMS residual: {sol.rms_residuals.max():.2e}")
            
            if step < 0.5:
                step *= 1.5
                
        except Exception as exc:
            step *= 0.5
            logger.warning(f"  [Interface charge: θ→{trial:.3f}] Failed ({exc}). Reducing step to {step:.4f}.")
            if step < min_step:
                raise RuntimeError("Interface charge homotopy failed: step size too small")
    
    phi = current_y[0]
    E = -current_y[1]  # Electric field E = -dφ/dx
    
    return phi, E


def solve_direct_newton_1d(
    phys: PhysicalParameters,
    geom: GeometricParameters1D,
    V_tip: float,
    x: np.ndarray,
    phi_init: np.ndarray,
    E_init: np.ndarray,
    Feenstra: bool,
    assume_full_ionization: bool,
    fallback_theta: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve directly with Newton (homotopy = 1.0), fallback to partial homotopy if needed
    
    Returns:
        phi: potential solution (dimensionless)
        E: electric field -dφ/dx (dimensionless)
    """
    logger.info("--- Attempting direct Newton solve (homotopy = 1.0) ---")
    
    y_init = np.vstack([phi_init, E_init])
    
    ode_func, bc_func, _, _ = setup_bvp_system(
        phys, geom, V_tip, 1.0, 1.0, Feenstra, assume_full_ionization
    )
    
    try:
        sol = solve_bvp(ode_func, bc_func, x, y_init, max_nodes=5000, tol=1e-6)
        
        if not sol.success:
            raise RuntimeError(sol.message)
        
        logger.info(f"  [Direct Newton] Converged. RMS residual: {sol.rms_residuals.max():.2e}")
        
        phi = sol.sol(x)[0]
        E = -sol.sol(x)[1]
        
        return phi, E
        
    except Exception as exc:
        logger.warning(f"  [Direct Newton] Failed: {exc}")
        logger.info(f"  Falling back to partial homotopy from θ={fallback_theta:.2f} to 1.0")
        
        # Fallback: partial homotopy
        theta = fallback_theta
        step = 0.1
        min_step = 1e-3
        current_y = y_init
        
        # Stage 1: Space charge from fallback_theta to 1.0 (σ = 0)
        while theta < 1.0 - 1e-12:
            trial = min(1.0, theta + step)
            
            ode_func, bc_func, _, _ = setup_bvp_system(
                phys, geom, V_tip, trial, 0.0, Feenstra, assume_full_ionization
            )
            
            try:
                sol = solve_bvp(ode_func, bc_func, x, current_y, max_nodes=5000, tol=1e-6)
                
                if not sol.success:
                    raise RuntimeError(sol.message)
                
                theta = trial
                current_y = sol.sol(x)
                logger.info(f"  [Fallback space charge: θ={theta:.3f}] Converged.")
                
                if step < 0.5:
                    step *= 1.5
                    
            except Exception as exc2:
                step *= 0.5
                logger.warning(f"  [Fallback: θ→{trial:.3f}] Failed. Reducing step to {step:.4f}.")
                if step < min_step:
                    raise RuntimeError("Fallback homotopy failed: step size too small")
        
        # Stage 2: Interface charge from fallback_theta to 1.0
        theta = fallback_theta
        step = 0.1
        
        while theta < 1.0 - 1e-12:
            trial = min(1.0, theta + step)
            
            ode_func, bc_func, _, _ = setup_bvp_system(
                phys, geom, V_tip, 1.0, trial, Feenstra, assume_full_ionization
            )
            
            try:
                sol = solve_bvp(ode_func, bc_func, x, current_y, max_nodes=5000, tol=1e-6)
                
                if not sol.success:
                    raise RuntimeError(sol.message)
                
                theta = trial
                current_y = sol.sol(x)
                logger.info(f"  [Fallback interface charge: θ={theta:.3f}] Converged.")
                
                if step < 0.5:
                    step *= 1.5
                    
            except Exception as exc2:
                step *= 0.5
                logger.warning(f"  [Fallback: θ→{trial:.3f}] Failed. Reducing step to {step:.4f}.")
                if step < min_step:
                    raise RuntimeError("Fallback homotopy failed: step size too small")
        
        phi = current_y[0]
        E = -current_y[1]
        
        return phi, E


def save_results_1d(
    x: np.ndarray,
    phi: np.ndarray,
    E: np.ndarray,
    mask_sic: np.ndarray,
    mask_sio2: np.ndarray,
    phys: PhysicalParameters,
    geom: GeometricParameters1D,
    V_tip: float,
    Feenstra: bool,
    out_dir: str,
):
    """Save 1D simulation results"""
    os.makedirs(out_dir, exist_ok=True)
    
    V_c = const.k * phys.T / const.e
    L_c = geom.L_c
    
    # Save raw data
    np.save(os.path.join(out_dir, "x_dimless.npy"), x)
    np.save(os.path.join(out_dir, "phi_dimless.npy"), phi)
    np.save(os.path.join(out_dir, "phi_volts.npy"), phi * V_c)
    np.save(os.path.join(out_dir, "E_dimless.npy"), E)
    np.save(os.path.join(out_dir, "E_Vpm.npy"), E * V_c / L_c)  # V/m
    
    # Save metadata
    meta = {
        "V_c": V_c,
        "L_c": L_c,
        "V_tip": V_tip,
        "Feenstra": Feenstra,
        "n_points": len(x),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot potential
    ax1 = axes[0]
    x_nm = x * L_c * 1e9
    ax1.plot(x_nm[mask_sic], phi[mask_sic] * V_c, 'b-', label='SiC', lw=2)
    ax1.plot(x_nm[mask_sio2], phi[mask_sio2] * V_c, 'r-', label='SiO2', lw=2)
    ax1.axvline(0, color='k', ls='--', alpha=0.3, label='Interface')
    ax1.set_xlabel('Position x [nm]')
    ax1.set_ylabel('Potential φ [V]')
    ax1.set_title(f'Electrostatic Potential (V_tip = {V_tip:.2f} V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot electric field
    ax2 = axes[1]
    E_MVpm = E * V_c / L_c / 1e6  # MV/m
    ax2.plot(x_nm[mask_sic], E_MVpm[mask_sic], 'b-', label='SiC', lw=2)
    ax2.plot(x_nm[mask_sio2], E_MVpm[mask_sio2], 'r-', label='SiO2', lw=2)
    ax2.axvline(0, color='k', ls='--', alpha=0.3, label='Interface')
    ax2.set_xlabel('Position x [nm]')
    ax2.set_ylabel('Electric Field E [MV/m]')
    ax2.set_title('Electric Field Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "solution.png"), dpi=150)
    plt.close()
    
    logger.info(f"Saved results to {out_dir}")


def run_simulation_1d(
    phys: PhysicalParameters,
    geom: GeometricParameters1D,
    V_tip_values: list[float],
    Feenstra: bool,
    assume_full_ionization: bool,
    out_dir: str,
):
    """Run 1D simulation for multiple V_tip values"""
    
    # Sort voltages by absolute value
    V_tip_sorted = sorted(V_tip_values, key=abs)
    logger.info(f"Processing V_tip values in order: {V_tip_sorted}")
    
    # Create grid once
    x, mask_sic, mask_sio2 = create_grid_1d(geom)
    
    # Loop through V_tip values
    phi_prev = None
    E_prev = None
    
    for i, V_tip in enumerate(V_tip_sorted):
        logger.info(f"{'='*60}")
        logger.info(f"Solving for V_tip = {V_tip:.3f} V ({i+1}/{len(V_tip_sorted)})")
        logger.info(f"{'='*60}")
        
        if i == 0:
            # First voltage: full homotopy
            phi, E = solve_with_homotopy_1d(
                phys, geom, V_tip, x, mask_sic, mask_sio2,
                Feenstra, assume_full_ionization
            )
        else:
            # Subsequent voltages: try direct Newton with fallback
            phi, E = solve_direct_newton_1d(
                phys, geom, V_tip, x, phi_prev, E_prev,
                Feenstra, assume_full_ionization
            )
        
        # Save results
        out_subdir = os.path.join(out_dir, f"V_tip_{V_tip:+.2f}V")
        save_results_1d(
            x, phi, E, mask_sic, mask_sio2,
            phys, geom, V_tip, Feenstra, out_subdir
        )
        
        # Store for next iteration
        phi_prev = phi
        E_prev = E
        
        logger.info(f"Results saved to {out_subdir}")


def parse_range_input(input_str: str) -> list[float]:
    """Parse voltage input (single value or range min:max:step)"""
    if ":" in input_str:
        parts = input_str.split(":")
        if len(parts) != 3:
            raise ValueError("Range input must be in the format min:max:step")
        min_val = float(parts[0])
        max_val = float(parts[1])
        step = float(parts[2])
        return list(np.arange(min_val, max_val + step/2, step))
    else:
        return [float(input_str)]


def main():
    parser = argparse.ArgumentParser(
        description="1D Planar Poisson Solver for SiC | SiO2 | metal System."
    )
    parser.add_argument(
        "--V_tip", type=str, default="2.0", help="Tip voltages (single value or range min:max:step)."
    )
    parser.add_argument(
        "--d_sic", type=float, default=100.0, help="Thickness of SiC layer in nm."
    )
    parser.add_argument(
        "--d_sio2", type=float, default=5.0, help="Thickness of SiO2 layer in nm."
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
        "--out_dir", type=str, default="out_1d", help="Output directory for results."
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
        help="Assume complete ionization of donors.",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    basicConfig(
        level=INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(args.out_dir, "main.log"),
    )
    
    print(f"Logging to {os.path.join(args.out_dir, 'main.log')}")
    start = datetime.now()
    logger.info(f"Started simulation at {start}")
    
    # Initialize physical parameters
    phys_params = PhysicalParameters(
        T=args.T,
        Nd=args.Nd * 1e6,  # cm^-3 -> m^-3
        sigma_s=args.sigma_s * 1e4,  # cm^-2 -> m^-2
    )
    
    # Calculate Fermi level
    Ef = find_fermi_level(phys_params, args.out_dir, plot=args.plot_fermi)
    phys_params.Ef = Ef
    logger.info(f"Fermi level: {Ef:.4f} eV")
    
    # Initialize geometric parameters
    geom_params = GeometricParameters1D(
        d_sic=args.d_sic,
        d_sio2=args.d_sio2,
    )
    
    # Parse V_tip values
    V_tip_values = parse_range_input(args.V_tip)
    
    # Save parameters
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
    
    # Run simulation
    run_simulation_1d(
        phys=phys_params,
        geom=geom_params,
        V_tip_values=V_tip_values,
        Feenstra=(args.model[0].upper() == "F"),
        assume_full_ionization=args.assume_full_ionization,
        out_dir=args.out_dir,
    )
    
    end = datetime.now()
    logger.info(f"Finished simulation at {end}, duration: {end - start}")


def find_vtip_subdirs_1d(out_dir: str) -> list[tuple[str, float]]:
    """Find V_tip_±X.XXV subdirectories and extract voltage values

    Returns:
        List of (subdir_path, V_tip_value) tuples, sorted by absolute V_tip value
    """
    import re
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


def load_results_1d(subdir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load 1D simulation results from a subdirectory
    
    Returns:
        x: grid points (dimensionless)
        phi: potential (dimensionless)
        E: electric field (dimensionless)
        phi_volts: potential in volts
        E_Vpm: electric field in V/m
        metadata: metadata dictionary
    """
    x = np.load(os.path.join(subdir, "x_dimless.npy"))
    phi = np.load(os.path.join(subdir, "phi_dimless.npy"))
    E = np.load(os.path.join(subdir, "E_dimless.npy"))
    phi_volts = np.load(os.path.join(subdir, "phi_volts.npy"))
    E_Vpm = np.load(os.path.join(subdir, "E_Vpm.npy"))
    
    with open(os.path.join(subdir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    return x, phi, E, phi_volts, E_Vpm, metadata


def calculate_donor_ionization_1d(
    phi_dimless: np.ndarray,
    phys: PhysicalParameters,
    V_c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ionized donor densities from dimensionless potential
    
    Returns:
        Ndp_h, Ndp_c, Ndp_total (all in m^-3)
    """
    phi_volt = phi_dimless * V_c
    kTeV = phys.kTeV
    
    Ndp_h = phys.Nd_h / (
        1 + 2 * np.exp((phys.Ef - phys.Edh + phi_volt) / kTeV)
    )
    Ndp_c = phys.Nd_c / (
        1 + 2 * np.exp((phys.Ef - phys.Edc + phi_volt) / kTeV)
    )
    Ndp_total = Ndp_h + Ndp_c
    
    return Ndp_h, Ndp_c, Ndp_total


def calculate_charge_densities_1d(
    phi_dimless: np.ndarray,
    phys: PhysicalParameters,
    V_c: float,
    assume_full_ionization: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate charge densities (n, p, Nd+) from dimensionless potential
    
    Returns:
        n, p, Ndp (all in m^-3)
    """
    phi_volt = phi_dimless * V_c
    kTeV = phys.kTeV
    
    # Electron density
    eta_n = (phys.Ef - phys.Ec + phi_volt) / kTeV
    n = phys.Nc * fermi_dirac_integral(eta_n)
    
    # Hole density
    eta_p = (phys.Ev - phys.Ef - phi_volt) / kTeV
    p = phys.Nv * fermi_dirac_integral(eta_p)
    
    # Ionized donor density
    if assume_full_ionization:
        Ndp = np.full_like(phi_dimless, phys.Nd)
    else:
        Ndp_h = phys.Nd_h / (
            1 + 2 * np.exp((phys.Ef - phys.Edh + phi_volt) / kTeV)
        )
        Ndp_c = phys.Nd_c / (
            1 + 2 * np.exp((phys.Ef - phys.Edc + phi_volt) / kTeV)
        )
        Ndp = Ndp_h + Ndp_c
    
    return n, p, Ndp


def process_single_vtip_1d(
    subdir: str,
    V_tip: float,
    geom_params: GeometricParameters1D,
    phys_params: PhysicalParameters,
    V_c: float,
    assume_full_ionization: bool,
    plot_donor_ionization: bool,
    plot_charge_density: bool,
):
    """Process a single V_tip directory and generate plots for 1D results"""
    print(f"\n{'=' * 60}")
    print(f"Processing V_tip = {V_tip:.3f} V")
    print(f"{'=' * 60}")
    
    # Load results
    x, phi, E, phi_volts, E_Vpm, metadata = load_results_1d(subdir)
    L_c = geom_params.L_c
    
    # Convert to physical units
    x_nm = x * L_c * 1e9
    E_MVpm = E_Vpm / 1e6
    
    # Identify SiC and SiO2 regions
    mask_sic = x < 0
    mask_sio2 = x >= 0
    
    # --- Basic plots: Potential and Electric Field ---
    print("Creating potential and electric field plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Potential plot
    ax1 = axes[0]
    ax1.plot(x_nm[mask_sic], phi_volts[mask_sic], 'b-', label='SiC', lw=2)
    ax1.plot(x_nm[mask_sio2], phi_volts[mask_sio2], 'r-', label='SiO2', lw=2)
    ax1.axvline(0, color='k', ls='--', alpha=0.3, label='Interface')
    ax1.set_xlabel('Position x [nm]')
    ax1.set_ylabel('Potential φ [V]')
    ax1.set_title(f'Electrostatic Potential (V_tip = {V_tip:.2f} V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Electric field plot
    ax2 = axes[1]
    ax2.plot(x_nm[mask_sic], E_MVpm[mask_sic], 'b-', label='SiC', lw=2)
    ax2.plot(x_nm[mask_sio2], E_MVpm[mask_sio2], 'r-', label='SiO2', lw=2)
    ax2.axvline(0, color='k', ls='--', alpha=0.3, label='Interface')
    ax2.set_xlabel('Position x [nm]')
    ax2.set_ylabel('Electric Field E [MV/m]')
    ax2.set_title('Electric Field Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "potential_and_field.png"), dpi=150)
    plt.close()
    print(f"Saved potential and field plot to {os.path.join(subdir, 'potential_and_field.png')}")
    
    # --- Donor ionization plots ---
    if plot_donor_ionization:
        print("Creating donor ionization plots...")
        
        # Only plot in SiC region
        x_sic = x[mask_sic]
        phi_sic = phi[mask_sic]
        
        Ndp_h, Ndp_c, Ndp_total = calculate_donor_ionization_1d(phi_sic, phys_params, V_c)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_sic_nm = x_sic * L_c * 1e9
        ax.plot(x_sic_nm, Ndp_h * 1e-6, label='$N_{D,h}^+$ (Hex)', color='blue', lw=1.5)
        ax.plot(x_sic_nm, Ndp_c * 1e-6, label='$N_{D,c}^+$ (Cubic)', color='red', lw=1.5)
        ax.plot(x_sic_nm, Ndp_total * 1e-6, label='$N_D^+$ (Total)', color='purple', lw=2)
        
        ax.axhline(y=phys_params.Nd_h * 1e-6, color='blue', ls='--', lw=1, 
                  label='$N_{D,h}$ (Complete Ionization)')
        ax.axhline(y=phys_params.Nd_c * 1e-6, color='red', ls='--', lw=1, 
                  label='$N_{D,c}$ (Complete Ionization)')
        ax.axhline(y=phys_params.Nd * 1e-6, color='purple', ls='--', lw=1.5, 
                  label='$N_D$ (Complete Ionization)')
        
        ax.set_xlabel('Position x [nm]')
        ax.set_ylabel('Ionized Donor Density [cm$^{-3}$]')
        ax.set_title(f'Donor Ionization Profile (V_tip = {V_tip:.2f} V)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "donor_ionization.png"), dpi=150)
        plt.close()
        print(f"Saved donor ionization plot to {os.path.join(subdir, 'donor_ionization.png')}")
        
        # Save data
        np.savetxt(
            os.path.join(subdir, "donor_ionization.txt"),
            np.column_stack((x_sic_nm, Ndp_h * 1e-6, Ndp_c * 1e-6, Ndp_total * 1e-6)),
            header="x_nm Ndp_h_cm-3 Ndp_c_cm-3 Ndp_total_cm-3"
        )
    
    # --- Charge density plots ---
    if plot_charge_density:
        print("Creating charge density plots...")
        
        # Only plot in SiC region
        x_sic = x[mask_sic]
        phi_sic = phi[mask_sic]
        
        n, p, Ndp = calculate_charge_densities_1d(phi_sic, phys_params, V_c, assume_full_ionization)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_sic_nm = x_sic * L_c * 1e9
        ax.semilogy(x_sic_nm, n * 1e-6, label='$n$ (Electron)', color='blue', lw=2)
        ax.semilogy(x_sic_nm, Ndp * 1e-6, label='$N_D^+$ (Ionized Donor)', color='red', lw=2)
        # ax.semilogy(x_sic_nm, p * 1e-6, label='$p$ (Hole)', color='green', lw=1.5, ls='--')
        
        ionization_status = "Full Ionization" if assume_full_ionization else "Partial Ionization"
        ax.set_xlabel('Position x [nm]')
        ax.set_ylabel('Charge Density [cm$^{-3}$]')
        ax.set_title(f'Charge Density Profile (V_tip = {V_tip:.2f} V)\n({ionization_status})')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "charge_density.png"), dpi=150)
        plt.close()
        print(f"Saved charge density plot to {os.path.join(subdir, 'charge_density.png')}")
        
        # Save data
        np.savetxt(
            os.path.join(subdir, "charge_density.txt"),
            np.column_stack((x_sic_nm, n * 1e-6, p * 1e-6, Ndp * 1e-6)),
            header="x_nm n_cm-3 p_cm-3 Ndp_cm-3"
        )


def create_comparison_plots_1d(vtip_dirs: list[tuple[str, float]], out_dir: str):
    """Create comparison plots overlaying all V_tip profiles"""
    print("\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)
    
    comparison_dir = os.path.join(out_dir, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Collect data from all V_tip directories
    all_data = []
    
    for subdir, V_tip in vtip_dirs:
        x, phi, E, phi_volts, E_Vpm, metadata = load_results_1d(subdir)
        L_c = metadata["L_c"]
        x_nm = x * L_c * 1e9
        E_MVpm = E_Vpm / 1e6
        
        all_data.append((V_tip, x_nm, phi_volts, E_MVpm))
    
    if not all_data:
        print("No data found for comparison plots.")
        return
    
    # Plot potential comparison
    fig_phi, ax_phi = plt.subplots(figsize=(10, 6))
    for V_tip, x_nm, phi_volts, _ in all_data:
        ax_phi.plot(x_nm, phi_volts, label=f"V_tip = {V_tip:+.2f} V", lw=1.5)
    ax_phi.axvline(0, color='k', ls='--', alpha=0.3)
    ax_phi.set_xlabel('Position x [nm]')
    ax_phi.set_ylabel('Potential φ [V]')
    ax_phi.set_title('Potential Profile Comparison')
    ax_phi.legend()
    ax_phi.grid(True, alpha=0.3)
    fig_phi.tight_layout()
    fig_phi.savefig(os.path.join(comparison_dir, "potential_comparison.png"), dpi=150)
    plt.close(fig_phi)
    print(f"Saved potential comparison to {comparison_dir}/potential_comparison.png")
    
    # Plot electric field comparison
    fig_E, ax_E = plt.subplots(figsize=(10, 6))
    for V_tip, x_nm, _, E_MVpm in all_data:
        ax_E.plot(x_nm, E_MVpm, label=f"V_tip = {V_tip:+.2f} V", lw=1.5)
    ax_E.axvline(0, color='k', ls='--', alpha=0.3)
    ax_E.set_xlabel('Position x [nm]')
    ax_E.set_ylabel('Electric Field E [MV/m]')
    ax_E.set_title('Electric Field Profile Comparison')
    ax_E.legend()
    ax_E.grid(True, alpha=0.3)
    fig_E.tight_layout()
    fig_E.savefig(os.path.join(comparison_dir, "electric_field_comparison.png"), dpi=150)
    plt.close(fig_E)
    print(f"Saved electric field comparison to {comparison_dir}/electric_field_comparison.png")


def postprocess_1d():
    """Post-process 1D simulation results"""
    parser = argparse.ArgumentParser(description="Post-process 1D simulation results")
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
        "--no_comparison",
        action="store_true",
        help="Skip comparison plots",
    )
    args = parser.parse_args()
    
    out_dir = args.out_dir
    
    print(f"Post-processing 1D results in {out_dir}...")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
    
    # Load global parameters
    params_file = os.path.join(out_dir, "parameters.json")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"No parameters.json found in {out_dir}")
    
    with open(params_file, "r") as f:
        params = json.load(f)
    
    geom_params = GeometricParameters1D(**params["geometric"])
    phys_params = PhysicalParameters(**params["physical"])
    assume_full_ionization = params["simulation"].get("assume_full_ionization", False)
    
    # Find V_tip directories
    vtip_dirs = find_vtip_subdirs_1d(out_dir)
    
    if not vtip_dirs:
        raise FileNotFoundError(f"No V_tip_±X.XXV subdirectories found in {out_dir}")
    
    print(f"Found {len(vtip_dirs)} V_tip directories:")
    for subdir, V_tip in vtip_dirs:
        print(f"  - {os.path.basename(subdir)} (V_tip = {V_tip:+.3f} V)")
    
    # Get V_c from first subdirectory
    first_metadata_path = os.path.join(vtip_dirs[0][0], "metadata.json")
    with open(first_metadata_path, "r") as f:
        metadata = json.load(f)
    V_c = metadata["V_c"]
    
    # Process each V_tip directory
    for subdir, V_tip in vtip_dirs:
        process_single_vtip_1d(
            subdir=subdir,
            V_tip=V_tip,
            geom_params=geom_params,
            phys_params=phys_params,
            V_c=V_c,
            assume_full_ionization=assume_full_ionization,
            plot_donor_ionization=args.plot_donor_ionization,
            plot_charge_density=args.plot_charge_density,
        )
    
    # Create comparison plots
    if not args.no_comparison:
        create_comparison_plots_1d(vtip_dirs, out_dir)
    
    print("\n" + "=" * 60)
    print("Post-processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if called with postprocess mode
    if len(sys.argv) > 1 and sys.argv[1] == "postprocess":
        # Remove "postprocess" from argv for argparse
        sys.argv.pop(1)
        postprocess_1d()
    else:
        main()
