import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import adios4dolfinx
import gmsh
import numpy as np
import scipy.constants as const
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# import matplotlib.pyplot as plt
from scipy.optimize import brentq

# MPIのランク0でのみログを出力するための設定
comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)


@dataclass
class PhysicalParameters:
    """シミュレーションで使用する物理パラメータを保持するデータクラス"""

    T: float = 300.0  # 温度 [K]
    Nd: float = 1e22  # ドナー濃度 [m^-3]
    Na: float = 0.0  # アクセプタ濃度 [m^-3]
    sigma_s: float = 1e15  # SiC/SiO2界面の面電荷密度 [m^-2]
    m_de: float = 0.42 * const.m_e
    m_dh: float = 1.0 * const.m_e
    Eg: float = 3.26  # バンドギャップ [eV]
    Ed_offset_hex: float = 0.124  # 六方サイトのドナー準位 [eV from Ec]
    Ed_offset_cub: float = 0.066  # 立方サイトのドナー準位 [eV from Ec]
    Ed_ratio_c_to_h: float = 1.88  # 立方/六方サイトの割合
    ni: float = 8.2e15  # 固有キャリア密度 [m^-3]
    eps_sic: float = 9.7
    eps_sio2: float = 3.9
    eps_vac: float = 1.0

    # 計算によって導出される物理量
    n0: float = 0.0
    p0: float = 0.0
    Nc: float = 0.0
    Nv: float = 0.0
    Ec: float = 0.0
    Ev: float = 0.0
    Edh: float = 0.0
    Edc: float = 0.0
    Nd_h: float = 0.0
    Nd_c: float = 0.0
    kTeV: float = 0.0
    Ef: float = 0.0

    def __post_init__(self):
        """初期化後に物理定数を計算する"""
        self.kTeV = const.k * self.T / const.e

        total_ratio = 1.0 + self.Ed_ratio_c_to_h
        self.Nd_h = self.Nd * 1.0 / total_ratio
        self.Nd_c = self.Nd * self.Ed_ratio_c_to_h / total_ratio

        # 電子/正孔の平衡密度
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
    """ジオメトリ関連のパラメータを保持するデータクラス"""

    l_sio2: float = 5.0  # SiO2層の厚さ [nm]
    tip_radius: float = 45.0  # 探針先端の曲率半径 [nm]
    tip_height: float = 8.0  # 探針と試料の距離 [nm]
    l_vac: float = 200.0  # 真空層の厚さ [nm]
    region_radius: float = 500.0  # 計算領域の半径 [nm]


def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
    """Aymerich-Humet 近似による F_{1/2}(x) の連続表現"""

    a1 = 6.316
    a2 = 12.92
    C_deg = 0.75224956896  # 4 / (3 * sqrt(pi))

    return np.piecewise(
        x,
        [x < -10.0],
        [
            lambda x: np.exp(x),
            lambda x: 1.0
            / (np.exp(-x) + (C_deg * (x**2 + a1 * x + a2) ** 0.75) ** (-1.0)),
        ],
    )


def find_fermi_level(
    params: PhysicalParameters, out_dir: str, plot: bool = False
) -> float:
    """
    電荷中性条件からフェルミ準位を数値的に計算する

    Args:
        params: 物理パラメータ
        out_dir: プロット画像を出力するディレクトリ
        plot: 計算過程をプロットするかどうか

    Returns:
        計算されたフェルミ準位 [eV]
    """

    def charge_neutrality_eq(Ef: float) -> float:
        p = params.Nv * fermi_dirac_integral((params.Ev - Ef) / params.kTeV)
        n = params.Nc * fermi_dirac_integral((Ef - params.Ec) / params.kTeV)
        Ndp_h = params.Nd_h / (1 + 2 * np.exp((Ef - params.Edh) / params.kTeV))
        Ndp_c = params.Nd_c / (1 + 2 * np.exp((Ef - params.Edc) / params.kTeV))
        Ndp = Ndp_h + Ndp_c
        # logスケールで解くことで数値的安定性を確保
        return np.log(p + Ndp) - np.log(n)

    search_min = params.Ev + params.kTeV
    search_max = params.Ec - params.kTeV
    Ef, res = brentq(charge_neutrality_eq, search_min, search_max, full_output=True)
    if not res.converged:
        raise ValueError("Root finding for Fermi level did not converge")

    if comm.rank == 0:
        logger.info(f"Calculated Fermi Level: Ef = {Ef:.4f} eV")

    if plot and comm.rank == 0:
        _plot_fermi_level_determination(params, Ef, out_dir)

    return Ef


def _plot_fermi_level_determination(
    params: PhysicalParameters, Ef: float, out_dir: str
):
    """フェルミ準位決定の過程をプロットするヘルパー関数"""
    try:
        import matplotlib.pyplot as plt

        ee = np.linspace(params.Ev - 0.1, params.Ec + 0.1, 500)
        p = params.Nv * fermi_dirac_integral((params.Ev - ee) / params.kTeV)
        n = params.Nc * fermi_dirac_integral((ee - params.Ec) / params.kTeV)
        Ndp_h = params.Nd_h / (1 + 2 * np.exp((ee - params.Edh) / params.kTeV))
        Ndp_c = params.Nd_c / (1 + 2 * np.exp((ee - params.Edc) / params.kTeV))
        Ndp = Ndp_h + Ndp_c

        plt.figure(figsize=(8, 6))
        plt.plot(
            ee, p + Ndp, label="Positive Charges ($p + N_D^+$)", color="blue", lw=2
        )
        plt.plot(ee, n, label="Negative Charges ($n$)", color="red", lw=2)
        plt.yscale("log")
        plt.title("Charge Concentrations vs. Fermi Level")
        plt.xlabel("Energy Level (E) / eV")
        plt.ylabel("Concentration / m$^{-3}$")
        plt.axvline(Ef, color="red", ls="-.", lw=1.5, label=f"Ef = {Ef:.2f} eV")
        plt.axvline(params.Ec, color="gray", ls=":", label="$E_c$")
        plt.axvline(params.Edh, color="cyan", ls="--", lw=1, label="$E_{d,h}$")
        plt.axvline(params.Edc, color="purple", ls="--", lw=1, label="$E_{d,c}$")
        plt.axvline(params.Ev, color="gray", ls=":", label="$E_v$")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(out_dir, "fermi_level_determination.png"), dpi=150)
        plt.close()

    except ImportError:
        logger.warning("matplotlib is not installed. Skipping Fermi level plot.")


def create_mesh(geom: GeometricParameters):
    """
    Gmshを使用して、シミュレーション領域のジオメトリとメッシュを生成する

    Args:
        geom: ジオメトリパラメータ

    Returns:
        dolfinx.mesh.Mesh: メッシュ
        dolfinx.mesh.MeshTags: セルタグ
        dolfinx.mesh.MeshTags: ファセットタグ
    """
    if comm.rank == 0:
        logger.info("Starting mesh generation with Gmsh...")

    # ジオメトリの無次元化 (代表長さ L_c = 1 nm)
    L_c = 1e-9
    R_dimless = geom.region_radius * 1e-9 / L_c
    sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    sic_depth_dimless = (geom.l_vac - geom.l_sio2) * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_slope_angle = 15 * np.pi / 180  # 75 deg tip angle -> 15 deg slope

    gmsh.initialize()
    gmsh.option.setNumber(
        "General.Terminal",
        1 if comm.rank == 0 and logging.getLogger().level == logging.DEBUG else 0,
    )
    gmsh.option.setNumber("Geometry.Tolerance", 1e-12)

    # 中心軸上 r=0
    p1 = gmsh.model.geo.addPoint(0, -sic_depth_dimless - sio2_depth_dimless, 0)
    p2 = gmsh.model.geo.addPoint(0, -sio2_depth_dimless, 0)
    origin = gmsh.model.geo.addPoint(0, 0, 0)

    # 探針
    tipO = gmsh.model.geo.addPoint(0, tip_z_dimless + tip_radius_dimless, 0)
    tip1 = gmsh.model.geo.addPoint(0, tip_z_dimless, 0)
    tip2 = gmsh.model.geo.addPoint(
        tip_radius_dimless * np.cos(tip_slope_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.sin(tip_slope_angle)),
        0,
    )
    tip3 = gmsh.model.geo.addPoint(
        tip_radius_dimless * np.cos(tip_slope_angle)
        + (
            vac_depth_dimless
            - tip_z_dimless
            - tip_radius_dimless * (1 - np.sin(tip_slope_angle))
        )
        * np.tan(tip_slope_angle),
        vac_depth_dimless,
        0,
    )

    # 遠方境界上
    q1 = gmsh.model.geo.addPoint(R_dimless, -sic_depth_dimless - sio2_depth_dimless, 0)
    q2 = gmsh.model.geo.addPoint(R_dimless, -sio2_depth_dimless, 0)
    q3 = gmsh.model.geo.addPoint(R_dimless, 0, 0)
    q4 = gmsh.model.geo.addPoint(R_dimless, vac_depth_dimless, 0)

    # lines
    p1p2 = gmsh.model.geo.addLine(p1, p2)
    p2q2 = gmsh.model.geo.addLine(p2, q2)
    q2q1 = gmsh.model.geo.addLine(q2, q1)
    q1p1 = gmsh.model.geo.addLine(q1, p1)
    q2q3 = gmsh.model.geo.addLine(q2, q3)
    q3O = gmsh.model.geo.addLine(q3, origin)
    Op2 = gmsh.model.geo.addLine(origin, p2)
    Ot1 = gmsh.model.geo.addLine(origin, tip1)
    tiparc = gmsh.model.geo.addCircleArc(tip1, tipO, tip2)
    t2t3 = gmsh.model.geo.addLine(tip2, tip3)
    t3q4 = gmsh.model.geo.addLine(tip3, q4)
    q4q3 = gmsh.model.geo.addLine(q4, q3)

    # surfaces
    loop_sic = gmsh.model.geo.addCurveLoop([p1p2, p2q2, q2q1, q1p1])
    surf_sic = gmsh.model.geo.addPlaneSurface([loop_sic])
    loop_sio2 = gmsh.model.geo.addCurveLoop([Op2, p2q2, q2q3, q3O])
    surf_sio2 = gmsh.model.geo.addPlaneSurface([loop_sio2])
    loop_vac = gmsh.model.geo.addCurveLoop([Ot1, tiparc, t2t3, t3q4, q4q3, q3O])
    surf_vac = gmsh.model.geo.addPlaneSurface([loop_vac])

    gmsh.model.geo.synchronize()

    # メッシュサイズ制御
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

    field_tag_counter = 1

    # --- Field 1: 探針先端の極微細領域 ---
    f_tip = gmsh.model.mesh.field.add("Distance", field_tag_counter)
    field_tag_counter += 1
    gmsh.model.mesh.field.setNumbers(f_tip, "CurvesList", [tiparc, Ot1])
    f_thresh_tip = gmsh.model.mesh.field.add("Threshold", field_tag_counter)
    field_tag_counter += 1
    gmsh.model.mesh.field.setNumber(f_thresh_tip, "InField", f_tip)
    gmsh.model.mesh.field.setNumber(f_thresh_tip, "SizeMin", 1)
    gmsh.model.mesh.field.setNumber(f_thresh_tip, "SizeMax", 10)
    gmsh.model.mesh.field.setNumber(f_thresh_tip, "DistMin", 10)
    gmsh.model.mesh.field.setNumber(f_thresh_tip, "DistMax", R_dimless / 5)

    # --- Field 2: SiO2層とその近傍の中間領域 ---
    f_sio2 = gmsh.model.mesh.field.add("Distance", field_tag_counter)
    field_tag_counter += 1
    # SiO2層を構成する界面
    gmsh.model.mesh.field.setNumbers(f_sio2, "CurvesList", [q3O, Op2, p2q2])
    f_thresh_sio2 = gmsh.model.mesh.field.add("Threshold", field_tag_counter)
    field_tag_counter += 1
    gmsh.model.mesh.field.setNumber(f_thresh_sio2, "InField", f_sio2)
    gmsh.model.mesh.field.setNumber(f_thresh_sio2, "SizeMin", 1)
    gmsh.model.mesh.field.setNumber(f_thresh_sio2, "SizeMax", 10)
    gmsh.model.mesh.field.setNumber(f_thresh_sio2, "DistMin", 10)
    gmsh.model.mesh.field.setNumber(f_thresh_sio2, "DistMax", sic_depth_dimless / 2)

    # --- 複数フィールドの最小値を採用 ---
    f_min = gmsh.model.mesh.field.add("Min", field_tag_counter)
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thresh_tip, f_thresh_sio2])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [surf_sic], 1, "sic")
    gmsh.model.addPhysicalGroup(2, [surf_sio2], 2, "sio2")
    gmsh.model.addPhysicalGroup(2, [surf_vac], 3, "vacuum")
    gmsh.model.addPhysicalGroup(1, [q1p1], 11, "ground")
    gmsh.model.addPhysicalGroup(1, [p1p2, Op2, tip1], 12, "axis")
    gmsh.model.addPhysicalGroup(1, [q2q1, q2q3, q4q3], 13, "far-field")
    gmsh.model.addPhysicalGroup(1, [tiparc, t2t3], 14, "tip")
    gmsh.model.addPhysicalGroup(1, [p2q2], 15, "sic_sio2_interface")

    gmsh.model.mesh.generate(2)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2, partitioner=partitioner
    )
    gmsh.finalize()

    if comm.rank == 0:
        logger.info("Mesh generation completed.")

    return msh, cell_tags, facet_tags


def run_fem_simulation(
    phys: PhysicalParameters,
    geom: GeometricParameters,
    V_tip: float,
    out_dir: str,
):
    """
    有限要素法を用いてポアソン方程式を解く

    Args:
        phys: 物理パラメータ
        geom: ジオメトリパラメータ
        V_tip: 印加する探針電圧 [V]
        out_dir: 結果を出力するディレクトリ
    """
    msh, cell_tags, facet_tags = create_mesh(geom)

    # 無次元化のための代表スケール
    L_c = 1e-9  # 代表長さ [1 nm]
    V_c = const.k * phys.T / const.e  # 代表電位 (熱電圧) [V]

    # 関数空間とテスト/トライアル関数
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name="potential_dimless")
    v = ufl.TestFunction(V)

    # 比誘電率を定義
    epsilon_r = fem.Function(
        fem.functionspace(msh, ("DG", 0)), name="relative_permittivity"
    )
    epsilon_r.x.array[cell_tags.find(1)] = phys.eps_sic
    epsilon_r.x.array[cell_tags.find(2)] = phys.eps_sio2
    epsilon_r.x.array[cell_tags.find(3)] = phys.eps_vac

    # ホモトピーパラメータ
    homotopy_charge = fem.Constant(msh, ScalarType(0.0))
    homotopy_sigma = fem.Constant(msh, ScalarType(0.0))

    # 弱形式の定義
    F, J = _setup_weak_form(
        u,
        v,
        epsilon_r,
        phys,
        V_c,
        L_c,
        homotopy_charge,
        homotopy_sigma,
        geom,
        cell_tags,
        facet_tags,
    )

    # 境界条件
    dofs_ground = fem.locate_dofs_topological(V, 1, facet_tags.find(11))
    bc_ground = fem.dirichletbc(ScalarType(0), dofs_ground, V)
    dofs_tip = fem.locate_dofs_topological(V, 1, facet_tags.find(14))
    bc_tip = fem.dirichletbc(ScalarType(V_tip / V_c), dofs_tip, V)
    bcs = [bc_ground, bc_tip]

    # 線形問題 (ラプラス方程式) で初期解を計算
    _warm_start_with_linear_solve(
        V, u, epsilon_r, V_tip, V_c, bc_ground, dofs_tip, geom
    )
    # check_vector_device(u, "potential 'u' after linear solve")

    # ホモトピー法で非線形問題を解く
    solve_with_homotopy(F, J, u, bcs, homotopy_charge, homotopy_sigma)
    # check_vector_device(u, "potential 'u' after nonlinear solve")

    # 結果を保存
    save_results(msh, u, epsilon_r, V_c, out_dir)


def _setup_weak_form(
    u,
    v,
    epsilon_r,
    phys,
    V_c,
    L_c,
    homotopy_charge,
    homotopy_sigma,
    geom,
    cell_tags,
    facet_tags,
):
    """弱形式とヤコビアンを定義する"""
    x = ufl.SpatialCoordinate(u.function_space.mesh)
    r = x[0]  # 円筒座標系の半径
    dx = ufl.Measure("dx", domain=u.function_space.mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=u.function_space.mesh, subdomain_data=facet_tags)
    dS = ufl.Measure("dS", domain=u.function_space.mesh, subdomain_data=facet_tags)

    # 無次元化された係数
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    C_Nc = homotopy_charge * C0 * phys.Nc
    C_Nv = homotopy_charge * C0 * phys.Nv
    C_Nd_h = homotopy_charge * C0 * phys.Nd_h
    C_Nd_c = homotopy_charge * C0 * phys.Nd_c
    sigma_s_target = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)
    sigma_s_dimless = homotopy_sigma * sigma_s_target

    # 無次元化されたエネルギー準位
    Ef_dimless = phys.Ef / V_c
    Ec_dimless = phys.Ec / V_c
    Ev_dimless = phys.Ev / V_c
    Edh_dimless = phys.Edh / V_c
    Edc_dimless = phys.Edc / V_c

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
                "sigma_s_target": sigma_s_target,
            },
            indent=2,
        )
    )

    clip_potential = 120.0
    clip_exp = 40.0

    def clamp(val, bound):
        return ufl.max_value(ufl.min_value(val, bound), -bound)

    def safe_exp(x):
        return ufl.exp(clamp(x, clip_exp))

    # 電荷密度項（数値的安定性のために電位をクリップ）
    u_clip = clamp(u, clip_potential)

    def fermi_dirac_ufl(x):
        """Aymerich-Humet 近似の UFL 実装"""

        a1 = 6.316
        a2 = 12.92
        C_deg = 0.75224956896

        boltzmann_approx = safe_exp(x)
        exp_neg_x = safe_exp(-x)
        x_safe = ufl.max_value(x, -4.0)
        G_inv_denominator = C_deg * (x_safe**2 + a1 * x_safe + a2) ** 0.75
        G_inv = G_inv_denominator ** (-1.0)
        full_approx = 1.0 / (exp_neg_x + G_inv)

        return ufl.conditional(ufl.gt(x, -10.0), full_approx, boltzmann_approx)

    n_term = C_Nc * fermi_dirac_ufl((Ef_dimless - Ec_dimless) + u_clip)
    p_term = C_Nv * fermi_dirac_ufl((Ev_dimless - Ef_dimless) - u_clip)
    Ndp_h_term = C_Nd_h / (1 + 2 * safe_exp((Ef_dimless - Edh_dimless) + u_clip))
    Ndp_c_term = C_Nd_c / (1 + 2 * safe_exp((Ef_dimless - Edc_dimless) + u_clip))
    Ndp_term = Ndp_h_term + Ndp_c_term
    rho_dimless = p_term + Ndp_term - n_term

    # ポアソン方程式の弱形式
    a = ufl.inner(epsilon_r * ufl.grad(u), ufl.grad(v)) * r
    L_bulk = rho_dimless * v * r
    # L_surface = sigma_s_dimless * ufl.avg(v) * r
    L_surface = sigma_s_dimless * v("+") * r("+")
    lambda_ff = 1 / (geom.region_radius * 1e-9 / L_c)

    F = (
        a * dx
        - L_bulk * dx(1)
        - L_surface * dS(15)
        + epsilon_r * lambda_ff * u * v * r * ds(13)
    )
    J = ufl.derivative(F, u)

    return F, J


def _warm_start_with_linear_solve(
    V, u, epsilon_r, V_tip, V_c, bc_ground, dofs_tip, geom
):
    """電圧を徐々に印加しながら線形問題を解き、非線形問題の初期値を準備する"""
    if comm.rank == 0:
        logger.info("Performing warm-start with linear Poisson equation...")

    w, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    r = x[0]
    lambda_ff = 1 / (geom.region_radius * 1e-9 / 1e-9)

    a_lin = ufl.inner(
        epsilon_r * ufl.grad(w), ufl.grad(v)
    ) * r * ufl.dx + epsilon_r * lambda_ff * w * v * r * ufl.ds(13)
    L_lin = fem.Constant(V.mesh, ScalarType(0.0)) * v * ufl.dx

    u.x.array[:] = 0.0
    uh = fem.Function(V)

    # 電圧を少しずつ上げて解くことで安定性を確保
    voltages_warmup = np.linspace(0.0, V_tip, 5)[1:]
    for v_val in voltages_warmup:
        # bcs[-1].g.value = ScalarType(v_val / V_c) # Update tip voltage
        bc_tip = fem.dirichletbc(ScalarType(v_val / V_c), dofs_tip, V)
        bcs = [bc_ground, bc_tip]
        # problem = LinearProblem(a_lin, L_lin, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem = LinearProblem(
            a_lin, L_lin, bcs=bcs, petsc_options={"ksp_type": "gmres", "pc_type": "ilu"}
        )  # optimize for GPU
        uh = problem.solve()
        u.x.array[:] = uh.x.array
        if comm.rank == 0:
            logger.info(f"  [Linear Warm-up] Solved at V_tip = {v_val:.2f} V")


def solve_with_homotopy(F, J, u, bcs, homotopy_charge, homotopy_sigma):
    """
    2段階のホモトピー法を用いて非線形問題を解く
    Stage 1: 空間電荷を徐々に導入
    Stage 2: 界面電荷を徐々に導入
    """
    # Stage 1: 空間電荷の導入
    homotopy_sigma.value = 0.0
    _solve_homotopy_stage(F, J, u, bcs, homotopy_charge, stage_name="Space Charge")

    # Stage 2: 界面電荷の導入
    homotopy_charge.value = 1.0
    _solve_homotopy_stage(F, J, u, bcs, homotopy_sigma, stage_name="Interface Charge")


def _solve_homotopy_stage(F, J, u, bcs, homotopy_param, stage_name: str):
    """ホモトピー法の1ステージを実行する共通関数"""
    if comm.rank == 0:
        logger.info(f"--- Starting Homotopy Stage: {stage_name} ---")

    theta = 0.0
    step = 0.1
    min_step = 1e-4
    uh = fem.Function(u.function_space)
    uh.x.array[:] = u.x.array

    problem = NonlinearProblem(F, u, bcs=bcs, J=J)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 0.7

    ksp = solver.krylov_solver
    # ksp.setType("preonly")
    ksp.setType("gmres")  # optimize for GPU
    pc = ksp.getPC()
    # pc.setType("lu")
    pc.setType("ilu")  # optimize for GPU

    while theta < 1.0 - 1e-12:
        trial = min(1.0, theta + step)
        homotopy_param.value = ScalarType(trial)

        try:
            n, converged = solver.solve(u)
        except RuntimeError:
            converged = False

        if converged:
            theta = trial
            uh.x.array[:] = u.x.array  # 収束した解をバックアップ
            if comm.rank == 0:
                logger.info(
                    f"  [{stage_name} Homotopy: θ={theta:.3f}] Converged in {n} Newton iterations."
                )
            # 収束が速ければステップサイズを増やす
            if n <= 3 and step < 0.5:
                step *= 1.5
        else:
            u.x.array[:] = uh.x.array  # 失敗したので安定した解に戻す
            step *= 0.5
            if comm.rank == 0:
                logger.warning(
                    f"  [{stage_name} Homotopy: θ→{trial:.3f}] Failed to converge. Reducing step to {step:.4f}."
                )
            if step < min_step:
                raise RuntimeError(
                    f"Homotopy stage '{stage_name}' failed: step size became too small."
                )


def save_results(msh, u, epsilon_r, V_c, out_dir: str):
    """計算結果をファイルに保存する"""
    # 電位を次元ありの値 [V] に戻す
    u_dim = fem.Function(u.function_space, name="potential")
    u_dim.x.array[:] = u.x.array * V_c

    # XDMF形式で保存
    xdmf_path = os.path.join(out_dir, "solution.xdmf")
    with XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(u_dim, 0.0)
        xdmf.write_function(epsilon_r, 0.0)

    # ADIOS形式で保存 (後処理・解析用)
    potential_path = os.path.join(out_dir, "potential.bp")
    adios4dolfinx.write_function(potential_path, u_dim, name="potential")
    epsilon_path = os.path.join(out_dir, "epsilon_r.bp")
    adios4dolfinx.write_function(epsilon_path, epsilon_r, name="epsilon_r")

    if comm.rank == 0:
        logger.info(f"Solution saved to {xdmf_path}")
        logger.info(f"Data for analysis saved to {out_dir}/*.bp")


def check_vector_device(vector, vector_name: str):
    """
    dolfinxのベクトルがCPUとGPUのどちらにあるかを確認して表示する
    """
    # dolfinx FunctionからPETSc Vecオブジェクトを取得
    petsc_vec = vector.x.petsc_vec

    # PETSc Vecの現在のタイプを取得
    vec_type = petsc_vec.getType()

    # ランク0のプロセスでのみメッセージを出力
    if vector.function_space.mesh.comm.rank == 0:
        if "cuda" in vec_type:
            logger.info(
                f"✅ DEBUG: Vector '{vector_name}' is on the GPU (type: {vec_type})"
            )
        else:
            logger.warning(
                f"⚠️ DEBUG: Vector '{vector_name}' is on the CPU (type: {vec_type})"
            )


def main():
    """メイン実行関数"""
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
    args, _ = parser.parse_known_args()

    if comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
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

    if comm.rank == 0:
        print("Logging to", os.path.join(args.out_dir, "main.log"))
        start = datetime.now()
        logger.info(f"Started simulation at {start}")

    # 物理パラメータの初期化 (単位をm^-3, m^-2に変換)
    phys_params = PhysicalParameters(
        T=args.T,
        Nd=args.Nd * 1e6,  # cm^-3 -> m^-3
        sigma_s=args.sigma_s * 1e4,  # cm^-2 -> m^-2
    )

    # フェルミ準位の計算
    Ef = find_fermi_level(phys_params, args.out_dir, plot=args.plot_fermi)
    phys_params.Ef = Ef

    # ジオメトリパラメータの初期化
    geom_params = GeometricParameters(
        l_sio2=args.l_sio2, tip_radius=args.tip_radius, tip_height=args.tip_height
    )

    # パラメータをJSONファイルに保存
    if comm.rank == 0:
        all_params = {
            "physical": asdict(phys_params),
            "geometric": asdict(geom_params),
            "simulation": {"V_tip": args.V_tip},
            "mpi_procs": comm.size,
        }
        with open(os.path.join(args.out_dir, "parameters.json"), "w") as f:
            json.dump(all_params, f, indent=2)

    # FEMシミュレーションの実行
    run_fem_simulation(
        phys=phys_params, geom=geom_params, V_tip=args.V_tip, out_dir=args.out_dir
    )

    end = datetime.now()
    logger.info(f"Finished simulation at {end}, duration: {end - start}")


if __name__ == "__main__":
    main()
