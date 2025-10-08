import os
import json
import logging
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt

import ngsolve as ng
from netgen.geom2d import SplineGeometry

from scipy.optimize import brentq
import scipy.constants as const



logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
    Ed_offset: float = 0.09  # ドナー準位のオフセット [eV from Ec]
    Ea_offset: float = 0.2  # アクセプタ準位のオフセット [eV from Ev]
    ni: float = 8.2e15  # 固有キャリア密度 [m^-3]
    eps_sic: float = 9.7
    eps_sio2: float = 3.9
    eps_vac: float = 1.0

    # 計算によって導出される物理量
    Nc: float = 0.0
    Nv: float = 0.0
    Ec: float = 0.0
    Ev: float = 0.0
    Ed: float = 0.0
    Ea: float = 0.0
    kTeV: float = 0.0
    Ef: float = 0.0

    def __post_init__(self):
        """初期化後に物理定数を計算する"""
        self.kTeV = const.k * self.T / const.e
        self.Nc = 2 * (2 * np.pi * self.m_de * const.k * self.T / (const.h**2)) ** 1.5
        self.Nv = 2 * (2 * np.pi * self.m_dh * const.k * self.T / (const.h**2)) ** 1.5
        self.Ev = 0
        self.Ec = self.Ev + self.Eg
        self.Ed = self.Ec - self.Ed_offset
        self.Ea = self.Ev + self.Ea_offset


@dataclass
class GeometricParameters:
    """ジオメトリ関連のパラメータを保持するデータクラス"""
    l_sio2: float = 5.0  # SiO2層の厚さ [nm]
    tip_radius: float = 45.0  # 探針先端の曲率半径 [nm]
    tip_height: float = 8.0  # 探針と試料の距離 [nm]
    l_vac: float = 200.0  # 真空層の厚さ [nm]
    region_radius: float = 500.0  # 計算領域の半径 [nm]


def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
    """フェルミ・ディラック積分の半整数次 (j=1/2) の近似式"""
    return np.piecewise(
        x,
        [x > 25],
        [
            lambda x: (2 / np.sqrt(np.pi)) * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
            lambda x: np.exp(x) / (1 + 0.27 * np.exp(x)),
        ],
    )


def find_fermi_level(params: PhysicalParameters, out_dir: str, plot: bool = False) -> float:
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
        Ndp = params.Nd / (1 + 2 * np.exp((Ef - params.Ed) / params.kTeV))
        # logスケールで解くことで数値的安定性を確保
        return np.log(p + Ndp) - np.log(n)

    search_min = params.Ev + params.kTeV
    search_max = params.Ec - params.kTeV
    Ef, res = brentq(charge_neutrality_eq, search_min, search_max, full_output=True)
    if not res.converged:
        raise ValueError("Root finding for Fermi level did not converge")

    _plot_fermi_level_determination(params, Ef, out_dir)

    return Ef

def _plot_fermi_level_determination(params: PhysicalParameters, Ef: float, out_dir: str):
    """フェルミ準位決定の過程をプロットするヘルパー関数"""

    ee = np.linspace(params.Ev - 0.1, params.Ec + 0.1, 500)
    p = params.Nv * fermi_dirac_integral((params.Ev - ee) / params.kTeV)
    n = params.Nc * fermi_dirac_integral((ee - params.Ec) / params.kTeV)
    Ndp = params.Nd / (1 + 2 * np.exp((ee - params.Ed) / params.kTeV))

    plt.figure(figsize=(8, 6))
    plt.plot(ee, p + Ndp, label="Positive Charges ($p + N_D^+$)", color="blue", lw=2)
    plt.plot(ee, n, label="Negative Charges ($n$)", color="red", lw=2)
    plt.yscale("log")
    plt.title("Charge Concentrations vs. Fermi Level")
    plt.xlabel("Energy Level (E) / eV")
    plt.ylabel("Concentration / m$^{-3}$")
    plt.axvline(Ef, color="red", ls="-.", lw=1.5, label=f"Ef = {Ef:.2f} eV")
    plt.axvline(params.Ec, color="gray", ls=":", label="$E_c$")
    plt.axvline(params.Ed, color="gray", ls="--", label="$E_d$")
    plt.axvline(params.Ev, color="gray", ls=":", label="$E_v$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(out_dir, "fermi_level_determination.png"), dpi=150)
    plt.close()


def create_mesh(geom: GeometricParameters):
    """
    netgen を使用して、シミュレーション領域のジオメトリとメッシュを生成する

    Args:
        geom: ジオメトリパラメータ
    """
    
    # # ジオメトリの無次元化 (代表長さ L_c = 1 nm)
    # L_c = 1e-9
    # R_dimless = geom.region_radius * 1e-9 / L_c
    # sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    # vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    # sic_depth_dimless = (geom.l_vac - geom.l_sio2) * 1e-9 / L_c
    # tip_z_dimless = geom.tip_height * 1e-9 / L_c
    # tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    # tip_slope_angle = 15 * np.pi / 180 # 75 deg tip angle -> 15 deg slope

    # gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 1 if comm.rank == 0 and logging.getLogger().level == logging.DEBUG else 0)
    # gmsh.option.setNumber("Geometry.Tolerance", 1e-12)

    # # 中心軸上 r=0
    # p1 = gmsh.model.geo.addPoint(0, -sic_depth_dimless - sio2_depth_dimless, 0)
    # p2 = gmsh.model.geo.addPoint(0, -sio2_depth_dimless, 0)
    # O = gmsh.model.geo.addPoint(0, 0, 0)

    # # 探針
    # tipO = gmsh.model.geo.addPoint(0, tip_z_dimless + tip_radius_dimless, 0)
    # tip1 = gmsh.model.geo.addPoint(0, tip_z_dimless, 0)
    # tip2 = gmsh.model.geo.addPoint(
    #     tip_radius_dimless * np.cos(tip_slope_angle),
    #     tip_z_dimless + tip_radius_dimless * (1 - np.sin(tip_slope_angle)),
    #     0,
    # )
    # tip3 = gmsh.model.geo.addPoint(
    #     tip_radius_dimless * np.cos(tip_slope_angle)
    #     + (
    #         vac_depth_dimless
    #         - tip_z_dimless
    #         - tip_radius_dimless * (1 - np.sin(tip_slope_angle))
    #     )
    #     * np.tan(tip_slope_angle),
    #     vac_depth_dimless,
    #     0,
    # )

    # # 遠方境界上
    # q1 = gmsh.model.geo.addPoint(R_dimless, -sic_depth_dimless - sio2_depth_dimless, 0)
    # q2 = gmsh.model.geo.addPoint(R_dimless, -sio2_depth_dimless, 0)
    # q3 = gmsh.model.geo.addPoint(R_dimless, 0, 0)
    # q4 = gmsh.model.geo.addPoint(R_dimless, vac_depth_dimless, 0)

    # # lines
    # p1p2 = gmsh.model.geo.addLine(p1, p2)
    # p2q2 = gmsh.model.geo.addLine(p2, q2)
    # q2q1 = gmsh.model.geo.addLine(q2, q1)
    # q1p1 = gmsh.model.geo.addLine(q1, p1)
    # q2q3 = gmsh.model.geo.addLine(q2, q3)
    # q3O = gmsh.model.geo.addLine(q3, O)
    # Op2 = gmsh.model.geo.addLine(O, p2)
    # Ot1 = gmsh.model.geo.addLine(O, tip1)
    # tiparc = gmsh.model.geo.addCircleArc(tip1, tipO, tip2)
    # t2t3 = gmsh.model.geo.addLine(tip2, tip3)
    # t3q4 = gmsh.model.geo.addLine(tip3, q4)
    # q4q3 = gmsh.model.geo.addLine(q4, q3)

    # # surfaces
    # loop_sic = gmsh.model.geo.addCurveLoop([p1p2, p2q2, q2q1, q1p1])
    # surf_sic = gmsh.model.geo.addPlaneSurface([loop_sic])
    # loop_sio2 = gmsh.model.geo.addCurveLoop([Op2, p2q2, q2q3, q3O])
    # surf_sio2 = gmsh.model.geo.addPlaneSurface([loop_sio2])
    # loop_vac = gmsh.model.geo.addCurveLoop([Ot1, tiparc, t2t3, t3q4, q4q3, q3O])
    # surf_vac = gmsh.model.geo.addPlaneSurface([loop_vac])

    # gmsh.model.geo.synchronize()

    # # メッシュサイズ制御
    # gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

    # field_tag_counter = 1

    # # --- Field 1: 探針先端の極微細領域 ---
    # f_tip = gmsh.model.mesh.field.add("Distance", field_tag_counter)
    # field_tag_counter += 1
    # gmsh.model.mesh.field.setNumbers(f_tip, "CurvesList", [tiparc, Ot1])
    # f_thresh_tip = gmsh.model.mesh.field.add("Threshold", field_tag_counter)
    # field_tag_counter += 1
    # gmsh.model.mesh.field.setNumber(f_thresh_tip, "InField", f_tip)
    # gmsh.model.mesh.field.setNumber(f_thresh_tip, "SizeMin", 1)
    # gmsh.model.mesh.field.setNumber(f_thresh_tip, "SizeMax", 10)
    # gmsh.model.mesh.field.setNumber(f_thresh_tip, "DistMin", 10)
    # gmsh.model.mesh.field.setNumber(f_thresh_tip, "DistMax", R_dimless / 5)

    # # --- Field 2: SiO2層とその近傍の中間領域 ---
    # f_sio2 = gmsh.model.mesh.field.add("Distance", field_tag_counter)
    # field_tag_counter += 1
    # # SiO2層を構成する界面
    # gmsh.model.mesh.field.setNumbers(f_sio2, "CurvesList", [q3O, Op2, p2q2])
    # f_thresh_sio2 = gmsh.model.mesh.field.add("Threshold", field_tag_counter)
    # field_tag_counter += 1
    # gmsh.model.mesh.field.setNumber(f_thresh_sio2, "InField", f_sio2)
    # gmsh.model.mesh.field.setNumber(f_thresh_sio2, "SizeMin", 2.5)
    # gmsh.model.mesh.field.setNumber(f_thresh_sio2, "SizeMax", 10)
    # gmsh.model.mesh.field.setNumber(f_thresh_sio2, "DistMin", 10)
    # gmsh.model.mesh.field.setNumber(f_thresh_sio2, "DistMax", sic_depth_dimless / 2)

    # # --- 複数フィールドの最小値を採用 ---
    # f_min = gmsh.model.mesh.field.add("Min", field_tag_counter)
    # gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thresh_tip, f_thresh_sio2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # # Physical groups
    # gmsh.model.addPhysicalGroup(2, [surf_sic], 1, "sic")
    # gmsh.model.addPhysicalGroup(2, [surf_sio2], 2, "sio2")
    # gmsh.model.addPhysicalGroup(2, [surf_vac], 3, "vacuum")
    # gmsh.model.addPhysicalGroup(1, [q1p1], 11, "ground")
    # gmsh.model.addPhysicalGroup(1, [p1p2, Op2, tip1], 12, "axis")
    # gmsh.model.addPhysicalGroup(1, [q2q1, q2q3, q4q3], 13, "far-field")
    # gmsh.model.addPhysicalGroup(1, [tiparc, t2t3], 14, "tip")
    # gmsh.model.addPhysicalGroup(1, [q3O], 15, "sic_sio2_interface")

    # gmsh.model.mesh.generate(2)
    # partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    # msh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2, partitioner=partitioner)
    # gmsh.finalize()

    # return msh, cell_tags, facet_tags

    # ジオメトリの無次元化 (代表長さ L_c = 1 nm)
    L_c = 1e-9
    R_dimless = geom.region_radius * 1e-9 / L_c
    sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    sic_depth_dimless = (geom.l_vac - geom.l_sio2) * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_slope_angle = 15 * np.pi / 180  # 75 deg tip angle -> 15 deg slope

    # SplineGeometry で2D軸対称ジオメトリを構築
    geo = SplineGeometry()

    # 点の定義 (中心軸上 r=0)
    p1 = geo.AppendPoint(0, -sic_depth_dimless - sio2_depth_dimless)  # SiC底部
    p2 = geo.AppendPoint(0, -sio2_depth_dimless)  # SiC/SiO2界面
    O = geo.AppendPoint(0, 0)  # SiO2/真空界面（原点）

    # 探針の点
    # 円弧の中心座標（計算用）
    tipO_x = 0
    tipO_y = tip_z_dimless + tip_radius_dimless
    
    tip1 = geo.AppendPoint(0, tip_z_dimless)  # 探針最下点（円弧始点）
    
    # 円弧終点
    tip2 = geo.AppendPoint(
        tip_radius_dimless * np.cos(tip_slope_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.sin(tip_slope_angle))
    )
    
    # 円弧上の中間点（spline3用）: 始点と終点の中間角度での点
    mid_angle = tip_slope_angle / 2  # 90度から始まり、90-tip_slope_angleまでの中間
    tipM = geo.AppendPoint(
        tip_radius_dimless * np.sin(mid_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.cos(mid_angle))
    )
    
    # 探針の円錐部分終点
    tip3 = geo.AppendPoint(
        tip_radius_dimless * np.cos(tip_slope_angle)
        + (vac_depth_dimless - tip_z_dimless 
           - tip_radius_dimless * (1 - np.sin(tip_slope_angle))) * np.tan(tip_slope_angle),
        vac_depth_dimless
    )

    # 遠方境界上の点
    q1 = geo.AppendPoint(R_dimless, -sic_depth_dimless - sio2_depth_dimless)  # SiC底部右端
    q2 = geo.AppendPoint(R_dimless, -sio2_depth_dimless)  # SiC/SiO2界面右端
    q3 = geo.AppendPoint(R_dimless, 0)  # SiO2/真空界面右端
    q4 = geo.AppendPoint(R_dimless, vac_depth_dimless)  # 真空層上端

    # 境界の定義（共有境界は一度だけ定義し、leftdomainとrightdomainで両側を指定）
    # 各領域は反時計回りに閉じたループを形成
    
    # Domain 1 (SiC): p1 → p2 → q2 → q1 → p1
    geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="axis")
    geo.Append(["line", p2, q2], leftdomain=1, rightdomain=2)  # SiC/SiO2共有境界
    geo.Append(["line", q2, q1], leftdomain=1, rightdomain=0, bc="far-field")
    geo.Append(["line", q1, p1], leftdomain=1, rightdomain=0, bc="ground")
    
    # Domain 2 (SiO2): p2 → O → q3 → q2 (→ p2は既に定義済み)
    geo.Append(["line", p2, O], leftdomain=2, rightdomain=0, bc="axis")
    geo.Append(["line", O, q3], leftdomain=2, rightdomain=3, bc="sic_sio2_interface")  # SiO2/vacuum共有境界
    geo.Append(["line", q3, q2], leftdomain=2, rightdomain=0, bc="far-field")
    # p2→q2は既に定義済み（Domain 1で）
    
    # Domain 3 (vacuum): O → tip1 → tipM → tip2 → tip3 → q4 → q3 (→ Oは既に定義済み)
    geo.Append(["line", O, tip1], leftdomain=3, rightdomain=0, bc="axis")
    geo.Append(["spline3", tip1, tipM, tip2], leftdomain=3, rightdomain=0, bc="tip")
    geo.Append(["line", tip2, tip3], leftdomain=3, rightdomain=0, bc="tip")
    geo.Append(["line", tip3, q4], leftdomain=3, rightdomain=0)
    geo.Append(["line", q4, q3], leftdomain=3, rightdomain=0, bc="far-field")
    # O→q3は既に定義済み（Domain 2で）

    # 領域のマテリアル名を設定
    geo.SetMaterial(1, "sic")
    geo.SetMaterial(2, "sio2")
    geo.SetMaterial(3, "vacuum")

    # ジオメトリ情報をデバッグ出力
    logger.info("Geometry defined with:")
    logger.info(f"  - Domain radius: {R_dimless:.2f}")
    logger.info(f"  - SiC depth: {sic_depth_dimless:.2f}")
    logger.info(f"  - SiO2 thickness: {sio2_depth_dimless:.2f}")
    logger.info(f"  - Vacuum height: {vac_depth_dimless:.2f}")
    logger.info(f"  - Tip radius: {tip_radius_dimless:.2f}")
    logger.info(f"  - Tip height: {tip_z_dimless:.2f}")
    logger.info("Starting mesh generation...")
    
    # メッシュサイズの制御
    # SplineGeometry (2D) では、SetDomainMaxH で領域ごとのメッシュサイズを設定
    # または GenerateMesh の maxh パラメータでグローバル制御
    # ポイント毎の制御は PointInfo を使うが、ここでは簡潔さのため領域制御を使用
    
    # 領域ごとのメッシュサイズ設定（オプション）
    # geo.SetDomainMaxH(1, 5.0)  # SiC: やや粗く
    # geo.SetDomainMaxH(2, 2.5)  # SiO2: 中間
    # geo.SetDomainMaxH(3, 1.0)  # 真空(探針含む): 細かく
    
    # メッシュ生成（グローバルな最大要素サイズ）
    # 最初は粗いメッシュでテスト（maxh=50.0 → 後で調整）
    # 探針先端付近は自動的に細かくなる傾向がある
    ngmesh = geo.GenerateMesh(maxh=50.0)
    logger.info("Mesh generation completed")
    
    # NGSolveのメッシュオブジェクトに変換
    mesh = ng.Mesh(ngmesh)
    
    logger.info(f"Mesh generated with {mesh.ne} elements and {mesh.nv} vertices")
    
    return mesh


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
    msh = create_mesh(geom)
    # 無次元化のための代表スケール
    L_c = 1e-9  # 代表長さ [1 nm]
    V_c = const.k * phys.T / const.e  # 代表電位 (熱電圧) [V]

    # # 関数空間とテスト/トライアル関数
    # V = fem.functionspace(msh, ("Lagrange", 1))
    # u = fem.Function(V, name="potential_dimless")
    # v = ufl.TestFunction(V)

    # # 比誘電率を定義
    # epsilon_r = fem.Function(fem.functionspace(msh, ("DG", 0)), name="relative_permittivity")
    # epsilon_r.x.array[cell_tags.find(1)] = phys.eps_sic
    # epsilon_r.x.array[cell_tags.find(2)] = phys.eps_sio2
    # epsilon_r.x.array[cell_tags.find(3)] = phys.eps_vac

    # # ホモトピーパラメータ
    # homotopy_charge = fem.Constant(msh, ScalarType(0.0))
    # homotopy_sigma = fem.Constant(msh, ScalarType(0.0))

    # # 弱形式の定義
    # F, J = _setup_weak_form(u, v, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, cell_tags, facet_tags)
    
    # # 境界条件
    # dofs_ground = fem.locate_dofs_topological(V, 1, facet_tags.find(11))
    # bc_ground = fem.dirichletbc(ScalarType(0), dofs_ground, V)
    # dofs_tip = fem.locate_dofs_topological(V, 1, facet_tags.find(14))
    # bc_tip = fem.dirichletbc(ScalarType(V_tip / V_c), dofs_tip, V)
    # bcs = [bc_ground, bc_tip]

    # 関数空間とテスト/トライアル関数
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")
    
    # 比誘電率を定義
    epsilon_r = ng.CoefficientFunction([phys.eps_sic, phys.eps_sio2, phys.eps_vac])
    
    # ホモトピーパラメータ
    homotopy_charge = ng.Parameter(0.0)
    homotopy_sigma = ng.Parameter(0.0)

    # 弱形式の定義
    a, f = _setup_weak_form(fes, u, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, msh)
    
     # 境界条件の設定
    # NGSolveでは境界条件は文字列名で指定
    u.Set(0, definedon=msh.Boundaries("ground"))  # ground境界で u = 0
    u.Set(V_tip / V_c, definedon=msh.Boundaries("tip"))  # tip境界で u = V_tip/V_c
    
    # 線形問題 (ラプラス方程式) で初期解を計算
    _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh)
    
    # ホモトピー法で非線形問題を解く
    solve_with_homotopy(a, f, u, fes, msh, homotopy_charge, homotopy_sigma)

    save_results(msh, u, epsilon_r, V_c, out_dir)

# def _setup_weak_form(u, v, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, cell_tags, facet_tags):
#     """弱形式とヤコビアンを定義する"""
#     x = ufl.SpatialCoordinate(u.function_space.mesh)
#     r = x[0] # 円筒座標系の半径
#     dx = ufl.Measure("dx", domain=u.function_space.mesh, subdomain_data=cell_tags)
#     ds = ufl.Measure("ds", domain=u.function_space.mesh, subdomain_data=facet_tags)
#     dS = ufl.Measure("dS", domain=u.function_space.mesh, subdomain_data=facet_tags)

#     # 無次元化された係数
#     C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
#     C_Nc = homotopy_charge * C0 * phys.Nc
#     C_Nv = homotopy_charge * C0 * phys.Nv
#     C_Nd = homotopy_charge * C0 * phys.Nd
#     sigma_s_target = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)
#     sigma_s_dimless = homotopy_sigma * sigma_s_target

#     # 無次元化されたエネルギー準位
#     Ef_dimless = phys.Ef / V_c
#     Ec_dimless = phys.Ec / V_c
#     Ev_dimless = phys.Ev / V_c
#     Ed_dimless = phys.Ed / V_c
    
#     # 電荷密度項（数値的安定性のために電位をクリップ）
#     u_clip = ufl.max_value(ufl.min_value(u, 160.0), -160.0)
    
#     def fermi_dirac_ufl(x):
#         return ufl.conditional(
#             ufl.gt(x, 25),
#             (2 / np.sqrt(np.pi)) * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
#             ufl.exp(x) / (1 + 0.27 * ufl.exp(x)),
#         )

#     n_term = C_Nc * fermi_dirac_ufl((Ef_dimless - Ec_dimless) + u_clip)
#     p_term = C_Nv * fermi_dirac_ufl((Ev_dimless - Ef_dimless) - u_clip)
#     Ndp_term = C_Nd / (1 + 2 * ufl.exp((Ef_dimless - Ed_dimless) + u_clip))
#     rho_dimless = p_term + Ndp_term - n_term

#     # ポアソン方程式の弱形式
#     a = ufl.inner(epsilon_r * ufl.grad(u), ufl.grad(v)) * r
#     L_bulk = rho_dimless * v * r
#     L_surface = sigma_s_dimless * ufl.avg(v) * r
#     lambda_ff = 1 / (geom.region_radius * 1e-9 / L_c)
    
#     F = (a * dx - L_bulk * dx(1) - L_surface * dS(15) + epsilon_r * lambda_ff * u * v * r * ds(13))
#     J = ufl.derivative(F, u)
    
#     return F, J

def _setup_weak_form(fes, u, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, msh):
    """弱形式を定義する"""
    
    # テスト関数
    v = fes.TestFunction()
    
    # 座標と円筒座標系の半径
    x, y = ng.x, ng.y
    r = x  # 円筒座標系の半径方向
    
    # 無次元化された係数
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    C_Nc = homotopy_charge * C0 * phys.Nc
    C_Nv = homotopy_charge * C0 * phys.Nv
    C_Nd = homotopy_charge * C0 * phys.Nd
    sigma_s_target = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)
    sigma_s_dimless = homotopy_sigma * sigma_s_target

    # 無次元化されたエネルギー準位
    Ef_dimless = phys.Ef / V_c
    Ec_dimless = phys.Ec / V_c
    Ev_dimless = phys.Ev / V_c
    Ed_dimless = phys.Ed / V_c
    
    # 電荷密度項（数値的安定性のために電位をクリップ）
    u_clip = ng.IfPos(u - 160.0, 160.0, ng.IfPos(-160.0 - u, -160.0, u))
    
    def fermi_dirac_ng(x):
        """NGSolveでのフェルミ・ディラック積分の近似"""
        return ng.IfPos(
            x - 25,
            (2 / np.sqrt(np.pi)) * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
            ng.exp(x) / (1 + 0.27 * ng.exp(x))
        )

    n_term = C_Nc * fermi_dirac_ng((Ef_dimless - Ec_dimless) + u_clip)
    p_term = C_Nv * fermi_dirac_ng((Ev_dimless - Ef_dimless) - u_clip)
    Ndp_term = C_Nd / (1 + 2 * ng.exp((Ef_dimless - Ed_dimless) + u_clip))
    rho_dimless = p_term + Ndp_term - n_term

    # 遠方境界のロビン境界条件パラメータ
    lambda_ff = 1 / (geom.region_radius * 1e-9 / L_c)
    
    # 双線形形式 (ポアソン方程式の左辺)
    a = ng.BilinearForm(fes, symmetric=False)
    a += epsilon_r * ng.grad(u) * ng.grad(v) * r * ng.dx
    # 遠方境界でのロビン境界条件
    a += epsilon_r * lambda_ff * u * v * r * ng.ds("far-field")
    
    # 線形形式 (ポアソン方程式の右辺)
    f = ng.LinearForm(fes)
    # SiC領域での空間電荷密度 (domain 1)
    f += rho_dimless * v * r * ng.dx(definedon=msh.Materials("sic"))
    # SiC/SiO2界面での面電荷密度
    # NGSolveでは内部境界での積分は dx(skeleton=True) または特殊な扱いが必要
    # ここでは境界に沿った積分として扱う
    f += sigma_s_dimless * v * r * ng.ds("sic_sio2_interface")
    
    return a, f

# def _warm_start_with_linear_solve(V, u, epsilon_r, V_tip, V_c, bc_ground, dofs_tip, geom):
#     """電圧を徐々に印加しながら線形問題を解き、非線形問題の初期値を準備する"""
#     if comm.rank == 0:
#         logger.info("Performing warm-start with linear Poisson equation...")
    
#     w, v = ufl.TrialFunction(V), ufl.TestFunction(V)
#     x = ufl.SpatialCoordinate(V.mesh)
#     r = x[0]
#     lambda_ff = 1 / (geom.region_radius * 1e-9 / 1e-9)

#     a_lin = ufl.inner(epsilon_r * ufl.grad(w), ufl.grad(v)) * r * ufl.dx + \
#             epsilon_r * lambda_ff * w * v * r * ufl.ds(13)
#     L_lin = fem.Constant(V.mesh, ScalarType(0.0)) * v * ufl.dx
    
#     u.x.array[:] = 0.0
#     uh = fem.Function(V)
    
#     # 電圧を少しずつ上げて解くことで安定性を確保
#     voltages_warmup = np.linspace(0.0, V_tip, 5)[1:]
#     for v_val in voltages_warmup:
#         # bcs[-1].g.value = ScalarType(v_val / V_c) # Update tip voltage
#         bc_tip = fem.dirichletbc(ScalarType(v_val / V_c), dofs_tip, V)
#         bcs = [bc_ground, bc_tip]
#         # problem = LinearProblem(a_lin, L_lin, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#         problem = LinearProblem(a_lin, L_lin, bcs=bcs, petsc_options={"ksp_type": "gmres", "pc_type": "ilu"}) # optimize for GPU
#         uh = problem.solve()
#         u.x.array[:] = uh.x.array
#         if comm.rank == 0:
#             logger.info(f"  [Linear Warm-up] Solved at V_tip = {v_val:.2f} V")

def _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh):
    """電圧を徐々に印加しながら線形問題を解き、非線形問題の初期値を準備する"""
    logger.info("Performing warm-start with linear Poisson equation...")
    
    # テスト関数とトライアル関数
    w = fes.TrialFunction()
    v = fes.TestFunction()
    
    # 座標と円筒座標系の半径
    x, y = ng.x, ng.y
    r = x  # 円筒座標系の半径方向
    
    # 遠方境界のロビン境界条件パラメータ
    lambda_ff = 1 / (geom.region_radius * 1e-9 / 1e-9)
    
    # 線形ポアソン方程式の双線形形式
    a_lin = ng.BilinearForm(fes)
    a_lin += epsilon_r * ng.grad(w) * ng.grad(v) * r * ng.dx
    a_lin += epsilon_r * lambda_ff * w * v * r * ng.ds("far-field")
    
    # 右辺 (ゼロ)
    f_lin = ng.LinearForm(fes)
    # 右辺は0なので項を追加しない
    
    # 初期化
    u.vec[:] = 0.0
    uh = ng.GridFunction(fes)
    
    # 電圧を少しずつ上げて解くことで安定性を確保
    voltages_warmup = np.linspace(0.0, V_tip, 5)[1:]
    for v_val in voltages_warmup:
        # 境界条件を更新
        u.Set(0, definedon=msh.Boundaries("ground"))
        u.Set(v_val / V_c, definedon=msh.Boundaries("tip"))
        
        # 双線形形式と線形形式を組み立て
        a_lin.Assemble()
        f_lin.Assemble()
        
        # Dirichlet境界条件を考慮して線形システムを解く
        u.vec.data = a_lin.mat.Inverse(fes.FreeDofs(definedon=msh.Boundaries("ground|tip")), inverse="sparsecholesky") * f_lin.vec
        
        logger.info(f"  [Linear Warm-up] Solved at V_tip = {v_val:.2f} V")

def solve_with_homotopy(F, J, u, bcs, homotopy_charge, homotopy_sigma):
    """
    2段階のホモトピー法を用いて非線形問題を解く
    Stage 1: 空間電荷を徐々に導入
    Stage 2: 界面電荷を徐々に導入
    """
    # Stage 1: 空間電荷の導入
    homotopy_sigma.value = 0.0
    _solve_homotopy_stage(
        F, J, u, bcs, homotopy_charge, stage_name="Space Charge"
    )

    # Stage 2: 界面電荷の導入
    homotopy_charge.value = 1.0
    _solve_homotopy_stage(
        F, J, u, bcs, homotopy_sigma, stage_name="Interface Charge"
    )


# def _solve_homotopy_stage(F, J, u, bcs, homotopy_param, stage_name: str):
#     """ホモトピー法の1ステージを実行する共通関数"""
#     if comm.rank == 0:
#         logger.info(f"--- Starting Homotopy Stage: {stage_name} ---")

#     theta = 0.0
#     step = 0.1
#     min_step = 1e-4
#     uh = fem.Function(u.function_space)
#     uh.x.array[:] = u.x.array

#     problem = NonlinearProblem(F, u, bcs=bcs, J=J)
#     solver = NewtonSolver(comm, problem)
#     solver.convergence_criterion = "residual"
#     solver.rtol = 1e-8
#     solver.atol = 1e-10
#     solver.max_it = 50
#     solver.relaxation_parameter = 0.7

#     ksp = solver.krylov_solver
#     # ksp.setType("preonly")
#     ksp.setType("gmres") # optimize for GPU
#     pc = ksp.getPC()
#     # pc.setType("lu")
#     pc.setType("ilu") # optimize for GPU

#     while theta < 1.0 - 1e-12:
#         trial = min(1.0, theta + step)
#         homotopy_param.value = ScalarType(trial)
        
#         try:
#             n, converged = solver.solve(u)
#         except RuntimeError:
#             converged = False
        
#         if converged:
#             theta = trial
#             uh.x.array[:] = u.x.array  # 収束した解をバックアップ
#             if comm.rank == 0:
#                 logger.info(f"  [{stage_name} Homotopy: θ={theta:.3f}] Converged in {n} Newton iterations.")
#             # 収束が速ければステップサイズを増やす
#             if n <= 3 and step < 0.5:
#                 step *= 1.5
#         else:
#             u.x.array[:] = uh.x.array  # 失敗したので安定した解に戻す
#             step *= 0.5
#             if comm.rank == 0:
#                 logger.warning(
#                     f"  [{stage_name} Homotopy: θ→{trial:.3f}] Failed to converge. Reducing step to {step:.4f}."
#                 )
#             if step < min_step:
#                 raise RuntimeError(f"Homotopy stage '{stage_name}' failed: step size became too small.")

def solve_with_homotopy(a, f, u, fes, msh, homotopy_charge, homotopy_sigma):
    """
    2段階のホモトピー法を用いて非線形問題を解く
    Stage 1: 空間電荷を徐々に導入
    Stage 2: 界面電荷を徐々に導入
    """
    # Stage 1: 空間電荷の導入
    homotopy_sigma.Set(0.0)
    _solve_homotopy_stage(
        a, f, u, fes, msh, homotopy_charge, stage_name="Space Charge"
    )

    # Stage 2: 界面電荷の導入
    homotopy_charge.Set(1.0)
    _solve_homotopy_stage(
        a, f, u, fes, msh, homotopy_sigma, stage_name="Interface Charge"
    )

def _solve_homotopy_stage(a, f, u, fes, msh, homotopy_param, stage_name: str):
    """ホモトピー法の1ステージを実行する共通関数"""
    logger.info(f"--- Starting Homotopy Stage: {stage_name} ---")

    theta = 0.0
    step = 0.1
    min_step = 1e-4
    
    # バックアップ用のGridFunction
    uh = ng.GridFunction(fes)
    uh.vec.data = u.vec
    
    # Newton法のパラメータ
    max_iterations = 50
    rtol = 1e-8
    atol = 1e-10
    damping = 0.7

    while theta < 1.0 - 1e-12:
        trial = min(1.0, theta + step)
        homotopy_param.Set(trial)
        
        # 双線形形式と線形形式を再組み立て（パラメータ依存のため）
        a.Assemble()
        f.Assemble()
        
        # Newton法で非線形問題を解く
        converged = False
        try:
            # NGSolveのNewton法
            # 残差ベクトル r = A*u - f
            res = u.vec.CreateVector()
            du = u.vec.CreateVector()
            
            for iteration in range(max_iterations):
                # 残差を計算
                res.data = a.mat * u.vec - f.vec
                
                # 残差のノルム
                res_norm = ng.sqrt(ng.InnerProduct(res, res))
                
                if iteration == 0:
                    res_norm0 = res_norm
                
                # 収束判定
                if res_norm < atol or (res_norm0 > 0 and res_norm / res_norm0 < rtol):
                    converged = True
                    n = iteration
                    break
                
                # ヤコビアン行列で線形システムを解く
                # A * du = -res
                freedofs = fes.FreeDofs(msh.Boundaries("ground|tip"))
                du.data = a.mat.Inverse(freedofs, inverse="sparsecholesky") * (-res)
                
                # 緩和付き更新
                u.vec.data += damping * du
            
        except Exception as e:
            logger.warning(f"  [{stage_name} Homotopy: θ→{trial:.3f}] Exception: {e}")
            converged = False
        
        if converged:
            theta = trial
            uh.vec.data = u.vec  # 収束した解をバックアップ
            logger.info(f"  [{stage_name} Homotopy: θ={theta:.3f}] Converged in {n} Newton iterations.")
            # 収束が速ければステップサイズを増やす
            if n <= 3 and step < 0.5:
                step *= 1.5
        else:
            u.vec.data = uh.vec  # 失敗したので安定した解に戻す
            step *= 0.5
            logger.warning(
                f"  [{stage_name} Homotopy: θ→{trial:.3f}] Failed to converge. Reducing step to {step:.4f}."
            )
            if step < min_step:
                raise RuntimeError(f"Homotopy stage '{stage_name}' failed: step size became too small.")

def save_results(msh, u, epsilon_r, V_c, out_dir: str):
    """
    結果をNGSolve形式/汎用形式で保存

    保存物:
      mesh.vol            Netgenメッシュ
      solution.vtu        VTK (Paraview等)
      u_dimless.npy       無次元ポテンシャル DOF ベクトル
      u_volts.npy         [V] ポテンシャル DOF ベクトル
      epsilon_r.json      各マテリアルの誘電率
      metadata.json       代表スケールなど
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. メッシュ保存 (Netgen形式)
    msh.ngmesh.Save(os.path.join(out_dir, "mesh.vol"))

    # 2. GridFunction の DOF ベクトル保存
    u_np = u.vec.FV().NumPy()
    np.save(os.path.join(out_dir, "u_dimless.npy"), u_np)
    np.save(os.path.join(out_dir, "u_volts.npy"), u_np * V_c)

    # 3. 誘電率（領域順に）を保存
    # epsilon_r は piecewise なので材料名と値の対応を JSON で保持
    mat_names = list(msh.GetMaterials())
    # 与えた順 [sic, sio2, vacuum] を仮定
    eps_map = {name: val for name, val in zip(mat_names, [float(v) for v in epsilon_r.components])} \
              if hasattr(epsilon_r, "components") else {name: None for name in mat_names}
    with open(os.path.join(out_dir, "epsilon_r.json"), "w") as f:
        json.dump(eps_map, f, indent=2)

    # 4. メタデータ
    meta = {
        "V_c": V_c,
        "ndof": u.space.ndof,
        "fes": "H1",
        "order": u.space.globalorder,
        "materials": mat_names,
        "boundaries": list(msh.GetBoundaries()),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 5. VTK 出力
    try:
        from ngsolve import VTKOutput
        vtk = VTKOutput(ma=msh,
                        coefs=[u],
                        names=["potential_dimless"],
                        filename=os.path.join(out_dir, "solution"),
                        subdivision=0)
        vtk.Do()
    except Exception as e:
        logger.warning(f"VTK export failed: {e}")

    logger.info(f"Saved results to {out_dir}")

def load_results(out_dir: str, geom: GeometricParameters, V_c: float):
    """
    保存された結果を再読み込み

    手順:
      1. create_mesh(geom) で同一メッシュを再生成 (パラメータ不変前提)
      2. ndof 整合を確認
      3. DOF ベクトルを代入
    戻り値: (mesh, u, u_volts_numpy)

    Usage:
    >>> # 保存後
    >>> save_results(msh, u, epsilon_r, V_c, "out")
    >>> # 別プロセス/後処理スクリプトで
    >>> geom = GeometricParameters()
    >>> msh2, u_loaded, uV = load_results("out", geom, V_c)
    """
    # メッシュ再生成（パラメータ変更していると整合しない）
    msh = create_mesh(geom)
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")

    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("metadata.json が見つかりません")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["ndof"] != fes.ndof:
        raise RuntimeError(f"DOF数不一致: saved={meta['ndof']} current={fes.ndof}")

    u_dim_path = os.path.join(out_dir, "u_dimless.npy")
    if not os.path.isfile(u_dim_path):
        raise FileNotFoundError("u_dimless.npy が見つかりません")

    u_vec = np.load(u_dim_path)
    if u_vec.size != fes.ndof:
        raise RuntimeError("ベクトル長が現在のFESと一致しません")

    u.vec.FV().NumPy()[:] = u_vec

    # 物理単位のポテンシャル
    u_volts = u_vec * V_c

    logger.info("Loaded solution from disk")
    return msh, u, u_volts

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="2D Axisymmetric Poisson Solver for a Tip-on-Semiconductor System.")
    parser.add_argument("--V_tip", type=float, default=2.0, help="Tip voltage in Volts.")
    parser.add_argument("--tip_radius", type=float, default=45.0, help="Tip radius in nm.")
    parser.add_argument("--tip_height", type=float, default=8.0, help="Tip-sample distance in nm.")
    parser.add_argument("--l_sio2", type=float, default=5.0, help="Thickness of SiO2 layer in nm.")
    parser.add_argument("--Nd", type=float, default=1e16, help="Donor concentration in cm^-3.")
    parser.add_argument("--sigma_s", type=float, default=1e11, help="Surface charge density at SiC/SiO2 interface in cm^-2.")
    parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin.")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory for results.")
    parser.add_argument("--plot_fermi", action="store_true", help="Plot the Fermi level determination process.")
    args, _ = parser.parse_known_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.out_dir, exist_ok=True)

    # 物理パラメータの初期化 (単位をm^-3, m^-2に変換)
    phys_params = PhysicalParameters(
        T=args.T,
        Nd=args.Nd * 1e6, # cm^-3 -> m^-3
        sigma_s=args.sigma_s * 1e4, # cm^-2 -> m^-2
    )

    # フェルミ準位の計算
    Ef = find_fermi_level(phys_params, args.out_dir, plot=args.plot_fermi)
    phys_params.Ef = Ef

    # ジオメトリパラメータの初期化
    geom_params = GeometricParameters(
        l_sio2=args.l_sio2,
        tip_radius=args.tip_radius,
        tip_height=args.tip_height
    )

    # パラメータをJSONファイルに保存
    all_params = {
        "physical": asdict(phys_params),
        "geometric": asdict(geom_params),
        "simulation": {"V_tip": args.V_tip}
    }
    with open(os.path.join(args.out_dir, "parameters.json"), "w") as f:
        json.dump(all_params, f, indent=2)

    # FEMシミュレーションの実行
    run_fem_simulation(
        phys=phys_params,
        geom=geom_params,
        V_tip=args.V_tip,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()