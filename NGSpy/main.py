import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
import scipy.constants as const
from netgen.geom2d import SplineGeometry
from ngsolve.solvers import Newton
from scipy.optimize import brentq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
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
    n_tip_arc_points: int = 7  # 探針円弧部分の中間点数 (奇数)

    def __post_init__(self):
        assert self.n_tip_arc_points % 2 == 1, "n_tip_arc_points should be odd"


def fermi_dirac_integral(x: np.ndarray) -> np.ndarray:
    """フェルミ・ディラック積分の半整数次 (j=1/2) の近似式"""
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


def _plot_fermi_level_determination(
    params: PhysicalParameters, Ef: float, out_dir: str
):
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

    # ジオメトリの無次元化 (代表長さ L_c = 1 nm)
    L_c = 1e-9
    R_dimless = geom.region_radius * 1e-9 / L_c
    sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    vac_depth_dimless = geom.l_vac * 1e-9 / L_c
    sic_depth_dimless = (geom.l_vac - geom.l_sio2) * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_arc_angle = 75 * np.pi / 180  # 探針円弧の中心角
    n_middle_points = geom.n_tip_arc_points
    # SplineGeometry で2D軸対称ジオメトリを構築
    geo = SplineGeometry()

    # 点の定義 (中心軸上 r=0)
    p1 = geo.AppendPoint(0, -sic_depth_dimless - sio2_depth_dimless)  # SiC底部
    p2 = geo.AppendPoint(0, -sio2_depth_dimless)  # SiC/SiO2界面
    origin = geo.AppendPoint(0, 0)  # SiO2/真空界面 (原点)

    # 探針の先端
    tip1 = geo.AppendPoint(0, tip_z_dimless)  # 探針最下点 (円弧始点)

    # 円弧終点
    tip2 = geo.AppendPoint(
        tip_radius_dimless * np.sin(tip_arc_angle),
        tip_z_dimless + tip_radius_dimless * (1 - np.cos(tip_arc_angle)),
    )

    # 円弧上の中間点 (spline3用)
    tipMlst = [
        geo.AppendPoint(
            tip_radius_dimless * np.sin(mid_angle),
            tip_z_dimless + tip_radius_dimless * (1 - np.cos(mid_angle)),
        )
        for mid_angle in np.linspace(0, tip_arc_angle, n_middle_points + 2)[1:-1]
    ]

    # 探針の円錐部分終点
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

    # 遠方境界上の点
    q1 = geo.AppendPoint(
        R_dimless, -sic_depth_dimless - sio2_depth_dimless
    )  # SiC底部右端
    q2 = geo.AppendPoint(R_dimless, -sio2_depth_dimless)  # SiC/SiO2界面右端
    q3 = geo.AppendPoint(R_dimless, 0)  # SiO2/真空界面右端
    q4 = geo.AppendPoint(R_dimless, vac_depth_dimless)  # 真空層上端

    # 境界の定義 (ref.pyのパターンに厳密に従う)
    # 重要: すべての境界を一度だけ定義し、共有境界はleftdomain/rightdomainで指定

    # Bottom rectangle (SiC, domain=1): p1 → p2 → q2 → q1 → p1
    geo.Append(["line", p1, p2], bc="axis", leftdomain=0, rightdomain=1)
    geo.Append(["line", p2, q2], bc="sic/sio2", leftdomain=2, rightdomain=1)
    geo.Append(["line", q2, q1], bc="far-field", leftdomain=0, rightdomain=1)
    geo.Append(["line", q1, p1], bc="ground", leftdomain=0, rightdomain=1)

    # Middle rectangle (SiO2, domain=2): p2 → origin → q3 → q2
    # p2→q2 は既に定義済み
    geo.Append(["line", origin, p2], bc="axis", leftdomain=2, rightdomain=0)
    geo.Append(["line", q2, q3], bc="far-field", leftdomain=2, rightdomain=0)
    geo.Append(["line", q3, origin], bc="sio2/vacuum", leftdomain=2, rightdomain=3)

    # Top with tip (vacuum, domain=3): origin → tip1 → tip2 → tip3 → q4 → q3
    geo.Append(["line", origin, tip1], bc="axis", leftdomain=0, rightdomain=3)
    for i in range(0, len(tipMlst), 2):
        points = [
            tip1 if i == 0 else tipMlst[i - 1],
            tipMlst[i],
            tip2 if i == len(tipMlst) - 1 else tipMlst[i + 1],
        ]
        geo.Append(["spline3", *points], bc="tip", leftdomain=0, rightdomain=3)
    geo.Append(["line", tip2, tip3], bc="tip", leftdomain=0, rightdomain=3)
    geo.Append(["line", tip3, q4], bc="top", leftdomain=0, rightdomain=3)
    geo.Append(["line", q4, q3], bc="far-field", leftdomain=0, rightdomain=3)
    # O→q3 は既に定義済み

    # 材料の定義
    geo.SetMaterial(1, "sic")
    geo.SetMaterial(2, "sio2")
    geo.SetMaterial(3, "vac")

    # ジオメトリ情報をデバッグ出力
    logger.info("Geometry defined with:")
    logger.info(f"  - Domain radius: {R_dimless:.2f}")
    logger.info(f"  - SiC depth: {sic_depth_dimless:.2f}")
    logger.info(f"  - SiO2 thickness: {sio2_depth_dimless:.2f}")
    logger.info(f"  - Vacuum height: {vac_depth_dimless:.2f}")
    logger.info(f"  - Tip radius: {tip_radius_dimless:.2f}")
    logger.info(f"  - Tip height: {tip_z_dimless:.2f}")

    # ポイント数とセグメント数の確認
    logger.info("Checking geometry integrity...")
    logger.info(
        f"  - Number of points defined: {len([p1, p2, origin, tip1, *tipMlst, tip2, tip3, q1, q2, q3, q4])}"
    )

    logger.info("Starting mesh generation...")
    logger.info("  (This may take a while for complex geometries...)")

    # メッシュサイズの制御

    # 領域ごとのメッシュサイズ設定
    # geo.SetDomainMaxH(1, 5.0)  # SiC
    geo.SetDomainMaxH(2, 2.5)  # SiO2
    # geo.SetDomainMaxH(3, 1.0)  # 真空

    # メッシュ生成 (グローバルな最大要素サイズ)
    # 探針先端付近は自動的に細かくなる傾向がある
    ngmesh = geo.GenerateMesh(maxh=5)
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

    # 関数空間とテスト/トライアル関数
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")

    # 比誘電率を定義
    epsilon_r = ng.CoefficientFunction([phys.eps_sic, phys.eps_sio2, phys.eps_vac])

    # ホモトピーパラメータ
    homotopy_charge = ng.Parameter(0.0)
    homotopy_sigma = ng.Parameter(0.0)

    # 弱形式の定義
    a = _setup_weak_form(
        fes, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, msh
    )

    # 境界条件の設定
    # NGSolveでは境界条件は文字列名で指定
    u.Set(0, definedon=msh.Boundaries("ground"))  # ground境界で u = 0
    u.Set(V_tip / V_c, definedon=msh.Boundaries("tip"))  # tip境界で u = V_tip/V_c

    # 線形問題 (ラプラス方程式) で初期解を計算
    _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh)

    # ホモトピー法で非線形問題を解く
    solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma)

    save_results(msh, u, epsilon_r, V_c, out_dir)


def _setup_weak_form(
    fes, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma, geom, msh
):
    """NGSolveでの弱形式 (非線形項込み) を構築する"""

    uh, vh = fes.TnT()
    r = ng.x

    # 係数の前処理
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    Ef_dimless = phys.Ef / V_c
    Ec_dimless = phys.Ec / V_c
    Ev_dimless = phys.Ev / V_c
    Ed_dimless = phys.Ed / V_c

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
                "Ed_dimless": Ed_dimless,
                "lambda_ff": lambda_ff,
                "sigma_s_target": sigma_s_target,
            },
            indent=2,
        )
    )

    clip_potential = 120.0
    clip_exp = 40.0

    def clamp(val, bound):
        # return ng.IfPos(val - bound, bound, ng.IfPos(-bound - val, -bound, val))
        def ngtanh(x):
            e2x = ng.exp(2 * x)
            return (e2x - 1) / (e2x + 1)

        return bound * ngtanh(val / bound)

    def safe_exp(x):
        x_clip = clamp(x, clip_exp)
        return ng.exp(x_clip)

    def fermi_dirac_half(x):
        # x_clip = clamp(x, clip_exp)
        # high = (2 / np.sqrt(np.pi)) * (
        #     (2 / 3) * x_clip**1.5 + (np.pi**2 / 12) * x_clip ** (-0.5)
        # )
        # low = safe_exp(x_clip) / (1 + 0.27 * safe_exp(x_clip))
        # return ng.IfPos(x_clip - 25.0, high, low)
        x_clip = clamp(x, clip_exp)
        return safe_exp(x_clip)

    u_clip = clamp(uh, clip_potential)

    # 電荷密度 (SiC 領域のみで有効)
    n_term = C0 * phys.Nc * fermi_dirac_half((Ef_dimless - Ec_dimless) + u_clip)
    p_term = C0 * phys.Nv * fermi_dirac_half((Ev_dimless - Ef_dimless) - u_clip)
    Ndp_term = C0 * phys.Nd / (1 + 2 * safe_exp((Ef_dimless - Ed_dimless) + u_clip))
    rho_dimless = homotopy_charge * (p_term + Ndp_term - n_term)

    sigma_s_dimless = homotopy_sigma * sigma_s_target

    a = ng.BilinearForm(fes, symmetric=False)
    a += epsilon_r * ng.grad(uh) * ng.grad(vh) * r * ng.dx
    a += epsilon_r * lambda_ff * uh * vh * r * ng.ds("far-field")
    a += -rho_dimless * vh * r * ng.dx(definedon=msh.Materials("sic"))
    a += -sigma_s_dimless * vh * r * ng.ds("sic/sio2")

    return a


def _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh):
    """電圧を徐々に印加しながら線形問題を解き、非線形問題の初期値を準備する"""
    logger.info("Performing warm-start with linear Poisson equation...")

    # テスト関数とトライアル関数
    w = fes.TrialFunction()
    v = fes.TestFunction()

    # 座標と円筒座標系の半径
    r = ng.x

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
        # 境界条件を設定 (初期化)
        uh = ng.GridFunction(fes)
        uh.Set(0, definedon=msh.Boundaries("ground"))
        uh.Set(v_val / V_c, definedon=msh.Boundaries("tip"))

        # 双線形形式と線形形式を組み立て
        a_lin.Assemble()
        f_lin.Assemble()

        # 右辺を境界条件で修正
        r = f_lin.vec.CreateVector()
        r.data = f_lin.vec - a_lin.mat * uh.vec

        # 自由度を取得 (Dirichlet境界以外)
        freedofs = fes.FreeDofs()
        freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
        freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))

        # Dirichlet境界条件を考慮して線形システムを解く
        uh.vec.data += a_lin.mat.Inverse(freedofs, inverse="sparsecholesky") * r

        # 結果をuにコピー
        u.vec.data = uh.vec

        logger.info(f"  [Linear Warm-up] Solved at V_tip = {v_val:.2f} V")


def solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma):
    """ホモトピー法 (2段階) で非線形問題を解く"""

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
        maxerr=1e-10,
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

    # 3. 誘電率 (領域順に) を保存
    # epsilon_r は piecewise なので材料名と値の対応を JSON で保持
    mat_names = list(msh.GetMaterials())
    # 与えた順 [sic, sio2, vacuum]
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
    # メッシュ再生成 (パラメータ変更していると整合しない)
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
        default=1e11,
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

    # 出力ディレクトリの作成
    os.makedirs(args.out_dir, exist_ok=True)

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
    all_params = {
        "physical": asdict(phys_params),
        "geometric": asdict(geom_params),
        "simulation": {"V_tip": args.V_tip},
    }
    with open(os.path.join(args.out_dir, "parameters.json"), "w") as f:
        json.dump(all_params, f, indent=2)

    # FEMシミュレーションの実行
    run_fem_simulation(
        phys=phys_params, geom=geom_params, V_tip=args.V_tip, out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
