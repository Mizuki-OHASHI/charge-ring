# -*- coding: utf-8 -*-
# SiC 内のみ非線形 Poisson-Boltzmann (PB) を導入し Newton() で解く版
# 円筒座標（軸対称）対応：弱形式に回転体体積要素を導入
# ★ 原点(軸)の"つぶれ"対策：r=0 での自然境界(Neumann)を弱形式で滑らかに実現
#   - |x| で生じる尖りを回避するため、r = sqrt(x^2 + ε^2) の"滑らかな半径"を用いる
#   - 軸近傍での過度な勾配を抑える微小安定化（任意・デフォルトOFF）も用意
#
# 〈弱形式〉
#   ∫_Ω (2π r) ε ∇u·∇v dΩ + ∫_Ω (2π r) (PB反応) v dΩ = 0
#   r := sqrt(x^2 + ε^2)（ε→0で標準軸対称に一致）⇒ 原点で自然に ∂u/∂r = 0（Neumann）を実現

import math
import csv
import numpy as np
import pandas as pd
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.webgui import Draw
from ngsolve.solvers import *
import matplotlib.pyplot as plt
from pathlib import Path
import datetime, os
import time
from netgen.geom2d import unit_square

t1 = time.time()

# ----------------------------------------
# 1)── パラメータ定義（関数外）
# ----------------------------------------
# 水平方向の範囲
X_MIN, X_MAX = -500.0, +500.0  # [nm]
# 真空／半導体 境界 y=0
Y_SEMIVAC = 0
# 鉛直方向の範囲
Y_MAX, Y_MIN = +200.0, -200  # [nm]
# SiO2膜厚
FILM_THICKNESS = 5.0  # [nm]

# Mesh幅
MESH_W = 5
# Tipバイアス電圧
V_s = 2  # [V]
# CPD
CPD = 0  # [V]
phi_T = V_s - CPD  # [V]

# 比誘電率（vac/top=1.0, SiO2=3.9, SiC=9.7）
EPS_R_1 = 3.9
EPS_R_2 = 9.7

# ── PB（SiC 内のみ）パラメータ：明示計算 ─────────────────────

# 物理定数（SI）
T = 296.0  # [K]
q = 1.602176634e-19  # [C]
kB = 1.380649e-23  # [J/K]
eps0 = 8.8541878128e-12  # [F/m]
VT = kB * T / q  # [V]

# --- ドーピング & 真性キャリア（任意に調整可） ---
ND_cm3 = 1.0e16  # ドナー [cm^-3]
NA_cm3 = 0.0  # アクセプタ [cm^-3]
NI_cm3 = 8.2e9  # 4H-SiC の真性キャリア密度（目安）。必要に応じて変更。

# cm^-3 → m^-3 変換
cm3_to_m3 = 1e6
ND = ND_cm3 * cm3_to_m3
NA = NA_cm3 * cm3_to_m3
NI = NI_cm3 * cm3_to_m3

# --- 中性条件 & 質量作用則から参照密度 ---
Delta = ND - NA
n_ref = 0.5 * (Delta + math.sqrt(Delta**2 + 4.0 * NI**2))
p_ref = NI**2 / max(n_ref, 1e-300)

# --- 反応項のスケール分離 ---
ref_sum = n_ref + p_ref
n_fac = float(n_ref / ref_sum)
p_fac = float(p_ref / ref_sum)

# チップ形状パラメータ（既存のまま）
RADIUS = 35  # [nm]
CONE_HALF_ANGLE = 15  # [deg]
TIP_TO_SURFACE = 8  # [nm]
N_ARC = 32  # 円弧分割数
Y_LEVEL = -5  # [nm]


# ----------------------------------------
# 幾何作成ヘルパ（既存のまま）
# ----------------------------------------
def TopRectangle(geom, X_MIN, X_MAX, Y_MAX, Y_SEMIVAC):
    top_pnts = [(X_MAX, Y_SEMIVAC), (X_MAX, Y_MAX), (X_MIN, Y_MAX), (X_MIN, Y_SEMIVAC)]
    top_nums = [geom.AppendPoint(*p) for p in top_pnts]
    lines = [
        (top_nums[0], top_nums[1], 10, 1, 0),
        (top_nums[1], top_nums[2], 10, 1, 0),
        (top_nums[2], top_nums[3], 10, 1, 0),
        (top_nums[3], top_nums[0], 10, 1, 2),
    ]
    for p0, p1, bn, ml, mr in lines:
        geom.Append(["line", p0, p1], bc=bn, leftdomain=ml, rightdomain=mr)
    return (geom, top_nums)


def PunchTip_calc(geom, R, cone_half_angle_deg, s, y_base, n_arc):
    alpha = math.radians(cone_half_angle_deg)
    phi = math.pi / 2 - alpha
    xj, yj = R * math.cos(+phi - math.pi / 2), R * math.sin(+phi - math.pi / 2) + s
    x_base = (yj - s + y_base) * math.tan(alpha) + xj
    pts = []
    for theta in np.linspace(-phi - math.pi / 2, +phi - math.pi / 2, n_arc):
        pts.append((R * math.cos(theta), R * math.sin(theta) + s))
    pts.append((xj, yj))
    pts.append(((xj + x_base) / 2, (yj + y_base) / 2))
    pts.append((x_base, +y_base))
    pts.append((0, +y_base))
    pts.append((-x_base, +y_base))
    pts.append((-xj, yj))
    disc_nums = [geom.AppendPoint(x, y) for x, y in pts]
    N = len(disc_nums)
    curves = []
    for i in range(0, N, 2):
        p0 = disc_nums[i]
        p1 = disc_nums[(i + 1) % N]
        p2 = disc_nums[(i + 2) % N]
        curves.append((p0, p1, p2, 1, 0, 1))
    for p0, p1, p2, bc, left, right in curves:
        geom.Append(["spline3", p0, p1, p2], bc=bc, leftdomain=left, rightdomain=right)
    return (geom, disc_nums)


def BottomRectangle(geom, topn, X_MIN, X_MAX, FILM_THICKNESS):
    bot_pnts = [(X_MIN, -FILM_THICKNESS), (X_MAX, -FILM_THICKNESS)]
    botn = [geom.AppendPoint(*p) for p in bot_pnts]
    lines = [
        (topn[3], botn[0], 10, 2, 0),
        (botn[0], botn[1], 10, 2, 3),
        (botn[1], topn[0], 10, 2, 0),
    ]
    for p0, p1, bn, ml, mr in lines:
        geom.Append(["line", p0, p1], bc=bn, leftdomain=ml, rightdomain=mr)
    return (geom, botn)


def BottomRectangle2(geom, topn, X_MIN, X_MAX, Y_MIN):
    bot_pnts = [(X_MIN, Y_MIN), (X_MAX, Y_MIN)]
    botn = [geom.AppendPoint(*p) for p in bot_pnts]
    lines = [
        (topn[0], botn[0], 10, 3, 0),
        (botn[0], botn[1], 2, 3, 0),  # bc=2 (接地)
        (botn[1], topn[1], 10, 3, 0),
    ]
    for p0, p1, bn, ml, mr in lines:
        geom.Append(["line", p0, p1], bc=bn, leftdomain=ml, rightdomain=mr)
    return (geom, botn)


def PunchPlate(geom):
    plate_pnts = [
        (X_MAX - 0.1, Y_MIN + 0.1),
        (X_MAX - 0.1, Y_MIN + 0.2),
        (X_MIN + 0.1, Y_MIN + 0.2),
        (X_MIN + 0.1, Y_MIN + 0.1),
    ]
    plate_nums = [geom.AppendPoint(*p) for p in plate_pnts]
    lines = [
        (plate_nums[0], plate_nums[1], 2, 0, 3),
        (plate_nums[1], plate_nums[2], 2, 0, 3),
        (plate_nums[2], plate_nums[3], 2, 0, 3),
        (plate_nums[3], plate_nums[0], 2, 0, 3),
    ]
    for p0, p1, bn, ml, mr in lines:
        geom.Append(["line", p0, p1], bc=bn, leftdomain=ml, rightdomain=mr)
    return geom


def MakeMesh():
    geometry = SplineGeometry()
    geometry, top = TopRectangle(geometry, X_MIN, X_MAX, Y_MAX, Y_SEMIVAC)
    geometry, dsc = PunchTip_calc(
        geometry,
        R=RADIUS,
        cone_half_angle_deg=CONE_HALF_ANGLE,
        s=TIP_TO_SURFACE + RADIUS,
        y_base=Y_MAX - 1,
        n_arc=N_ARC,
    )
    geometry, bot1 = BottomRectangle(geometry, top, X_MIN, X_MAX, FILM_THICKNESS)
    geometry, bot2 = BottomRectangle2(geometry, bot1, X_MIN, X_MAX, Y_MIN)
    # geometry = PunchPlate(geometry)
    return Mesh(geometry.GenerateMesh(maxh=MESH_W))


# ----------------------------------------
# メッシュ & 近似空間
# ----------------------------------------
mesh = MakeMesh()
V = H1(mesh, order=2, dirichlet=[1, 2])  # tip(bc=1), bottom plate(bc=2)
uh, vh = V.TnT()

# 領域ごとの相対誘電率（材料=1:vac, 2:SiO2, 3:SiC）
epsr = CoefficientFunction([1.0, EPS_R_1, EPS_R_2])

# ----------------------------------------
# 軸対称重み：r = sqrt(x^2 + ε^2) を採用して原点で滑らかに（Neumann）
# ----------------------------------------
# 軸重み：正則化を極小に/ほぼゼロに
AXIS_EPS_NM = 0.2  # 原点正則化長 [nm]（メッシュ幅以下の小値が目安）
r_smooth = sqrt(x * x + (AXIS_EPS_NM**2))  # [nm]
w_axi = 2.0 * math.pi * r_smooth  # 2πr

# （任意）軸近傍の微小安定化：∂u/∂x を弱く抑える
USE_AXIS_STAB = False
AXIS_BAND_NM = 2.0  # 軸近傍とみなす半幅 [nm]
AXIS_STAB_SCALE = 0.0 if not USE_AXIS_STAB else 0.1  # 0.0 で無効。0.05〜0.2程度

axis_band = IfPos(AXIS_BAND_NM - sqrt(x**2), 1.0, 0.0)

# ----------------------------------------
# 非線形形式：Laplace + PB(SiCのみ)  ※ dx に w_axi を掛ける
# ----------------------------------------
a = BilinearForm(V)  # Newton用

# Laplace（全領域）: ∫ (2π r) ε ∇u·∇v
a += w_axi * epsr * grad(uh) * grad(vh) * dx

# 軸近傍の微小安定化（任意）：∫ (2π r) * γ * I_axis * (∂u/∂x)(∂v/∂x)
if AXIS_STAB_SCALE > 0.0:
    a += w_axi * AXIS_STAB_SCALE * axis_band * (uh.Deriv()[0]) * (vh.Deriv()[0]) * dx

# PB 反応項（SiCのみ）
pb_weight_SiC_nm2 = (q / (eps0)) * (n_ref + p_ref) * 1e-18  # [1/nm^2]
pb_weight = CoefficientFunction([0.0, 0.0, pb_weight_SiC_nm2])

# --- ホモトピー因子（0→1 で非線形を徐々に導入） ---
hom = Parameter(0.0)  # Set() で書き換え

# --- 非線形反応項（線形化で (u/VT) に一致）---
lin_part = uh / VT
nonlin_part = -(p_fac * exp(-uh / VT) - n_fac * exp(uh / VT) + (n_fac - p_fac))

# 反応項（SiC のみ）: ∫ (2π r) pb_weight * [ (1-α)*(u/VT) + α*nonlin_part ] v
reaction = (1.0 - hom) * lin_part + hom * nonlin_part

# ▼ここをフラグでON/OFF
isENABLE_PB = True  # ← 電荷ゼロ（線形解）にしたいとき False
if isENABLE_PB:
    a += w_axi * pb_weight * reaction * vh * dx

# --- ロビン（遠方）：∂_n u + λ u = 0 を bc=10 に課す ---
LAMBDA_FF = 1.0 / max(abs(X_MAX), abs(Y_MAX))  # [1/nm]
a += w_axi * epsr * LAMBDA_FF * uh * vh * ds(definedon=mesh.Boundaries("10"))


# ----------------------------------------
# 解ベクトルと Dirichlet の設定（既存のまま）
# ----------------------------------------
u = GridFunction(V, name="solution")


def set_dirichlet(gfu, phi_tip, phi_plate=0.0):
    gfu.Set(CoefficientFunction([phi_tip, 0.0]), BND)  # [bc=1, bc=2]


u.Set(0.0)
set_dirichlet(u, phi_T, 0.0)

# ----------------------------------------
# 計算パラメータ（既存のまま）
# ----------------------------------------
levels = [0.02, 0.04, 0.06, 0.08]  # [V]
Vs_list = np.linspace(2, 10, 1)
num_pts = 500
xs = np.linspace(X_MIN, X_MAX, num_pts)
R_dict = {lev: [] for lev in levels}

newton_kwargs = dict(
    freedofs=u.space.FreeDofs(),
    maxit=500,
    maxerr=1e-11,
    inverse="sparsecholesky",
    dampfactor=1.0,
    printing=True,
)

# inverse = "pardiso" or "sparsecholesky"

# ---- ホモトピーのスケジュール（線形→非線形） ----
HOMO_SCHEDULE = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]

# ----------------------------------------
# バイアス掃引（前解でウォームスタート）
# ----------------------------------------
for k, Vs in enumerate(Vs_list):
    phi_T = float(Vs - CPD)
    set_dirichlet(u, phi_T, 0.0)

    # 線形(α=0)から非線形(α=1)へ段階的に収束させる
    for alpha in HOMO_SCHEDULE:
        hom.Set(alpha)
        Newton(a, u, **newton_kwargs)

    # 水平ライン（y=Y_LEVEL）
    prof = []
    for xq in xs:
        try:
            val = u(mesh(xq, Y_LEVEL))
        except:
            val = np.nan
        prof.append(val)
    prof = np.array(prof)

    # 各レベルの交点 → R（右半径のみ使用）
    for lev in levels:
        dif = prof - lev
        idx = np.where(dif[1:] * dif[:-1] < 0)[0]
        if len(idx) == 0:
            R = np.nan
        else:
            right = idx[xs[idx] >= 0]  # 正の半径側
            if len(right) == 0:
                R = np.nan
            else:
                i = right[0]
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = prof[i], prof[i + 1]
                R = x0 + (lev - y0) / (y1 - y0) * (x1 - x0)
        R_dict[lev].append(R)

# ----------------------------------------
# プロット（既存のまま）
# ----------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
if os.path.exists("exp_data.csv"):
    exp_df = pd.read_csv("exp_data.csv")
    Vs_exp = exp_df.iloc[:, 0]
    R_exp = exp_df.iloc[:, 1]
    ax.plot(Vs_exp, R_exp, color="k", marker="s", label="Experiment")
for lev in levels:
    ax.plot(Vs_list, R_dict[lev], marker="o", label=f"{lev*1000:.0f} meV")
ax.set_xlabel("Bias Voltage $V_s$ [V]")
ax.set_ylabel("Ring radius $R$ [nm]")
ax.set_title(
    rf"$R(V_s)$ at R_tip={RADIUS}nm, $\theta={CONE_HALF_ANGLE}^\circ$"
    f"\nY_MIN={Y_MIN}nm, Y_LEVEL={Y_LEVEL}nm (axisymmetric, Neumann at r=0)",
    pad=10,
)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(True)
fig.tight_layout()

# 水平ラインプロファイル（y=Y_LEVEL）
num_samples = 500
xs = np.linspace(X_MIN, X_MAX, num_samples)
xs_valid = []
profile_h = []
for xq in xs:
    try:
        mip = mesh(xq, Y_LEVEL)
        val = u(mip)
    except Exception:
        continue
    xs_valid.append(xq)
    profile_h.append(val)

fig2, ax2 = plt.subplots(figsize=(6, 2))
ax2.plot(xs_valid, profile_h, color="b")
ax2.set_xlabel("x (radial) coordinate [nm]")
ax2.set_ylabel("Potential (V)")
ax2.set_title(f"Horizontal profile at y = {Y_LEVEL} nm (axisymmetric)", pad=20)
ax2.grid(True)

# 鉛直ライン（x=0）
ys = np.linspace(Y_MIN, Y_MAX, num_samples)
ys_valid = []
profile = []
for yq in ys:
    try:
        mip = mesh(0.0, yq)
        val = u(mip)
    except Exception:
        continue
    ys_valid.append(yq)
    profile.append(val)
y_apex = float(TIP_TO_SURFACE)  # 先端のy座標（nm）
phi_tip_current = float(phi_T)  # 直近Vsの境界電位（V）

# 既にほぼ同一点が入っていれば二重登録を避ける
if not any(abs(y - y_apex) < 1e-9 for y in ys_valid):
    ys_valid.append(y_apex)
    profile.append(phi_tip_current)

# y昇順に並べ替え
ord_idx = np.argsort(ys_valid)
ys_valid = [ys_valid[i] for i in ord_idx]
profile = [profile[i] for i in ord_idx]
plt.rcParams["font.size"] = 16
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.plot(ys_valid, profile, color="b")
ax3.axvline(x=0, color="gray", linestyle="--")
ax3.set_xlabel("Distance from the Surface (nm)")
ax3.set_ylabel("Potential(V)")
ax3.set_title("Along $Y$ Axis (axisymmetric)", pad=20)
ax3.grid(True)
ax3.set_xlim(-15, 10)
fig3.tight_layout()

# 可視化と VTK
Draw(u)
os.makedirs("output", exist_ok=True)

# ----------------------------------------
# 保存（設定ログに軸正則化パラメータを追記）
# ----------------------------------------
now = datetime.datetime.now()
stamp_dir = now.strftime("%Y%m%d")
stamp_file = now.strftime("%Y%m%d_%H%M")
t2 = time.time()
total_time = t2 - t1

base_dir = Path.cwd()
save_dir = base_dir / stamp_dir
save_dir.mkdir(parents=True, exist_ok=True)

# 画像
out_png3 = save_dir / f"{stamp_file}_lineprofile_r_0.png"
fig3.savefig(out_png3, dpi=300, bbox_inches="tight")
out_png2 = save_dir / f"{stamp_file}_lineprofile.png"
fig2.savefig(out_png2, dpi=300, bbox_inches="tight")
out_png = save_dir / f"{stamp_file}_ring_radius.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")

# 設定ファイル
cfg_path = save_dir / f"{stamp_file}_config.dat"
config_text = f"""# Simulation Configuration Log
# Timestamp: {stamp_file}
# Total time: {total_time} s

# Coordinate system
axisymmetric = true
axis_regularization_eps_nm = {AXIS_EPS_NM:.3f}
axis_stabilization = {USE_AXIS_STAB}
axis_band_nm = {AXIS_BAND_NM:.3f}
axis_stab_scale = {AXIS_STAB_SCALE:.3f}

# Tip shape parameters
R_tip = {RADIUS} nm
s     = {TIP_TO_SURFACE} nm
theta/2 = {CONE_HALF_ANGLE} degrees
N_ARC     = {N_ARC}

# Bias conditions
V_s = {V_s} V
CPD  = {CPD} V

# PB in SiC (material=3): charge-neutral PB (no Debye length)
T  = {T} K
VT = {VT:.6f} V

# Doping & intrinsic (inputs)
ND (donor) = {ND_cm3:.3e} cm^-3
NA (acceptor) = {NA_cm3:.3e} cm^-3
n_i (intrinsic) = {NI_cm3:.3e} cm^-3

# Reference carrier densities (from neutrality & mass-action)
n_ref = {n_ref:.6e} m^-3
p_ref = {p_ref:.6e} m^-3
n_ref - p_ref = {n_ref - p_ref:.6e} m^-3   # should equal (ND-NA) in m^-3

# Normalized weights in reaction term
n_fac = {n_fac:.6f}
p_fac = {p_fac:.6f}

# PB reaction scale (SiC only)
pb_weight_SiC = {pb_weight_SiC_nm2:.6e} 1/nm^2   # = (q^2/(eps0*kB*T))*(n_ref+p_ref)

# Grid settings
X_MIN, X_MAX = {X_MIN}, {X_MAX} nm
Y_MIN, Y_MAX = {Y_MIN}, {Y_MAX} nm
FILM_THICKNESS = {FILM_THICKNESS} nm
MESH_W = {MESH_W}

# Relative permittivity
EPS_R_1 (SiO2) = {EPS_R_1}
EPS_R_2 (SiC)  = {EPS_R_2}
"""
with open(cfg_path, "w", encoding="utf-8") as f:
    f.write(config_text)
print(f"[INFO] Config saved : {cfg_path}")

# データ保存
out_csv_R = save_dir / f"{stamp_file}_R_vs_Vs.csv"
with open(out_csv_R, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Vs"] + [f"R_{lev:.2f}eV" for lev in levels]
    writer.writerow(header)
    for i, Vs in enumerate(Vs_list):
        row = [Vs] + [R_dict[lev][i] for lev in levels]
        writer.writerow(row)
print(f"[INFO] R(Vs) data saved to {out_csv_R}")

out_csv = save_dir / f"{stamp_file}_profile.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["y (nm)", "u (V)"])
    for y, val in zip(ys_valid, profile):
        writer.writerow([y, float(val)])
print(f"[INFO] Profile saved: {out_csv}")

# mesh 全体のノード値
ngm = mesh.ngmesh


def get_xy(p):
    if hasattr(p, "X") and callable(p.X):
        return float(p.X()), float(p.Y())
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    try:
        return float(p[0]), float(p[1])
    except:
        raise RuntimeError("Unknown MeshPoint API")


coords = [get_xy(p) for p in ngm.Points()]

out_csv_all = save_dir / f"{stamp_file}_u_all_nodes.csv"
with open(out_csv_all, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x (nm)", "y (nm)", "u (V)"])
    for xq, yq in coords:
        try:
            val = u(mesh(xq, yq))
        except Exception:
            continue
        writer.writerow([xq, yq, float(val)])
print(f"[INFO] All nodes saved to {out_csv_all}")
