import json
import os
import sys

import adios4dolfinx
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import ufl
from dolfinx import fem, geometry
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI

assert MPI.COMM_WORLD.size == 1, "This script should be run with a single process."


def process_mesh(out_dir: str, **kwargs):
    with XDMFFile(MPI.COMM_WORLD, os.path.join(out_dir, "solution.xdmf"), "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    with open(os.path.join(out_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    L_c = params.get("L_c", 1e-9)  # [m]
    V_tip = params["simulation"]["V_tip"]  # [V]

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V)
    adios4dolfinx.read_function(
        os.path.join(out_dir, "potential.bp"), u, name="potential"
    )
    print("[io] Potential function loaded.")

    Q = fem.functionspace(msh, ("DG", 0))
    epsilon_r = fem.Function(Q)
    adios4dolfinx.read_function(
        os.path.join(out_dir, "epsilon_r.bp"), epsilon_r, name="epsilon_r"
    )
    print("[io] Relative permittivity function loaded.")

    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    energy_density = (
        0.5 * const.epsilon_0 * epsilon_r * ufl.inner(ufl.grad(u), ufl.grad(u))
    )
    E_form = fem.form(2.0 * np.pi * energy_density * r * L_c * ufl.dx)
    total_energy = fem.assemble_scalar(E_form)
    print(f"[output] Total electrostatic energy: {total_energy} J")

    verts = msh.geometry.x
    r_coords = verts[:, 0] * L_c * 1e9  # r座標 [nm]
    z_coords = verts[:, 1] * L_c * 1e9  # z座標 [nm]

    # 各頂点上のポテンシャル: V が次数 1 の Lagrange なので、u.x.array がそのまま頂点値に対応
    u_vals_on_verts = u.x.array

    # メッシュの三角形接続情報: (2, 0) は 2次元セル(三角形)と0次元セル(頂点)の接続を意味する
    triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)

    # 描画範囲の設定
    r_max = np.max(r_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    # r_max = 500  # [nm]
    # z_min = -200  # [nm]
    # z_max = 200  # [nm]

    # カラーマップの範囲設定
    if V_tip < 0:
        vmin = V_tip
        vmax = np.max(u_vals_on_verts)
        cmap = "jet_r"
    else:
        vmin = np.min(u_vals_on_verts)
        vmax = V_tip
        cmap = "jet"
    # カラーバーのレベルを調整
    levels = np.linspace(vmin, vmax, 201)

    plt.figure()
    plt.tricontourf(
        r_coords,
        z_coords,
        u_vals_on_verts,
        triangles=triangles,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.triplot(
        r_coords,
        z_coords,
        triangles=triangles,
        color="k",
        linewidth=0.2,
        alpha=0.5,
    )
    plt.colorbar(label="Potential / V")
    # 界面の線を描画
    sio2_z_nm = params["geometric"]["l_sio2"] * 1e9
    plt.axhline(y=0, color="w", linestyle="-", linewidth=0.25, alpha=0.5)
    plt.axhline(y=-sio2_z_nm, color="w", linestyle="-", linewidth=0.25, alpha=0.5)

    plt.xlabel("$r$ / nm")
    plt.ylabel("$z$ / nm")
    plt.title(f"Tip-SiC Potential ($V_\\text{{tip}}={V_tip:.1f}\\ \\text{{V}}$)")
    plt.axis("equal")
    plt.xlim(0, r_max)
    plt.ylim(z_min, z_max)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    save_path = os.path.join(out_dir, "potential_map.png")
    plt.savefig(save_path, dpi=300)
    if kwargs.get("show_map", False):
        plt.show()
    plt.close()
    print(f"[figure] Potential plot saved to {save_path}")

    # ラインプロファイル
    bb = geometry.bb_tree(msh, msh.topology.dim)

    # SiC/SiO2界面でのポテンシャルプロファイル
    r_vals_on_interface = np.arange(0, r_max + 1, 1.0)  # [nm]
    z_interface = -5.0  # [nm]
    points_on_interface = np.vstack(
        [
            r_vals_on_interface,
            np.full_like(r_vals_on_interface, z_interface),
            np.zeros_like(r_vals_on_interface),
        ]
    ).T
    candidates_on_interface = geometry.compute_collisions_points(
        bb, points_on_interface
    )
    colliding_cells_on_interface = geometry.compute_colliding_cells(
        msh, candidates_on_interface, points_on_interface
    )
    cells_on_interface = np.full_like(r_vals_on_interface, -1, dtype=np.int32)
    for i in range(len(r_vals_on_interface)):
        links = colliding_cells_on_interface.links(i)
        if len(links) > 0:
            cells_on_interface[i] = links[0]
    mask_on_interface = cells_on_interface != -1
    u_vals_on_interface = np.full_like(r_vals_on_interface, np.nan, dtype=np.float64)
    u_vals_on_interface[mask_on_interface] = u.eval(
        points_on_interface[mask_on_interface], cells_on_interface[mask_on_interface]
    ).reshape(-1)

    # 回転軸上のポテンシャル/電荷密度プロファイル
    z_vals_on_axis = np.arange(z_min, -params["geometric"]["l_sio2"] * 1e9, 1.0)  # [nm]
    r_axis = 0.0  # [nm]
    points_on_axis = np.vstack(
        [
            np.full_like(z_vals_on_axis, r_axis),
            z_vals_on_axis,
            np.zeros_like(z_vals_on_axis),
        ]
    ).T
    candidates_on_axis = geometry.compute_collisions_points(bb, points_on_axis)
    colliding_cells_on_axis = geometry.compute_colliding_cells(
        msh, candidates_on_axis, points_on_axis
    )
    cells_on_axis = np.full_like(z_vals_on_axis, -1, dtype=np.int32)
    for i in range(len(z_vals_on_axis)):
        links = colliding_cells_on_axis.links(i)
        if len(links) > 0:
            cells_on_axis[i] = links[0]
    mask_on_axis = cells_on_axis != -1
    print(
        f"[debug] {np.sum(mask_on_axis)} points found on axis out of {len(z_vals_on_axis)}"
    )
    u_vals_on_axis = np.full_like(z_vals_on_axis, np.nan, dtype=np.float64)
    u_vals_on_axis[mask_on_axis] = u.eval(
        points_on_axis[mask_on_axis], cells_on_axis[mask_on_axis]
    ).reshape(-1)

    # charfe density on axis
    # u_vals_on_axis の2階微分を計算
    rho_vals_on_axis = np.full_like(z_vals_on_axis, np.nan, dtype=np.float64)
    epsilon_r_axis = np.full_like(z_vals_on_axis, np.nan, dtype=np.float64)
    epsilon_r_axis[mask_on_axis] = epsilon_r.eval(
        points_on_axis[mask_on_axis], cells_on_axis[mask_on_axis]
    ).reshape(-1)
    rho_vals_on_axis[mask_on_axis] = (
        -const.epsilon_0
        * epsilon_r_axis[mask_on_axis]
        * np.gradient(
            np.gradient(u_vals_on_axis[mask_on_axis], z_vals_on_axis[mask_on_axis]),
            z_vals_on_axis[mask_on_axis],
        )
        * 1e18  # [C/m^3] -> [C/nm^3]
    )
    # スムージング (ベクトル計算)
    window_size = 5
    rho_vals_on_axis_smoothed = np.convolve(
        rho_vals_on_axis, np.ones(window_size) / window_size, mode="same"
    )
    rho_vals_on_axis = rho_vals_on_axis_smoothed

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    axes[0].plot(r_vals_on_interface, u_vals_on_interface, "b-", markersize=2)
    axes[0].set_xlabel("$r$ / nm")
    axes[0].set_ylabel("Potential / V")
    axes[0].set_title(
        f"Potential on SiC/SiO2 Interface ($z={z_interface:.1f}\\ \\text{{nm}}$)"
    )
    axes[0].grid()

    axes[1].plot(z_vals_on_axis, u_vals_on_axis, "r-", markersize=2)
    axes[1].set_xlabel("$z$ / nm")
    axes[1].set_ylabel("Potential / V")
    axes[1].set_title("Potential on Rotation Axis ($r=0$ nm)")
    axes[1].grid()

    axes[2].plot(z_vals_on_axis, rho_vals_on_axis, "m-", markersize=2)
    axes[2].set_xlabel("$z$ / nm")
    axes[2].set_ylabel("Charge Density / $\\mu$C/nm$^3$")
    axes[2].set_title("Charge Density on Rotation Axis ($r=0$ nm)")
    axes[2].grid()

    fig.suptitle(
        f"Line Profiles ($V_\\text{{tip}}={V_tip:.1f}\\ \\text{{V}}, R_\\text{{tip}}={params['geometry']['tip_radius'] * 1e9:.1f}\\ \\text{{nm}}$)"
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = os.path.join(out_dir, "potential_line_profiles.png")
    plt.savefig(save_path, dpi=300)
    if kwargs.get("show_line_profiles", False):
        plt.show()
    plt.close()


def argv_to_dict(argv):
    arg_dict = {}
    for arg in argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                value = str(value)
            arg_dict[key] = value
    return arg_dict


if __name__ == "__main__":
    args = argv_to_dict(sys.argv)
    out_dir = args.get("out_dir", "out")
    process_mesh(out_dir)
