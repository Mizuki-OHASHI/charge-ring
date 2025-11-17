import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mpl_backend_ssh  # noqa: F401


def load_experimental_data(exp_vtip_path, exp_htip_path):
    data_Vtip_sweep = np.loadtxt(exp_vtip_path, delimiter=",", skiprows=1)
    data_Htip_sweep = np.loadtxt(exp_htip_path, delimiter=",", skiprows=1)
    return data_Vtip_sweep, data_Htip_sweep


def load_simulation_data(cache_path):
    data = np.load(cache_path)
    Vtip_values = data["Vtip"]
    Rtip_values = data["Rtip"]
    Htip_values = data["Htip"]
    r = data["r"]
    potentials = data["data"]
    return Vtip_values, Rtip_values, Htip_values, r, potentials


def main(
    sim_path,
    exp_vtip_path,
    exp_htip_path,
    *,
    fixed_Vtip,
    fixed_Htip,
    exp_zoffset,
    exp_vtip_offset,
):
    exp_Vtip_sweep, exp_Htip_sweep = load_experimental_data(
        exp_vtip_path, exp_htip_path
    )
    exp_Htip_sweep[:, 0] += exp_zoffset
    # exp_Vtip_sweep[:, 0] += exp_vtip_offset
    # fixed_Vtip += exp_vtip_offset
    Vtip_values, Rtip_values, Htip_values, r, potentials = load_simulation_data(
        sim_path
    )
    print(
        f"{Vtip_values.shape=}, {Rtip_values.shape=}, {Htip_values.shape=}, {r.shape=}, {potentials.shape=}"
    )
    # Vtip_values.shape=(13,), Rtip_values.shape=(9,), Htip_values.shape=(27,), z.shape=(101,), r.shape=(201,), potentials.shape=(13, 9, 27, 201)
    Vtip_values += exp_vtip_offset

    # 実験データと合わせて範囲を設定する
    Vtip_min = exp_Vtip_sweep[:, 0].min()
    Vtip_max = exp_Vtip_sweep[:, 0].max()
    Vtip_range = Vtip_max - Vtip_min
    Htip_min = exp_Htip_sweep[:, 0].min()
    Htip_max = exp_Htip_sweep[:, 0].max()
    Htip_range = Htip_max - Htip_min
    r_max = max(exp_Vtip_sweep[:, 1].max(), exp_Htip_sweep[:, 1].max())
    r_mask = r <= r_max * 1.1
    Vtip_mask = (Vtip_values >= Vtip_min - 0.1 * Vtip_range) & (
        Vtip_values <= Vtip_max + 0.1 * Vtip_range
    )
    Htip_mask = (Htip_values >= Htip_min - 0.1 * Htip_range) & (
        Htip_values <= Htip_max + 0.1 * Htip_range
    )
    r = r[r_mask]
    Vtip_values = Vtip_values[Vtip_mask]
    Htip_values = Htip_values[Htip_mask]
    potentials = potentials[Vtip_mask][:, :, Htip_mask][:, :, :, r_mask]

    tip_radius = 45.0

    fixed_Vtip_idx = np.abs(Vtip_values - fixed_Vtip).argmin()
    fixed_Htip_idx = np.abs(Htip_values - fixed_Htip).argmin()

    def plot_with_params(tip_radius):
        t_idx = np.abs(Rtip_values - tip_radius).argmin()

        data1 = potentials[:, t_idx, fixed_Htip_idx].T * 1e3
        data2 = potentials[fixed_Vtip_idx, t_idx, :].T * 1e3
        global_min = min(np.nanmin(data1), np.nanmin(data2))
        global_max = max(np.nanmax(data1), np.nanmax(data2))
        common_levels = np.linspace(global_min, global_max, 20)

        axes[0].cla()
        cs1 = axes[0].contour(
            Vtip_values,
            r,
            data1,
            levels=common_levels,
            cmap="plasma_r",
            linewidths=1.0,
        )
        axes[0].clabel(cs1, inline=True, fontsize=8, fmt="%.0fmV")
        axes[0].plot(
            exp_Vtip_sweep[:, 0], exp_Vtip_sweep[:, 1], "b.", label="Experimental Data"
        )
        axes[0].set_title(f"Vtip sweep at Htip={fixed_Htip}nm")
        axes[0].set_xlabel("Vtip / V")
        axes[0].set_ylabel("Ring radius / nm")

        axes[1].cla()
        cs2 = axes[1].contour(
            Htip_values,
            r,
            data2,
            levels=common_levels,
            cmap="plasma_r",
            linewidths=1.0,
        )
        axes[1].clabel(cs2, inline=True, fontsize=8, fmt="%.0fmV")
        axes[1].plot(
            exp_Htip_sweep[:, 0], exp_Htip_sweep[:, 1], "b.", label="Experimental Data"
        )
        axes[1].set_title(f"Htip sweep at Vtip={fixed_Vtip}V")
        axes[1].set_xlabel("Htip / nm")
        axes[1].set_ylabel("Ring radius / nm")
        fig.canvas.draw_idle()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    # スライダーの位置と範囲を設定
    axcolor = "lightgoldenrodyellow"
    ax_tip_radius = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

    slider_tip_radius = Slider(
        ax_tip_radius,
        "Tip Radius (nm)",
        float(Rtip_values.min()),
        float(Rtip_values.max()),
        valinit=tip_radius,
        valstep=(Rtip_values[1] - Rtip_values[0]),
    )

    def update(val):
        plot_with_params(slider_tip_radius.val)

    slider_tip_radius.on_changed(update)

    plot_with_params(tip_radius)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to cached simulation data file",
    )
    parser.add_argument(
        "--exp_vtip",
        type=str,
        required=True,
        help="Path to experimental Vtip sweep data file",
    )
    parser.add_argument(
        "--exp_htip",
        type=str,
        required=True,
        help="Path to experimental Htip sweep data file",
    )
    parser.add_argument(
        "--fixed_vtip",
        type=float,
        default=2.0,
        help="Fixed Vtip (V) value for Htip sweep plot",
    )
    parser.add_argument(
        "--fixed_htip",
        type=float,
        default=8,
        help="Fixed Htip (nm) value for Vtip sweep plot",
    )
    parser.add_argument(
        "--exp_zoffset",
        type=float,
        default=0.0,
        help="Experimental Z offset (nm)",
    )
    parser.add_argument(
        "--exp_vtip_offset",
        "--cpd",
        type=float,
        default=0.0,
        help="Experimental Vtip offset (V)",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")
    sim_path = args.cache

    print(dict(**vars(args)))
    main(
        sim_path,
        args.exp_vtip,
        args.exp_htip,
        fixed_Vtip=args.fixed_vtip,
        fixed_Htip=args.fixed_htip,
        exp_zoffset=args.exp_zoffset,
        exp_vtip_offset=args.exp_vtip_offset,
    )
