import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mpl_backend_ssh  # noqa: F401

def load_experimental_data():
    data_Vtip_sweep = np.loadtxt(
        "diamond_data_6B3645/exp_data_diamond_6B3645.csv",
        delimiter=",",
        skiprows=1,
    )
    data_Htip_sweep = np.loadtxt(
        "diamond_data_6B3645/exp_data_z_diamond_6B3645.csv",
        delimiter=",",
        skiprows=1,
    )
    return data_Vtip_sweep, data_Htip_sweep


def load_simulation_data(
    cache_path="_ring_radius_cache/outputs_diamond+20251105_180001.npz",
):
    data = np.load(cache_path)
    Vtip_values = data["Vtip"]
    Rtip_values = data["Rtip"]
    Htip_values = data["Htip"]
    z = data["z"]
    r = data["r"]
    potentials = data["potentials"]
    return Vtip_values, Rtip_values, Htip_values, z, r, potentials


def main():
    exp_Vtip_sweep, exp_Htip_sweep = load_experimental_data()
    Vtip_values, Rtip_values, Htip_values, z, r, potentials = load_simulation_data()
    print(
        f"{Vtip_values.shape=}, {Rtip_values.shape=}, {Htip_values.shape=}, {z.shape=}, {r.shape=}, {potentials.shape=}"
    )
    # Vtip_values.shape=(13,), Rtip_values.shape=(9,), Htip_values.shape=(27,), z.shape=(101,), r.shape=(201,), potentials.shape=(13, 9, 27, 101, 201)

    tip_radius = 45.0
    ring_z = -5.0

    fixed_Vtip = 2.0
    fixed_Htip = 5.5
    fixed_Vtip_idx = np.abs(Vtip_values - fixed_Vtip).argmin()
    fixed_Htip_idx = np.abs(Htip_values - fixed_Htip).argmin()

    def plot_with_params(tip_radius, ring_z):
        t_idx = np.abs(Rtip_values - tip_radius).argmin()
        z_idx = np.abs(z - ring_z).argmin()
        
        data1 = potentials[:, t_idx, fixed_Htip_idx, z_idx].T * 1e3
        data2 = potentials[fixed_Vtip_idx, t_idx, :, z_idx].T * 1e3
        global_min = min(np.nanmin(data1), np.nanmin(data2))
        global_max = max(np.nanmax(data1), np.nanmax(data2))
        common_levels = np.linspace(global_min, global_max, 20)

        axes[0].cla()
        cs1 = axes[0].contour(
            Vtip_values,
            r,
            data1,
            levels=common_levels,
            cmap="plasma",
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
            cmap="plasma",
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
    axcolor = 'lightgoldenrodyellow'
    ax_tip_radius = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_ring_z = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_tip_radius = Slider(
        ax_tip_radius, 'Tip Radius (nm)', float(Rtip_values.min()), float(Rtip_values.max()), valinit=tip_radius, valstep=(Rtip_values[1]-Rtip_values[0])
    )
    slider_ring_z = Slider(
        ax_ring_z, 'Ring Z (nm)', float(z.min()), float(z.max()), valinit=ring_z, valstep=(z[1]-z[0])
    )

    def update(val):
        plot_with_params(slider_tip_radius.val, slider_ring_z.val)

    slider_tip_radius.on_changed(update)
    slider_ring_z.on_changed(update)

    plot_with_params(tip_radius, ring_z)
    plt.show()


if __name__ == "__main__":
    main()
