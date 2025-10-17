import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

out_dir_1 = "NGSpy/outputs/20251017_133658/Vtip-4.0_Rtip50"
out_dir_2 = "NGSpy/outputs/20251017_132602/Vtip-4.0_Rtip50"
title_1 = "NGSpy Feenstra consider donor ionization rates"
title_2 = "NGSpy Feenstra assume full ionization"
data_vert_1 = np.loadtxt(
    f"{out_dir_1}/line_profile_vertical.txt", skiprows=1
)
data_horiz_1 = np.loadtxt(
    f"{out_dir_1}/line_profile_horizontal.txt", skiprows=1
)
ngspy_data_vert_2 = np.loadtxt(
    f"{out_dir_2}/line_profile_vertical.txt", skiprows=1
)
data_horiz_2 = np.loadtxt(
   f"{out_dir_2}/line_profile_horizontal.txt", skiprows=1
)
print(
    "[INFO] Data loaded",
    data_vert_1.shape,
    data_horiz_1.shape,
    ngspy_data_vert_2.shape,
    data_horiz_2.shape,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(
    data_vert_1[:, 0],
    data_vert_1[:, 1],
    label=f"[1] {title_1}",
    color="red",
    linestyle="--",
)
axes[0].plot(
    ngspy_data_vert_2[:, 0],
    ngspy_data_vert_2[:, 1],
    label=f"[2] {title_2}",
    color="blue",
    linestyle="-",
)
axes[0].set_title("Vertical Line Profile")
axes[0].set_xlabel("z / nm")
axes[0].set_ylabel("Potential / V")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    data_horiz_1[:, 0],
    data_horiz_1[:, 1],
    label=f"[1] {title_1}",
    color="red",
    linestyle="--",
)
axes[1].plot(
    data_horiz_2[:, 0],
    data_horiz_2[:, 1],
    label=f"[2] {title_2}",
    color="blue",
    linestyle="-",
)
axes[1].set_title("Horizontal Line Profile")
axes[1].set_xlabel("r / nm")
axes[1].set_ylabel("Potential / V")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
filename = f"figures/line_profiles_comparison_{title_1.replace(' ', '_')}_{title_2.replace(' ', '_')}.png"
plt.savefig(filename, dpi=300)
plt.close()
print(f"[INFO] Figure saved to {filename}")
