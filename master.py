import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

fx_out_dir = "_3+"
ng_out_dir = "example_output"
fenics_data_vert = np.loadtxt(
    f"FEniCS/{fx_out_dir}/line_profile_vertical.txt", skiprows=1
)
fenics_data_horiz = np.loadtxt(
    f"FEniCS/{fx_out_dir}/line_profile_horizontal.txt", skiprows=1
)
ngspy_data_vert = np.loadtxt(
   f"NGSpy/{ng_out_dir}/line_profile_vertical.txt", skiprows=1
)
ngspy_data_horiz = np.loadtxt(
   f"NGSpy/{ng_out_dir}/line_profile_horizontal.txt", skiprows=1
)
print(
    "[INFO] Data loaded",
    fenics_data_vert.shape,
    fenics_data_horiz.shape,
    ngspy_data_vert.shape,
    ngspy_data_horiz.shape,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(
    fenics_data_vert[:, 0],
    fenics_data_vert[:, 1],
    label="FEniCS",
    color="red",
    linestyle="--",
)
axes[0].plot(
    ngspy_data_vert[:, 0],
    ngspy_data_vert[:, 1],
    label="NGSpy",
    color="blue",
    linestyle="-",
)
axes[0].set_title("Vertical Line Profile")
axes[0].set_xlabel("z / nm")
axes[0].set_ylabel("Potential / V")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    fenics_data_horiz[:, 0],
    fenics_data_horiz[:, 1],
    label="FEniCS",
    color="red",
    linestyle="--",
)
axes[1].plot(
    ngspy_data_horiz[:, 0],
    ngspy_data_horiz[:, 1],
    label="NGSpy",
    color="blue",
    linestyle="-",
)
axes[1].set_title("Horizontal Line Profile")
axes[1].set_xlabel("r / nm")
axes[1].set_ylabel("Potential / V")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig("figures/line_profiles_comparison.png", dpi=150)
plt.close()
print("[INFO] Figure saved to figures/line_profiles_comparison.png")
