import os
import numpy as np
import matplotlib.pyplot as plt

figwidth_mm = 160
figwidth_in = figwidth_mm / 25.4
# 12pt の本文に合わせたフォント設定
plt.rcParams.update(
    {
        "font.size": 10,  # 目盛りや凡例の基本サイズ
        "axes.titlesize": 11.5,  # 軸のタイトル
        "axes.labelsize": 11.5,  # 軸ラベル (x, y)
        "xtick.labelsize": 10,  # x軸目盛り
        "ytick.labelsize": 10,  # y軸目盛り
        "legend.fontsize": 10,  # 凡例
        # "figure.titlesize": 14,  # fig.suptitle()
        # PDF保存時にフォントを埋め込む設定
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

os.makedirs("figures", exist_ok=True)

out_dir_1 = "NGSpy/out"  # シミュレーションに採用したメッシュサイズ
out_dir_2 = "NGSpy/out2"  # メッシュを細かくした場合
title_1 = "Reference Mesh"
title_2 = "Higher Resolution"
# output = "ngsolve_vs_fenics.pdf"
output = "meshsize.pdf"
data_vert_1 = np.loadtxt(f"{out_dir_1}/line_profile_vertical.txt", skiprows=1)
data_horiz_1 = np.loadtxt(f"{out_dir_1}/line_profile_horizontal.txt", skiprows=1)
data_vert_2 = np.loadtxt(f"{out_dir_2}/line_profile_vertical.txt", skiprows=1)
data_horiz_2 = np.loadtxt(f"{out_dir_2}/line_profile_horizontal.txt", skiprows=1)
print(
    "[INFO] Data loaded",
    data_vert_1.shape,
    data_horiz_1.shape,
    data_vert_2.shape,
    data_horiz_2.shape,
)

fig, axes = plt.subplots(1, 2, figsize=(figwidth_in, figwidth_in * 0.4))
mask_1 = data_vert_1[:, 0] <= 0
axes[0].plot(
    data_vert_1[mask_1, 0],
    data_vert_1[mask_1, 1],
    label=f"[1] {title_1}",
    color="red",
    linestyle="--",
    lw=2,
)
mask_2 = data_vert_2[:, 0] <= 0
axes[0].plot(
    data_vert_2[mask_2, 0],
    data_vert_2[mask_2, 1],
    label=f"[2] {title_2}",
    color="blue",
    linestyle="-",
    lw=1,
)
axes[0].axvline(-5.0, color="gray", linestyle=":", label="SiC/SiO$_2$ interface")
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
    lw=2,
)
axes[1].plot(
    data_horiz_2[:, 0],
    data_horiz_2[:, 1],
    label=f"[2] {title_2}",
    color="blue",
    linestyle="-",
    lw=1,
)
axes[1].set_title("Horizontal Line Profile")
axes[1].set_xlabel("r / nm")
axes[1].set_ylabel("Potential / V")
axes[1].set_xticks(np.arange(0, 501, 100))
axes[1].legend()
axes[1].grid()

plt.tight_layout(rect=(0, 0, 1, 0.95))
if output is None:
    filename = f"figures/line_profiles_comparison_{title_1.replace(' ', '_')}_{title_2.replace(' ', '_')}.png"
else:
    filename = f"figures/line_profiles_comparison_{output}"
plt.savefig(filename)
plt.close()
print(f"[INFO] Figure saved to {filename}")
