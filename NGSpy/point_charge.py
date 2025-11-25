import os
import re
import numpy as np


def load_electrostatic_energy(dirpath):
    dls = os.listdir(dirpath)
    Vtips = None
    tip_radii = set()
    tip_heights = set()
    energies_dict = {}
    pattern = re.compile(r"Rtip_(\d+)_Htip_([\d.]+)")
    for dl in dls:
        if not os.path.isdir(os.path.join(dirpath, dl)):
            continue
        match = pattern.match(dl)
        if not match:
            continue
        Rtip = int(match.group(1))
        Htip = float(match.group(2))
        energy_file = os.path.join(dirpath, dl, "electrostatic_energy.txt")
        dat = np.loadtxt(energy_file, comments="#")
        # sort by V_tip
        sort_idx = np.argsort(dat[:, 0])
        dat = dat[sort_idx, :]
        if Vtips is None:
            Vtips = dat[:, 0]
        else:
            assert np.allclose(Vtips, dat[:, 0]), "V_tip values do not match!"
        data = dat[:, 1]
        tip_radii.add(Rtip)
        tip_heights.add(Htip)
        energies_dict[(Rtip, Htip)] = data
    tip_radii = np.array(list(tip_radii))
    tip_heights = np.array(list(tip_heights))
    tip_radii.sort()
    tip_heights.sort()
    energies = np.zeros((len(tip_radii), len(tip_heights), len(Vtips)))
    for i, Rtip in enumerate(tip_radii):
        for j, Htip in enumerate(tip_heights):
            energies[i, j, :] = energies_dict[(Rtip, Htip)]
    return energies, tip_radii, tip_heights, Vtips


def force_curve(energies, tip_heights):
    forces = -np.gradient(energies, tip_heights, axis=-1)
    return forces


def main():
    dir_with_pc = "outputs/20251125_170228_cpd_w"
    dir_without_pc = "outputs/20251125_171655_cpd_wo"
    data_with_pc, Rtips, Htips, Vtips = load_electrostatic_energy(dir_with_pc)
    data_without_pc, Rtips_wo, Htips_wo, Vtips_wo = load_electrostatic_energy(
        dir_without_pc
    )
    print(f"loaded data: {data_with_pc.shape=}, {data_without_pc.shape=}")
    assert np.allclose(Rtips, Rtips_wo), "Tip radii do not match!"
    assert np.allclose(Htips, Htips_wo), "Tip heights do not match!"
    assert np.allclose(Vtips, Vtips_wo), "V_tip values do not match!"

    # 静電エネルギー vs 探針ー試料間距離を計算して、
    # その結果をz微分するとフォースカーブが得られます。
    # charge trapに電子がいるときといないときの2つのフォースカーブを並べて、
    # ヒステリシスループがどれぐらいになるのかシミュレーションできませんか？
    # 計算でしか求められないので、理解が深まると思うのですが。
    # 加えて、Vtip vs 静電エネルギーから、Vtip vs キャパシタンス も描けませんか？
    # 静電エネルギー= 1/2 C(V_tip) (V_tip - V_CPD)^2

    forces_with_pc = force_curve(data_with_pc, Htips)
    forces_without_pc = force_curve(data_without_pc, Htips)
    print(f"calculated forces: {forces_with_pc.shape=}, {forces_without_pc.shape=}")


if __name__ == "__main__":
    main()
