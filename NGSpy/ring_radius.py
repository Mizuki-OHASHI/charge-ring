"""
```
cat Vtip_-6.0_Rtip_25_Htip_11.5/line_profile_horizontal.txt
# r_nm potential_V
0.000000000000000000e+00 -2.688709042481345901e-01
2.512562814070351980e+00 -2.673426777296630807e-01
5.025125628140703959e+00 -2.633452947209631856e-01
7.537688442211056383e+00 -2.568610662480635409e-01
1.005025125628140792e+01 -2.480075799725940389e-01
1.256281407035175945e+01 -2.370238258051312052e-01
1.507537688442211277e+01 -2.239929111089585256e-01
1.758793969849246253e+01 -2.092346122219203020e-01
...
4.949748743718593573e+02 1.561516641448404130e-01
4.974874371859297071e+02 1.561894236206977093e-01
5.000000000000000000e+02 1.562933814277288613e-01
```
"""

import argparse
import os
import re

import numpy as np


def input_dir_path(base_dir, selected):
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dirs = sorted(dirs)
    print(base_dir + "/")
    if selected is None:
        for i, d in enumerate(dirs):
            print(f"[{i}]\t{d}")
        idx = int(input("Select directory index: "))
        selected = dirs[idx]
        return input_dir_path(base_dir, selected)
    print(f"Selected: {selected}")
    q = (
        input("C to choose this, R to go back, or L to list in this directory: ")
        .strip()
        .upper()
    )
    if q == "C":
        return os.path.join(base_dir, selected)
    if q == "R":
        return input_dir_path(base_dir, None)
    if q == "L":
        return input_dir_path(os.path.join(base_dir, selected), None)


def load_line_profiles(dir_path):
    data_dirs = {}
    pattern = re.compile(
        r"Vtip_(?P<Vtip>-?\d+\.?\d*)_Rtip_(?P<Rtip>\d+\.?\d*)_Htip_(?P<Htip>\d+\.?\d*)"
    )
    for d in os.listdir(dir_path):
        match = pattern.match(d)
        if match:
            Vtip = float(match.group("Vtip"))
            Rtip = float(match.group("Rtip"))
            Htip = float(match.group("Htip"))
            profile_path = os.path.join(dir_path, d, "line_profile_horizontal.txt")
            if os.path.isfile(profile_path):
                data = np.loadtxt(profile_path, comments="#")
                data_dirs[(Vtip, Rtip, Htip)] = (d, data)
            else:
                print(f"Warning: 'line_profile_horizontal.txt' not found in {d}")
    return data_dirs


def fmt_list(values):
    if len(values) <= 5:
        return ", ".join(f"{v:.2f}" for v in values)
    else:
        head = ", ".join(f"{v:.2f}" for v in values[:2])
        tail = ", ".join(f"{v:.2f}" for v in values[-2:])
        return f"{head}, ..., {tail}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing Vtip_Rtip_Htip subdirectories",
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Path to cache file",
    )
    args = parser.parse_args()
    dir_path = args.dir
    cache_path = args.cache
    if cache_path is None:
        if dir_path is None:
            dir_path = input_dir_path(".", None)
        print(f"Using directory: {dir_path}")
        data_dirs = load_line_profiles(dir_path)
        print(f"Loaded line profiles (count: {len(data_dirs)})")
        keys = sorted(data_dirs.keys())
        Vtip_values = sorted(set(k[0] for k in keys))
        Rtip_values = sorted(set(k[1] for k in keys))
        Htip_values = sorted(set(k[2] for k in keys))
        print(f"Vtip values: {fmt_list(Vtip_values)}")
        print(f"Rtip values: {fmt_list(Rtip_values)}")
        print(f"Htip values: {fmt_list(Htip_values)}")
        data = [
            [
                [data_dirs[(Vtip, Rtip, Htip)][1] for Htip in Htip_values]
                for Rtip in Rtip_values
            ]
            for Vtip in Vtip_values
        ]
        data = np.array(data)
        if not os.path.exists("_ring_radius_cache"):
            os.makedirs("_ring_radius_cache")
        cache_path = f"_ring_radius_cache/{dir_path.replace('/', '+')}.npz"
        np.savez_compressed(
            cache_path,
            Vtip=np.array(Vtip_values),
            Rtip=np.array(Rtip_values),
            Htip=np.array(Htip_values),
            data=data,
        )
        print(f"Cached data to: {cache_path}")
    else:
        print(f"Loading from cache: {cache_path}")
        cache = np.load(cache_path)
        Vtip_values = cache["Vtip"]
        Rtip_values = cache["Rtip"]
        Htip_values = cache["Htip"]
        data = cache["data"]
        print(
            f"Loaded line profiles (count: {data.shape[0] * data.shape[1] * data.shape[2]})"
        )
        print(f"Vtip values: {fmt_list(Vtip_values)}")
        print(f"Rtip values: {fmt_list(Rtip_values)}")
        print(f"Htip values: {fmt_list(Htip_values)}")
