import argparse
import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq


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
    r = None
    for d in os.listdir(dir_path):
        match = pattern.match(d)
        if match:
            Vtip = float(match.group("Vtip"))
            Rtip = float(match.group("Rtip"))
            Htip = float(match.group("Htip"))
            profile_path = os.path.join(dir_path, d, "line_profile_horizontal.txt")
            if os.path.isfile(profile_path):
                data = np.loadtxt(profile_path, comments="#")
                data_dirs[(Vtip, Rtip, Htip)] = (d, data[:, 1])
                if r is None:
                    r = data[:, 0]
                else:
                    assert np.allclose(r, data[:, 0]), f"r values do not match in {d}"
            else:
                print(f"Warning: 'line_profile_horizontal.txt' not found in {d}")
                if r is None:
                    raise ValueError(
                        "r values are not defined yet; cannot create placeholder"
                    )
                data_dirs[(Vtip, Rtip, Htip)] = (d, np.full_like(r, np.nan))
    return data_dirs, r


def fmt_list(values):
    if len(values) <= 5:
        return ", ".join(f"{v:.2f}" for v in values)
    else:
        head = ", ".join(f"{v:.2f}" for v in values[:2])
        tail = ", ".join(f"{v:.2f}" for v in values[-2:])
        return f"{head}, ..., {tail}"


def make_ring_radius_function(Vtip_values, Rtip_values, Htip_values, r, data):
    # RegularGridInterpolator の初期化に時間がかかるので注意
    interpolator = RegularGridInterpolator(
        (Vtip_values, Rtip_values, Htip_values, r),
        data,
        method="linear"
        if (len(Vtip_values) < 4 or len(Rtip_values) < 4 or len(Htip_values) < 4)
        else "cubic",
        bounds_error=False,
        fill_value=None,  # extrapolate
    )

    def f(Rtip_target, Vring_target, Vtip_query, Htip_query):
        if Rtip_target < Rtip_values.min() or Rtip_target > Rtip_values.max():
            print(
                f"Warning: Rtip={Rtip_target} is outside range [{Rtip_values.min()}, {Rtip_values.max()}]"
            )
        if Vtip_query < Vtip_values.min() or Vtip_query > Vtip_values.max():
            print(
                f"Warning: Vtip={Vtip_query} is outside range [{Vtip_values.min()}, {Vtip_values.max()}]"
            )
        if Htip_query < Htip_values.min() or Htip_query > Htip_values.max():
            print(
                f"Warning: Htip={Htip_query} is outside range [{Htip_values.min()}, {Htip_values.max()}]"
            )
        query_points = np.array(
            [[Vtip_query, Rtip_target, Htip_query, r_val] for r_val in r]
        )
        potential_profile = interpolator(query_points)

        f_interp = interp1d(
            r, potential_profile, kind="cubic", bounds_error=False, fill_value=np.nan
        )

        pot_min, pot_max = potential_profile.min(), potential_profile.max()
        if not (pot_min <= Vring_target <= pot_max):
            print(
                f"Warning: Vring={Vring_target} is outside potential range [{pot_min:.3f}, {pot_max:.3f}]"
            )
            return np.nan

        try:

            def objective(r_val):
                return f_interp(r_val) - Vring_target

            r_result = brentq(objective, r.min(), r.max())
            return r_result
        except ValueError:
            idx = np.argmin(np.abs(potential_profile - Vring_target))
            if idx > 0 and idx < len(r) - 1:
                r_local = r[idx - 1 : idx + 2]
                pot_local = potential_profile[idx - 1 : idx + 2]
                f_local = interp1d(pot_local, r_local, kind="linear")
                try:
                    return float(f_local(Vring_target))
                except Exception:
                    return r[idx]
            return r[idx]

    return f


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
        data_dirs, r = load_line_profiles(dir_path)
        print(f"Loaded line profiles (count: {len(data_dirs)})")
        keys = sorted(data_dirs.keys())
        Vtip_values = sorted(set(k[0] for k in keys))
        Rtip_values = sorted(set(k[1] for k in keys))
        Htip_values = sorted(set(k[2] for k in keys))
        print(f"Vtip values: {fmt_list(Vtip_values)}")
        print(f"Rtip values: {fmt_list(Rtip_values)}")
        print(f"Htip values: {fmt_list(Htip_values)}")
        print(f"r values: {fmt_list(r)}")
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
        cache_path = (
            f"_ring_radius_cache/{dir_path.replace('./', '').replace('/', '+')}.npz"
        )
        Vtip_values = np.array(Vtip_values)
        Rtip_values = np.array(Rtip_values)
        Htip_values = np.array(Htip_values)
        r = np.array(r)
        np.savez_compressed(
            cache_path,
            Vtip=Vtip_values,
            Rtip=Rtip_values,
            Htip=Htip_values,
            r=r,
            data=data,
        )
        print(f"Cached data to: {cache_path}")
    else:
        print(f"Loading from cache: {cache_path}")
        cache = np.load(cache_path)
        Vtip_values = cache["Vtip"]
        Rtip_values = cache["Rtip"]
        Htip_values = cache["Htip"]
        r = cache["r"]
        data = cache["data"]
        print(
            f"Loaded line profiles (count: {data.shape[0] * data.shape[1] * data.shape[2]})"
        )
        print(f"Vtip values: {fmt_list(Vtip_values)}")
        print(f"Rtip values: {fmt_list(Rtip_values)}")
        print(f"Htip values: {fmt_list(Htip_values)}")
        print(f"r values: {fmt_list(r)}")

    print(
        "Shapes:",
        Vtip_values.shape,
        Rtip_values.shape,
        Htip_values.shape,
        r.shape,
        data.shape,
    )

    # Example usage
    print("\n--- Example: Creating ring radius function ---")
    if len(Rtip_values) > 0:
        example_Rtip = Rtip_values[0]
        example_Vring = 0.0
        data = np.nan_to_num(data, nan=0.0)
        ring_radius_func = make_ring_radius_function(
            Vtip_values,
            Rtip_values,
            Htip_values,
            r,
            data,
            # example_Rtip,
            # example_Vring,
        )
        if len(Vtip_values) > 0 and len(Htip_values) > 0:
            example_Vtip = Vtip_values[0]
            example_Htip = Htip_values[0]
            example_result = ring_radius_func(
                example_Rtip, example_Vring, example_Vtip, example_Htip
            )
            print(
                f"Example: Rtip={example_Rtip}, Vring={example_Vring}, Vtip={example_Vtip}, Htip={example_Htip}"
            )
            print(f"  -> Ring radius = {example_result:.2f} nm")
