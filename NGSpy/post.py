import argparse
import json
import os

from main import GeometricParameters, load_results

_ = load_results


def main():
    parser = argparse.ArgumentParser(description="Post-process NGSpy results")
    parser.add_argument("out_dir", type=str, help="Output directory")
    args, _ = parser.parse_known_args()
    out_dir = args.out_dir

    # load params
    with open(os.path.join(out_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    with open(os.path.join(out_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    geo_params = params["geometric"]
    V_c = metadata["V_c"]
    geom = GeometricParameters(**geo_params)
    msh2, u_loaded, uV = load_results(out_dir, geom, V_c)
