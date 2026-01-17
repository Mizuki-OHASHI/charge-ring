import json
import os
import multiprocessing as mp
from datetime import datetime, timedelta
from time import time
from main_fast import main

NUM_PROCESSES = 2

base_dir = "outputs/main_sweep"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

NGSPY_JSON_ARGS = {
    "V_tip": "0:8:0.2",
    "tip_radius": 45.0,
    "tip_height": 8.0,
    "l_sio2": 5.0,
    "Nd": 1e16,
    "sigma_s": 1e12,
    "T": 300.0,
    "out_dir": "out",
    "plot_fermi": False,
    "model": "Feenstra",
    "assume_full_ionization": False,
    "mesh_scale": 1.0,
    "maxerr": 1e-11,
    "point_charge": False,
    "silent": True,
}

V_tip_list = ["0:8:0.2", "-8:0:0.5"]
tip_radius_list = [2.0 * i + 24.0 for i in range(19)]  # 24.0 to 60.0 n
tip_height_list = [0.5 * i + 3.0 for i in range(31)]  # 3.0 to 18.0 nm


args_list = []
for tip_radius in tip_radius_list:
    for tip_height in tip_height_list:
        for V_tip in V_tip_list:
            args = NGSPY_JSON_ARGS.copy()
            args["V_tip"] = V_tip
            args["tip_radius"] = tip_radius
            args["tip_height"] = tip_height
            out_dir_name = f"Vtip{V_tip}_R{tip_radius}_H{tip_height}"
            args["out_dir"] = os.path.join(base_dir, out_dir_name)
            args_list.append(args)


def is_job_completed(out_dir: str) -> bool:
    """
    Check if a job has been completed.

    Completion criteria:
    1. out_dir exists
    2. parameters.json exists
    3. All V_tip directories (V_tip_+X.XXV format) listed in
       simulation.V_tips of parameters.json exist
    """
    if not os.path.isdir(out_dir):
        return False

    params_file = os.path.join(out_dir, "parameters.json")
    if not os.path.isfile(params_file):
        return False

    try:
        with open(params_file, "r") as f:
            params = json.load(f)

        v_tips = params.get("simulation", {}).get("V_tips", [])
        if not v_tips:
            return False

        for v_tip in v_tips:
            v_tip_dir = os.path.join(out_dir, f"V_tip_{v_tip:+.2f}V")
            if not os.path.isdir(v_tip_dir):
                return False

        return True

    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def filter_incomplete_jobs(args_list: list[dict]) -> tuple[list[dict], int, int]:
    """
    Filter out completed jobs and return only incomplete ones.

    Returns:
        (list of incomplete jobs, completed count, total count)
    """
    incomplete = []
    completed_count = 0

    for args in args_list:
        out_dir = args["out_dir"]
        if is_job_completed(out_dir):
            completed_count += 1
        else:
            incomplete.append(args)

    return incomplete, completed_count, len(args_list)


def run_single_job(args):
    """Execute a single job and return success/failure status."""
    try:
        main(args)
        return True, args["out_dir"]
    except Exception as e:
        return False, f"{args['out_dir']}: {str(e)}"


def run_parallel_jobs(args_list, num_processes=NUM_PROCESSES, resume=True):
    """
    Execute jobs in parallel with progress monitoring.

    Parameters:
    -----------
    args_list : list
        List of argument dictionaries for each job
    num_processes : int, optional
        Number of processes to use. Defaults to cpu_count // 2
    resume : bool, optional
        If True, skip completed jobs and resume from where it left off
    """
    assert num_processes > 0, "Number of processes must be greater than 0"
    assert num_processes <= mp.cpu_count(), "Number of processes exceeds CPU count"

    total_original = len(args_list)

    if resume:
        args_list, already_completed, _ = filter_incomplete_jobs(args_list)
        if already_completed > 0:
            print(f"[Resume] Skipping {already_completed}/{total_original} completed jobs")

    total_jobs = len(args_list)

    if total_jobs == 0:
        print("=" * 80)
        print("All jobs already completed.")
        print("=" * 80)
        return

    print("=" * 80)
    print("Parallel Job Controller")
    print(f"Jobs to run: {total_jobs}" + (f" (of {total_original} total)" if resume else ""))
    print(f"CPU cores: {num_processes} / {mp.cpu_count()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time()
    completed = 0
    failed = 0
    failed_jobs = []

    with mp.Pool(processes=num_processes) as pool:
        for success, info in pool.imap_unordered(run_single_job, args_list):
            completed += 1
            if not success:
                failed += 1
                failed_jobs.append(info)

            elapsed_time = time() - start_time
            avg_time_per_job = elapsed_time / completed
            remaining_jobs = total_jobs - completed
            eta_seconds = avg_time_per_job * remaining_jobs
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)

            progress_pct = (completed / total_jobs) * 100
            print(
                f"\rProgress: {completed}/{total_jobs} ({progress_pct:.1f}%) | "
                f"OK: {completed - failed} | Failed: {failed} | "
                f"Time: {datetime.now().strftime('%H:%M:%S')} | "
                f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
                f"ETA: {eta_time.strftime('%H:%M:%S')}",
                end="\n" if completed % 10 == 0 else "\r",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("All jobs finished!")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {timedelta(seconds=int(time() - start_time))}")
    print(f"OK: {total_jobs - failed} / Failed: {failed}")

    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            print(f"  - {job}")

    print("=" * 80)


if __name__ == "__main__":
    run_parallel_jobs(args_list)
