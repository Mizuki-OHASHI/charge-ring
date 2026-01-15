import os
import multiprocessing as mp
from datetime import datetime, timedelta
from time import time
from main_fast import main

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
tip_radius_list = [2.0 * i + 24.0 for i in range(14)]  # 24.0 to 50.0 nm
tip_height_list = [0.5 * i + 3.0 for i in range(31)]  # 3.0 to 18.0 nm


args_list = []
for V_tip in V_tip_list:
    for tip_radius in tip_radius_list:
        for tip_height in tip_height_list:
            args = NGSPY_JSON_ARGS.copy()
            args["V_tip"] = V_tip
            args["tip_radius"] = tip_radius
            args["tip_height"] = tip_height
            out_dir_name = f"Vtip{V_tip}_R{tip_radius}_H{tip_height}"
            args["out_dir"] = os.path.join(base_dir, out_dir_name)
            args_list.append(args)


def run_single_job(args):
    """単一ジョブを実行し、成功/失敗を返す"""
    try:
        main(args)
        return True, args["out_dir"]
    except Exception as e:
        return False, f"{args['out_dir']}: {str(e)}"


def run_parallel_jobs(args_list, num_processes=None):
    """
    並列でジョブを実行し、進捗を監視する

    Parameters:
    -----------
    args_list : list
        実行する引数のリスト
    num_processes : int, optional
        使用するプロセス数. Noneの場合は, 利用可能なCPUコア数-1を使用
    """
    if num_processes is None:
        num_processes = mp.cpu_count() - 1  # 1コアを残す

    total_jobs = len(args_list)

    print("=" * 80)
    print(f"並列ジョブコントローラー")
    print(f"総ジョブ数: {total_jobs}")
    print(f"使用するCPUコア数: {num_processes} / {mp.cpu_count()}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time()
    completed = 0
    failed = 0
    failed_jobs = []

    # プロセスプールを使用して並列実行
    with mp.Pool(processes=num_processes) as pool:
        # imap_unorderedで結果を順不同で取得（効率的）
        for success, info in pool.imap_unordered(run_single_job, args_list):
            completed += 1
            if not success:
                failed += 1
                failed_jobs.append(info)

            # 進捗情報の計算
            elapsed_time = time() - start_time
            avg_time_per_job = elapsed_time / completed
            remaining_jobs = total_jobs - completed
            eta_seconds = avg_time_per_job * remaining_jobs
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)

            # 進捗表示
            progress_pct = (completed / total_jobs) * 100
            print(
                f"\r進捗: {completed}/{total_jobs} ({progress_pct:.1f}%) | "
                f"成功: {completed - failed} | 失敗: {failed} | "
                f"経過時間: {timedelta(seconds=int(elapsed_time))} | "
                f"予想終了時刻: {eta_time.strftime('%H:%M:%S')}",
                end="\n" if completed % 10 == 0 else "\r",
                flush=True,
            )

    # 最終結果の表示
    print("\n" + "=" * 80)
    print(f"全ジョブ完了!")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総実行時間: {timedelta(seconds=int(time() - start_time))}")
    print(f"成功: {total_jobs - failed} / 失敗: {failed}")

    if failed_jobs:
        print("\n失敗したジョブ:")
        for job in failed_jobs:
            print(f"  - {job}")

    print("=" * 80)


if __name__ == "__main__":
    run_parallel_jobs(args_list)
