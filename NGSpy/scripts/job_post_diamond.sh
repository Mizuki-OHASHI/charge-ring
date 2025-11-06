#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <base_directory>" >&2
	exit 1
}

if [[ $# -ne 1 ]]; then
	usage
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(realpath "${script_dir}/..")
python_cmd=${PYTHON:-python}

if ! command -v "${python_cmd}" >/dev/null 2>&1; then
	echo "Error: ${python_cmd} is not available in PATH" >&2
	exit 1
fi

base_dir=$(realpath "$1")

if [[ ! -d "${base_dir}" ]]; then
	echo "Error: ${base_dir} is not a directory" >&2
	exit 1
fi

mapfile -d '' -t subdirs < <(find "${base_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
total=${#subdirs[@]}

if [[ ${total} -eq 0 ]]; then
	echo "No subdirectories found under ${base_dir}."
	exit 0
fi

echo "Found ${total} directories under ${base_dir}."

default_procs=$(nproc 2>/dev/null || echo 1)
parallel_procs=${JOB_POST_PROCS:-${default_procs}}

if [[ ${parallel_procs} -lt 1 ]]; then
	parallel_procs=1
fi

if [[ ${parallel_procs} -gt ${total} ]]; then
	parallel_procs=${total}
fi

echo "Using ${parallel_procs} parallel process(es)."

export BASE_DIR="${base_dir}" PROJECT_ROOT="${project_root}" PYTHON_CMD="${python_cmd}"

printf '%s\0' "${subdirs[@]}" | xargs -0 -r -n1 -P "${parallel_procs}" bash -c '
	dir="$1"
	rel_path=$(realpath --relative-to "${BASE_DIR}" "${dir}")
	log_file="${dir}/post.log"

	echo "[post_diamond] Processing ${rel_path} ..."
	if ! (cd "${PROJECT_ROOT}" && "${PYTHON_CMD}" post_diamond.py "${dir}" >"${log_file}" 2>&1); then
		echo "Error while processing ${rel_path}. See ${log_file} for details." >&2
		exit 1
	fi
	echo "[post_diamond] Finished ${rel_path}"
' _

echo "All directories processed."
