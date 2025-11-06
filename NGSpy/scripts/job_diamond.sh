#!/bin/bash
#PBS -N NGSpy_job

# micromamba
export PATH="$HOME/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"

micromamba activate fem3

# check current directory pwd should be **/NGSpy
if [ "$(basename "$PWD")" != "NGSpy" ]; then
  echo "current directory: $PWD"
  echo "Please run this script from the NGSpy directory."
  exit 1
fi

# check python environment
if ! command -v python &> /dev/null; then
  echo "Python could not be found. Please install Python and try again."
  exit 1
fi
if ! python -c "import numpy, matplotlib, ngsolve, netgen" &> /dev/null; then
  echo "One or more required packages are not installed in the current Python environment. Please install them and try again."
  exit 1
fi

DATANAME=$1
if [ -z "$DATANAME" ]; then
  DATANAME=$(date +%Y%m%d_%H%M%S)
fi
echo "Data name: $DATANAME"
OUTPATH=outputs_diamond/$DATANAME
mkdir -p outputs_diamond
mkdir -p $OUTPATH
echo "Output path: $OUTPATH"

# create args.txt
ARGSFILE=$OUTPATH/args.txt
for Vtip in $(seq 0 0.5 6); do
  for Rtip in $(seq 25 5 65); do
    for Htip in $(seq 3 0.5 16); do
      printf "%.1f %d %.1f %s/Vtip_%.1f_Rtip_%d_Htip_%.1f\n" "$Vtip" "$Rtip" "$Htip" "$OUTPATH" "$Vtip" "$Rtip" "$Htip"
    done
  done
done > $ARGSFILE
NARGS=$(wc -l < $ARGSFILE)
echo "Total $NARGS argument sets saved to $ARGSFILE"

# run jobs
NPROCS=7  # physical CPU cores - 1
echo "Using $NPROCS parallel processes"
cat $ARGSFILE | xargs -n 4 -P $NPROCS sh -c '
    mkdir -p "$4"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running with V_tip=$1, R_tip=$2, H_tip=$3"
  python main_diamond.py --sigma_s=-1e13 --V_tip "$1" --tip_radius "$2" --tip_height "$3" --out_dir "$4" >> "$4/main.job.log" 2>&1 && \
  python post_diamond.py "$4" >> "$4/post.job.log" 2>&1
' sh && \
echo "All jobs completed."
echo "Results are saved in $OUTPATH"
