#!/bin/bash
#PBS -N NGSpy_job

if [ -d "$HOME/micromamba" ]; then
  # micromamba
  export PATH="$HOME/micromamba/bin:$PATH"
  eval "$(micromamba shell hook -s bash)"

  micromamba activate fem3
elif [ -d "$HOME/ngenv10" ]; then
  # pyenv + virtualenv
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
  alias python="$HOME/ngenv10/bin/python"
else
  echo "No suitable Python environment found. Please set up micromamba or pyenv with the required packages."
  exit 1
fi

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
OUTPATH=outputs/$DATANAME
mkdir -p outputs
mkdir -p $OUTPATH
echo "Output path: $OUTPATH"

# create args.txt
ARGSFILE=$OUTPATH/args.txt
for Vtip in $(seq -8 0.5 6); do
  for Rtip in $(seq 20 5 55); do
    for Htip in $(seq 3 0.5 18); do
      printf "%.1f %d %.1f %s/Vtip_%.1f_Rtip_%d_Htip_%.1f\n" "$Vtip" "$Rtip" "$Htip" "$OUTPATH" "$Vtip" "$Rtip" "$Htip"
    done
  done
done > $ARGSFILE
NARGS=$(wc -l < $ARGSFILE)
echo "Total $NARGS argument sets saved to $ARGSFILE"

# run jobs
NPROCS=8  # physical CPU cores - 1
echo "Using $NPROCS parallel processes"
cat $ARGSFILE | xargs -n 4 -P $NPROCS sh -c '
    mkdir -p "$4"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running with V_tip=$1, R_tip=$2, H_tip=$3"
    python main.py --V_tip "$1" --tip_radius "$2" --tip_height "$3" --out_dir "$4" --sigma_s 3000000000000 >> "$4/main.job.log" 2>&1 && \
    python post.py "$4" >> "$4/post.job.log" 2>&1
' sh && \
echo "All jobs completed."
echo "Results are saved in $OUTPATH"
