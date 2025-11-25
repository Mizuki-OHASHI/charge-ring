#!/bin/bash
#PBS -N cpd

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
OUTPATH=outputs/${DATANAME}_cpd_wo
mkdir -p outputs
mkdir -p $OUTPATH
echo "Output path: $OUTPATH"

# create args.txt
ARGSFILE=$OUTPATH/args.txt
# Split V_tip range into negative and positive to avoid crossing zero
for Rtip in 40; do
  for Htip in $(seq 3 0.5 28); do
    # Negative V_tip range: -8 to -0.5 (step 0.5)
    printf "%d %.1f %s/Rtip_%d_Htip_%.1f\n" "$Rtip" "$Htip" "$OUTPATH" "$Rtip" "$Htip"
  done
done > $ARGSFILE
NARGS=$(wc -l < $ARGSFILE)
echo "Total $NARGS argument sets saved to $ARGSFILE"

# run jobs
NPROCS=8
echo "Using $NPROCS parallel processes"
cat $ARGSFILE | xargs -n 3 -P $NPROCS sh -c '
    mkdir -p "$3"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running with R_tip=$1, H_tip=$2"
    python main_fast.py --V_tip="-4:-0.2:0.2" --tip_radius "$1" --tip_height "$2" --out_dir "$3" --sigma_s 1E11 >> "$3/main.job.log" 2>&1 && \
    python main_fast.py --V_tip="0:4:0.2" --tip_radius "$1" --tip_height "$2" --out_dir "$3" --sigma_s 1E11 >> "$3/main.job.log" 2>&1 && \
    python post_fast.py "$3" --calc_energy >> "$3/post.job.log" 2>&1
' sh && \
echo "All jobs completed."
echo "Results are saved in $OUTPATH"
