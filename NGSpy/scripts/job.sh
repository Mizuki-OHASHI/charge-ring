#!/bin/bash

# check current directory pwd should be **/NGSpy
if [ "$(basename "$PWD")" != "NGSpy" ]; then
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
for Vtip in $(seq -8 1 8); do
for Rtip in $(seq 30 10 60); do
    printf "%.1f %d %s/Vtip%.1f_Rtip%d\n" "$Vtip" "$Rtip" "$OUTPATH" "$Vtip" "$Rtip"
  done
done > $ARGSFILE
NARGS=$(wc -l < $ARGSFILE)
echo "Total $NARGS argument sets saved to $ARGSFILE"

# run jobs
NPROCS=7  # physical CPU cores - 1
echo "Using $NPROCS parallel processes"
cat $ARGSFILE | xargs -n 3 -P $NPROCS sh -c '
    mkdir -p "$3"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running with V_tip=$1, R_tip=$2"
    python main.py --V_tip "$1" --R_tip "$2" --out_dir "$3" >> "$3/main.job.log" 2>&1 && \
    python post.py "$3" >> "$3/post.job.log" 2>&1
' sh && \
echo "All jobs completed."
echo "Results are saved in $OUTPATH"
