#!/bin/sh
# Run a list of campaign jobs in parallel. Each line of the job file is "problem configuration seed".
#   sh campaigns/launch.sh campaigns/jobs-lander.txt 8
# The navigation problem uses the Python environment with Safety-Gymnasium (BPG_SAFETY_PYTHON); the others use
# BPG_PYTHON. Logs go to campaigns/logs/.
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$1"; PAR="${2:-4}"
grep -v '^ *#' "$JOBS" | grep -v '^ *$' | xargs -L 1 -P "$PAR" sh "$HERE/run_one.sh"
