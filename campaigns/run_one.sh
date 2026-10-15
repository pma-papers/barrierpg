#!/bin/sh
# One campaign job: sh campaigns/run_one.sh problem configuration seed (used by launch.sh).
HERE="$(cd "$(dirname "$0")" && pwd)"
problem="$1"; config="$2"; seed="$3"
py="${BPG_PYTHON:-$HOME/.venvs/bpg-toolbox/bin/python}"
[ "$problem" = "navigation" ] && py="${BPG_SAFETY_PYTHON:-$HOME/.venvs/constrained-rl-safety310/bin/python}"
mkdir -p "$HERE/logs"
"$py" "$HERE/run.py" "$problem" "$config" --seed "$seed" > "$HERE/logs/$problem-$config-seed$seed.log" 2>&1
tail -1 "$HERE/logs/$problem-$config-seed$seed.log"
