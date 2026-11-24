#!/usr/bin/env bash
# One experiment end to end: tune (if the config has a "tune" section), sweep, M4 measurement, plots.
#   bash scripts/run_experiment.sh stage2_pendulum_table            # all seeds in the config
#   bash scripts/run_experiment.sh pendulum_main --seeds 0 1 2 3 4    # fewer seeds (say so when reporting)
#   JOBS=6 bash scripts/run_experiment.sh pendulum_main
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
E="$1"; shift
JOBS="${JOBS:-$(($(nproc) > 1 ? $(nproc) - 0 : 1))}"
if python3 -c "import json,sys; sys.exit(0 if 'tune' in json.load(open('configs/$E.json')) else 1)"; then
  if [ -f "configs/${E}_tuned.json" ]; then          # resuming: delete it to tune again
    echo "using the existing configs/${E}_tuned.json"
  else
    python3 scripts/tune.py "$E" --jobs "$JOBS"
  fi
  E="${E}_tuned"
  if [ -f "configs/${E}_select.json" ]; then        # D70/D73: selection runs on the tuning seeds
    python3 scripts/run_sweep.py "${E}_select" --jobs "$JOBS"
  fi
fi
python3 scripts/run_sweep.py "$E" --jobs "$JOBS" "$@"
python3 scripts/measure_m4.py "$E"
python3 scripts/make_plots.py "$E"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)  experiment: $E  args: $*"
