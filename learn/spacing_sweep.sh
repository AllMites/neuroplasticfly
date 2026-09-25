#!/bin/sh
# Rung-3 interference was measured at exactly one spacing (relax=30). This adds
# 10 and 90; relax=30 is already on disk as p1s0..p1s4 from the rung-3 run.
# lesion is dropped: it was flat within noise across all three phases in rung 3,
# and it costs a third of the wall clock.
set -e
cd "$(dirname "$0")/.."
for R in 10 90; do
  for S in 0 1; do
    echo "=== relax $R seed $S ==="
    .venv/Scripts/python.exe learn/persist.py --state sp${R}s${S} --relax $R --seed0 $S \
      --arms retain,interfere
  done
done
echo "=== analysis ==="
.venv/Scripts/python.exe learn/analyze_spacing.py sp10s0 sp10s1 p1s0 p1s1 sp90s0 sp90s1
