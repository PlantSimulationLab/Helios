#!/bin/bash
# Regenerate the summaries and plots in results/ from the runs in build/ (see README.md for how the runs were made).
cd "$(dirname "$0")"
PY=${PYTHON:-C:/isaacsim/python.bat}
B=build
mkdir -p results

compare() { out=$1; links=$2; sweep=$3; shift 3; "$PY" compare.py --links $links --sweep $sweep --out results/$out --plots "$@" > /dev/null || exit 1; }

R=$B/runs
compare apple_2ms $R/apple_links.csv $R/apple_sweep.json \
  --run "MuJoCo, no armature=$R/mujoco_apple_dt0.002_noarmature.csv" \
  --run "MuJoCo, armature ratio 1=$R/mujoco_apple_dt0.002_armature_ratio1.csv" \
  --run "MuJoCo, armature ratio 8=$R/mujoco_apple_dt0.002_armature.csv" \
  --run "PhysX, armature ratio 1=$R/isaac_apple_dt0.002_ratio1.csv" \
  --run "PhysX, armature ratio 8 (export default)=$R/isaac_apple_dt0.002_ratio8.csv"
compare apple_60hz $R/apple_links.csv $R/apple_sweep.json \
  --run "MuJoCo 2 ms, no armature=$R/mujoco_apple_dt0.002_noarmature.csv" \
  --run "MuJoCo 1/60 s, no armature=$R/mujoco_apple_dt0.016666667_noarmature.csv" \
  --run "MuJoCo 1/60 s, armature ratio 1=$R/mujoco_apple_dt0.016666667_armature_ratio1.csv" \
  --run "MuJoCo 1/60 s, armature ratio 8=$R/mujoco_apple_dt0.016666667_armature.csv" \
  --run "PhysX 1/60 s, armature ratio 1=$R/isaac_apple_dt0.0167_ratio1.csv" \
  --run "PhysX 1/60 s, armature ratio 8 (export default)=$R/isaac_apple_dt0.0167_ratio8.csv"
compare apple_30hz $R/apple_links.csv $R/apple_sweep.json \
  --run "MuJoCo 2 ms, no armature=$R/mujoco_apple_dt0.002_noarmature.csv" \
  --run "MuJoCo 1/30 s, no armature=$B/runs_steps/mujoco_apple_dt0.033333333_noarmature.csv" \
  --run "PhysX 1/30 s, armature ratio 1=$R/isaac_apple_dt0.0333_ratio1.csv" \
  --run "PhysX 1/30 s, armature ratio 8 (export default)=$R/isaac_apple_dt0.0333_ratio8.csv"
compare apple_friction $R/apple_links.csv $R/apple_sweep.json \
  --run "MuJoCo 1/60 s, armature ratio 1 (friction 0.5)=$R/mujoco_apple_dt0.016666667_armature_ratio1.csv" \
  --run "PhysX 1/60 s, armature ratio 1, dynamic friction 0.3 (export)=$R/isaac_apple_dt0.0167_ratio1.csv" \
  --run "PhysX 1/60 s, armature ratio 1, dynamic friction 0.5=$R/isaac_apple_dt0.0167_ratio1_mu05.csv"
compare mujoco_steps $R/apple_links.csv $R/apple_sweep.json \
  --run "MuJoCo 2 ms=$R/mujoco_apple_dt0.002_noarmature.csv" \
  --run "MuJoCo 1/60 s=$R/mujoco_apple_dt0.016666667_noarmature.csv" \
  --run "MuJoCo 1/30 s=$B/runs_steps/mujoco_apple_dt0.033333333_noarmature.csv" \
  --run "MuJoCo 1/20 s=$B/runs_steps/mujoco_apple_dt0.05_noarmature.csv" \
  --run "MuJoCo 1/10 s=$B/runs_steps/mujoco_apple_dt0.1_noarmature.csv"

S=$B/runs_settle
compare apple_static $S/apple_links.csv $S/apple_sweep.json \
  --run "MuJoCo 2 ms=$S/mujoco_apple_dt0.002_settle10_noarmature.csv" \
  --run "MuJoCo 1/60 s=$S/mujoco_apple_dt0.016666667_settle10_noarmature.csv" \
  --run "PhysX 2 ms, armature ratio 1=$S/isaac_apple_dt0.002_settle10_ratio1.csv" \
  --run "PhysX 1/60 s, armature ratio 1=$S/isaac_60_it32.csv" \
  --run "PhysX 1/60 s, armature ratio 1, 255/255 iterations=$S/isaac_60_it255.csv"

N=$B/runs_bean
compare bean_2ms $N/bean_links.csv $N/bean_sweep.json \
  --run "MuJoCo, no armature=$N/mujoco_bean_dt0.002_noarmature.csv" \
  --run "MuJoCo, armature ratio 1=$N/mujoco_bean_dt0.002_armature_ratio1.csv" \
  --run "MuJoCo, armature ratio 8=$N/mujoco_bean_dt0.002_armature.csv" \
  --run "PhysX, armature ratio 1=$N/isaac_bean_dt0.002_ratio1.csv" \
  --run "PhysX, armature ratio 8 (export default)=$N/isaac_bean_dt0.002_ratio8.csv"

cp $B/timing_results.txt results/timing_results.txt
echo "results/ regenerated"
