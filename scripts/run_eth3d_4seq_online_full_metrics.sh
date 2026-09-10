#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

OUTPUT_ROOT="output/ETH3D_rectified"
mkdir -p "${OUTPUT_ROOT}/logs"

SEQUENCES=(
  mannequin_face_1
  einstein_1
  sofa_3
  plant_scene_3
)

if [ "${CLEAN_OUTPUT:-0}" = "1" ]; then
  for seq in "${SEQUENCES[@]}"; do
    rm -rf "${OUTPUT_ROOT}/${seq}"
  done
fi

status=0
for seq in "${SEQUENCES[@]}"; do
  echo "============================================================"
  echo "Running ETH3D ${seq}"
  echo "============================================================"

  python run_eth3d_rectified.py "${seq}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${seq}.log"

  rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo "[ERROR] ${seq} exited with code ${rc}" >&2
    status=1
  fi
done

echo
python scripts/summarize_tartanair_v1.py \
  --output_root "${OUTPUT_ROOT}" \
  mannequin_face_1 einstein_1 sofa_3 plant_scene_3

exit "${status}"
