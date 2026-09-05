#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

OUTPUT_ROOT="output/TartanAir_V1"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

SEQUENCES=(SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003)
status=0

for seq in "${SEQUENCES[@]}"; do
  echo "============================================================"
  echo "Running ${seq}: ONLINE checkpoint + FULL final BA/refine"
  echo "============================================================"

  if [ "${CLEAN_OUTPUT:-0}" = "1" ]; then
    rm -rf "${OUTPUT_ROOT}/${seq}"
  fi

  python run_tartanair_v1.py "${seq}" \
    2>&1 | tee "${LOG_ROOT}/${seq}.log"

  rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo "[ERROR] ${seq} exited with code ${rc}" >&2
    status=1
  fi

done

echo
python scripts/summarize_tartanair_v1.py \
  SE000 SE001 SE002 SE003 SH000 SH001 SH002 SH003

exit "${status}"
