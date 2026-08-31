#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
mkdir -p output/TartanAir_V1/logs

status=0
for seq in SH000 SH001 SH002 SH003; do
  echo "============================================================"
  echo "Running ${seq}"
  echo "============================================================"

  python run_tartanair_v1.py "${seq}" \
    2>&1 | tee "output/TartanAir_V1/logs/${seq}.log"

  rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo "[ERROR] ${seq} exited with code ${rc}" >&2
    status=1
  fi

done

echo
python scripts/summarize_tartanair_v1.py SH000 SH001 SH002 SH003
exit "${status}"
