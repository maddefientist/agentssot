#!/usr/bin/env bash
# Download LongMemEval datasets from the official HuggingFace mirror.
# The Oracle variant (~evidence-only sessions) is smallest and recommended for first runs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

BASE="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

VARIANT="${1:-oracle}"
case "${VARIANT}" in
  oracle) FILE="longmemeval_oracle.json" ;;
  s)      FILE="longmemeval_s_cleaned.json" ;;
  m)      FILE="longmemeval_m_cleaned.json" ;;
  all)
    for v in longmemeval_oracle.json longmemeval_s_cleaned.json longmemeval_m_cleaned.json; do
      [[ -f "${v}" ]] || wget -q --show-progress "${BASE}/${v}"
    done
    exit 0
    ;;
  *) echo "usage: $0 {oracle|s|m|all}"; exit 1 ;;
esac

if [[ -f "${FILE}" ]]; then
  echo "${FILE} already present, skipping"
else
  wget -q --show-progress "${BASE}/${FILE}"
fi

echo "Done. Dataset at: ${DATA_DIR}/${FILE}"
