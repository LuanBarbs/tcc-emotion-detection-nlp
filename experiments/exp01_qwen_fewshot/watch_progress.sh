#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_LOG_ROOT="${PROJECT_ROOT}/logs/experiments/exp01_qwen"
CURRENT_RUN_POINTER="${EXP_LOG_ROOT}/current_run.txt"
mkdir -p "${EXP_LOG_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESET_FLAG=""
for arg in "$@"; do [[ "$arg" == "--reset" ]] && RESET_FLAG="--reset"; done

if [[ -n "${RESET_FLAG}" || ! -f "${CURRENT_RUN_POINTER}" ]]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"; echo "${RUN_NAME}" > "${CURRENT_RUN_POINTER}"
else
    RUN_NAME="$(cat "${CURRENT_RUN_POINTER}")"
    [[ -d "${EXP_LOG_ROOT}/${RUN_NAME}" ]] || { RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"; echo "${RUN_NAME}" > "${CURRENT_RUN_POINTER}"; }
fi
RUN_DIR="${EXP_LOG_ROOT}/${RUN_NAME}"; mkdir -p "${RUN_DIR}"
JOB_LOG="${RUN_DIR}/job_output.log"; PID_FILE="${RUN_DIR}/job.pid"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[WARN] Já existe job rodando (PID $(cat "${PID_FILE}"))."; exit 1
fi

cd "${PROJECT_ROOT}"   # roda a partir da raiz (caminhos relativos do config)
nohup ${PYTHON_BIN} "${SCRIPT_DIR}/run_experiment.py" ${RESET_FLAG} >> "${JOB_LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[OK] Job submetido (PID $(cat "${PID_FILE}")). Log: ${JOB_LOG}"