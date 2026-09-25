#!/usr/bin/env bash
# =============================================================================
# run_job.sh — Submete o experimento QWEN few-shot como job nohup persistente
# =============================================================================
# Cada rodada (run) vive em sua própria subpasta:
#   logs/experiments/exp01_qwen/run_<timestamp>/
#
# Comportamento:
#   sem --reset → continua a rodada ATUAL (mesma subpasta, retoma do checkpoint)
#   com --reset → cria uma rodada NOVA; a anterior fica preservada intacta
#
# Uso (da raiz do projeto):
#   chmod +x experiments/exp01_qwen_fewshot/run_job.sh
#   bash experiments/exp01_qwen_fewshot/run_job.sh
#   bash experiments/exp01_qwen_fewshot/run_job.sh --reset
#
# Monitore com:
#   bash experiments/exp01_qwen_fewshot/watch_progress.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_LOG_ROOT="${PROJECT_ROOT}/logs/experiments/exp01_qwen"
CURRENT_RUN_POINTER="${EXP_LOG_ROOT}/current_run.txt"

mkdir -p "${EXP_LOG_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

RESET_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "--reset" ]]; then
    RESET_FLAG="--reset"
  fi
done

if [[ -n "${RESET_FLAG}" ]]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
    echo "${RUN_NAME}" > "${CURRENT_RUN_POINTER}"
else
    if [[ -f "${CURRENT_RUN_POINTER}" ]]; then
        RUN_NAME="$(cat "${CURRENT_RUN_POINTER}")"
        if [[ ! -d "${EXP_LOG_ROOT}/${RUN_NAME}" ]]; then
            # Ponteiro órfão — pasta apagada manualmente
            RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
            echo "${RUN_NAME}" > "${CURRENT_RUN_POINTER}"
        fi
    else
        RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
        echo "${RUN_NAME}" > "${CURRENT_RUN_POINTER}"
    fi
fi

RUN_DIR="${EXP_LOG_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

JOB_LOG="${RUN_DIR}/job_output.log"
PID_FILE="${RUN_DIR}/job.pid"

echo ""
echo "========================================"
echo " exp01_qwen — GoEmotions few-shot (QWEN)"
echo "========================================"
echo " Projeto  : ${PROJECT_ROOT}"
echo " Rodada   : ${RUN_NAME}"
echo " Pasta    : ${RUN_DIR}"
echo " Log job  : ${JOB_LOG}"
echo " Python   : $(which ${PYTHON_BIN})"
echo "========================================"
echo ""

# Se já existe um job rodando NESTA rodada, avisa e sai
if [[ -f "${PID_FILE}" ]]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[WARN] Já existe um job rodando para esta rodada, PID ${OLD_PID}."
        echo "       Use 'kill ${OLD_PID}' para interrompê-lo antes de submeter novo job."
        exit 1
    else
        echo "[INFO] PID ${OLD_PID} não está mais ativo. Limpando..."
        rm -f "${PID_FILE}"
    fi
fi

cd "${PROJECT_ROOT}"

nohup ${PYTHON_BIN} \
    "${SCRIPT_DIR}/run_experiment.py" \
    ${RESET_FLAG} \
    >> "${JOB_LOG}" 2>&1 &

JOB_PID=$!
echo "${JOB_PID}" > "${PID_FILE}"

echo "[OK] Job submetido com PID ${JOB_PID}"
echo ""
echo "Comandos úteis:"
echo "  Ver log em tempo real  : tail -f ${JOB_LOG}"
echo "  Ver progresso          : watch -n 30 cat ${RUN_DIR}/progress.txt"
echo "  Script de monitoramento: bash experiments/exp01_qwen_fewshot/watch_progress.sh"
echo "  Verificar se está vivo : kill -0 ${JOB_PID} && echo 'rodando' || echo 'finalizado'"
echo "  Matar o job            : kill ${JOB_PID}"
echo ""
echo "Rodadas anteriores ficam preservadas em ${EXP_LOG_ROOT}/run_*/"
echo ""