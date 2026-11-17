#!/usr/bin/env bash
set -euo pipefail

# Uso: scripts/compilar_cuda.sh [executável]
# BARENN_CUDA_ORACLE=1 inclui a captura de diagnóstico.
# BARENN_SYSTEM_INCLUDE aponta, se necessário, para os cabeçalhos do sistema.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-"$root/barenn_cuda"}
nvcc=${NVCC:-nvcc}
flags=(-std=c++17 -O3 --fmad=false)
if [[ ${BARENN_CUDA_ORACLE:-0} == 1 ]]; then
    flags+=(-DBARENN_ORACLE)
fi
if [[ -n ${BARENN_SYSTEM_INCLUDE:-} ]]; then
    if [[ ! -f "$BARENN_SYSTEM_INCLUDE/features.h" ]]; then
        echo "Cabeçalhos de sistema não encontrados: $BARENN_SYSTEM_INCLUDE" >&2
        exit 1
    fi
    flags+=("-Xcompiler=-idirafter,$BARENN_SYSTEM_INCLUDE")
fi
"$nvcc" "${flags[@]}" "$root/cuda/barenn_cuda.cu" -o "$output"
