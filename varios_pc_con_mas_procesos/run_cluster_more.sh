#!/usr/bin/env bash
set -e

# Cambiar al directorio del script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Ejecución MPI en Cluster: N_procesos > N_ordenadores"
echo "=========================================================="

mkdir -p build
cd build

echo "[Info] Configurando y compilando con CMake y MPI..."
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target knn_mpi

# Comprobar presencia del dataset MNIST
if [ ! -f "train/train-images-idx3-ubyte" ] && [ ! -f "../train/train-images-idx3-ubyte" ]; then
    echo "[Info] Dataset no encontrado. Descargando MNIST..."
    (cd .. && bash ./get_dataset.sh) || true
fi

HOSTFILE="../hosts"
if [ ! -f "$HOSTFILE" ]; then
    echo "[Error] No se encontró el archivo de hosts en $HOSTFILE"
    exit 1
fi

# Detectar el total de slots configurados en el hostfile
TOTAL_SLOTS=$(awk -F'slots=' '{if(NF>1) sum+=$2} END {print sum+0}' "$HOSTFILE")
if [ "$TOTAL_SLOTS" -eq 0 ]; then
    TOTAL_SLOTS=12
fi

# Parámetros opcionales:
# $1: número de procesos (ej: 6 o 12 para sobrepasar el número de nodos físicos)
# $2: límite de muestras para ejecución rápida (ej: 1000)
NP="${1:-6}"
SAMPLES="${2:-1000}"

echo "----------------------------------------------------------"
echo "[Configuración] Hosts file: $HOSTFILE (Total slots: $TOTAL_SLOTS)"
echo "[Configuración] Lanzando $NP procesos distribuidos (más procesos que máquinas)..."
echo "[Configuración] Muestras de entrenamiento: $SAMPLES"
echo "----------------------------------------------------------"

# Ejecución con mpirun usando --hostfile, --oversubscribe y N procesos > N máquinas
mpirun --hostfile "$HOSTFILE" --oversubscribe -n "$NP" ./knn_mpi "$SAMPLES"

echo "=========================================================="
echo " Ejecución con más procesos completada con éxito."
echo "=========================================================="
