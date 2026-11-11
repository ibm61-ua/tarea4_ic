#!/usr/bin/env bash
set -e

# Cambiar al directorio del script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Ejecutando solución MPI en 1 PC (knn_mpi)"
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

# Parámetros opcionales:
# $1: número de procesos (por defecto ejecuta demostración con 2 y con 4 procesos)
# $2: límite de muestras para ejecución rápida (por defecto 1000)
NP="${1:-}"
SAMPLES="${2:-1000}"

# Flag adicional de OpenMPI para entornos con pocos cores o WSL/contenedores
MPI_FLAGS="--oversubscribe"

if [ -n "$NP" ]; then
    echo "----------------------------------------------------------"
    echo "[Ejecución] Lanzando con $NP procesos y $SAMPLES muestras..."
    echo "----------------------------------------------------------"
    mpirun $MPI_FLAGS -np "$NP" ./knn_mpi "$SAMPLES"
else
    echo "----------------------------------------------------------"
    echo "[Test 1/2] Ejecución con 2 procesos (1 test especulativo + 1 tuning worker):"
    echo "----------------------------------------------------------"
    mpirun $MPI_FLAGS -np 2 ./knn_mpi "$SAMPLES"

    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 2/2] Ejecución con 4 procesos (1 test especulativo + 3 tuning workers):"
    echo "----------------------------------------------------------"
    mpirun $MPI_FLAGS -np 4 ./knn_mpi "$SAMPLES"
fi

echo "=========================================================="
echo " Ejecución en 1 PC completada."
echo "=========================================================="
