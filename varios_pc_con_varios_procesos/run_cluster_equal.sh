#!/usr/bin/env bash
set -e

# Cambiar al directorio del script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Ejecución MPI en Cluster: N_procesos = N_ordenadores"
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

# Contar número de ordenadores definidos en el archivo hosts (ignorando líneas vacías y comentarios)
NUM_HOSTS=$(grep -vE '^\s*#|^\s*$' "$HOSTFILE" | wc -l)
if [ "$NUM_HOSTS" -eq 0 ]; then
    NUM_HOSTS=3
fi

# Parámetros: $1 número de procesos (por defecto igual al número de máquinas detectadas), $2 muestras
NP="${1:-$NUM_HOSTS}"
SAMPLES="${2:-1000}"

echo "----------------------------------------------------------"
echo "[Configuración] Hosts file: $HOSTFILE (Máquinas: $NUM_HOSTS)"
echo "[Configuración] Lanzando $NP procesos distribuidos ($NP procesos en $NUM_HOSTS máquinas)..."
echo "[Configuración] Muestras de entrenamiento: $SAMPLES"
echo "----------------------------------------------------------"

# Ejecución con mpirun usando --hostfile y N procesos = N máquinas
mpirun --hostfile "$HOSTFILE" -n "$NP" ./knn_mpi "$SAMPLES"

echo "=========================================================="
echo " Ejecución en cluster completada."
echo "=========================================================="
