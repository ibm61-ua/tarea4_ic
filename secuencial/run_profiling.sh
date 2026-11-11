#!/usr/bin/env bash
set -e

# Cambiar al directorio del script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Ejecutando profiling secuencial con gprof"
echo "=========================================================="

mkdir -p build
cd build

# Comprobar si se solicita compilación con -O2 (opcional: ./run_profiling.sh O2 [muestras])
OPT_FLAG="OFF"
if [ "${1:-}" = "O2" ] || [ "${1:-}" = "-O2" ]; then
    echo "[Info] Configurando con optimización -O2 y profiling (-pg -g -O2)..."
    OPT_FLAG="ON"
    shift
else
    echo "[Info] Configurando con profiling base (-pg -g -O0)..."
fi

cmake .. -DENABLE_PROFILING=ON -DPROFILING_O2="$OPT_FLAG"
cmake --build . --target cv

# Comprobar presencia del dataset MNIST
if [ ! -f "train/train-images-idx3-ubyte" ] && [ ! -f "../train/train-images-idx3-ubyte" ]; then
    echo "[Info] No se detectó el dataset MNIST. Intentando descargar con get_dataset.sh..."
    (cd .. && bash ./get_dataset.sh) || true
fi

# Si se pasa un argumento numérico, se usa como límite de muestras para ejecución rápida
SAMPLES="${1:-1000}"
echo "[Info] Ejecutando ./cv con $SAMPLES muestras para profiling..."
./cv "$SAMPLES"

# Comprobar que se ha generado gmon.out
if [ ! -f "gmon.out" ]; then
    echo "[Error] No se ha generado gmon.out. Verifica la compilación con -pg."
    exit 1
fi

echo "[Info] Extrayendo informe de profiling con gprof..."
gprof -b -p -q cv gmon.out > perfil.txt

echo "=========================================================="
echo " Profiling completado con éxito."
echo " Informe generado en: $(pwd)/perfil.txt"
echo "=========================================================="
