#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Mirror de CVDF allotjat en Google Cloud Storage.
base="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p build/train build/test

for name in train-images-idx3-ubyte train-labels-idx1-ubyte             t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    case "$name" in
        train-*) folder="build/train" ;;
        *)       folder="build/test" ;;
    esac

    # No tornar a descarregar fitxers que ja tenim.
    if [ -s "$folder/$name" ]; then
        continue
    fi

    archive=$(mktemp "$folder/archive.XXXXXX")
    unpacked=$(mktemp "$folder/unpacked.XXXXXX")
    trap 'rm -f -- "$archive" "$unpacked"' EXIT

    # Reintentar errors temporals, inclosos 429 i 503.
    wget --tries=8 --waitretry=5 --timeout=30          --retry-on-http-error=429,500,502,503,504          -O "$archive" "$base/$name.gz"

    # Només publicar el fitxer quan la descompressió acaba correctament.
    gzip -dc "$archive" > "$unpacked"
    mv -- "$unpacked" "$folder/$name"
    rm -f -- "$archive"
    trap - EXIT
done

cmake -S . -B build
cmake --build build
