#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run.sh <comando> <escena> <salida_o_modelo> [opciones] [-- args_extra]

Comandos:
  train <escena> <salida.pt>    Entrenar ACE sobre una escena
  test  <escena> <modelo.pt>    Evaluar un modelo ACE sobre una escena

La <escena> debe contener subcarpetas train/ y/o test/ con el formato ACE:
  escena/{train,test}/{rgb,poses,calibration}/

Opciones:
  --data-root PATH   Carpeta base de datos (default: preprocesamiento/data)
  --cpu              Forzar modo CPU (sin GPU)
  -h, --help         Mostrar esta ayuda

Ejemplos:
  ./run.sh train serie-1/ace output/serie-1.pt
  ./run.sh test serie-1/ace output/serie-1.pt
  ./run.sh train serie-1/ace output/serie-1.pt -- --num_head_blocks 2
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
FORCE_CPU=0
COMMAND=""
SCENE_PATH=""
MODEL_PATH=""
EXTRA_ARGS=()

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$1"
            elif [ -z "$SCENE_PATH" ]; then
                SCENE_PATH="$1"
            elif [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$COMMAND" ] || [ -z "$SCENE_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: se requieren <comando> <escena> <salida_o_modelo>."
    echo ""
    usage
    exit 1
fi

case "$COMMAND" in
    train|test) ;;
    *)
        echo "Error: comando invalido '$COMMAND'. Usa 'train' o 'test'."
        exit 1
        ;;
esac

DATA_ROOT=$(realpath "$DATA_ROOT")

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: la carpeta de datos '$DATA_ROOT' no existe."
    exit 1
fi

SCENE_FULL="${DATA_ROOT}/${SCENE_PATH}"
if [ ! -d "$SCENE_FULL" ]; then
    echo "Error: la escena '$SCENE_FULL' no existe."
    exit 1
fi

if [ "$COMMAND" = "train" ] && [ ! -d "$SCENE_FULL/train" ]; then
    echo "Error: no se encontro '$SCENE_FULL/train/'."
    echo "La escena debe tener subcarpeta train/ con {rgb,poses,calibration}/."
    exit 1
fi

if [ "$COMMAND" = "test" ] && [ ! -d "$SCENE_FULL/test" ]; then
    echo "Error: no se encontro '$SCENE_FULL/test/'."
    echo "La escena debe tener subcarpeta test/ con {rgb,poses,calibration}/."
    exit 1
fi

# --- Docker image selection ---
select_ace_image() {
    if docker image inspect ace:latest >/dev/null 2>&1; then
        echo "ace:latest"
    else
        echo "No se encontro la imagen ace:latest. Construyendo..." >&2
        local ace_src="${PROJECT_ROOT}/preprocesamiento/models/ace"
        docker build -f "${ace_src}/docker/Dockerfile" -t ace:latest "$ace_src" >&2
        echo "ace:latest"
    fi
}

# --- GPU detection ---
configure_gpu_args() {
    local image="$1"

    GPU_ARGS=()
    USE_GPU=0

    if [ "$FORCE_CPU" -eq 1 ]; then
        return
    fi

    if docker run --rm --gpus all --entrypoint nvidia-smi "$image" >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all)
        GPU_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all)
        GPU_ARGS+=(-e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
        return
    fi

    if docker run --rm --runtime=nvidia --entrypoint nvidia-smi "$image" >/dev/null 2>&1; then
        GPU_ARGS+=(--runtime=nvidia)
        GPU_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all)
        GPU_ARGS+=(-e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
    fi
}

ACE_IMAGE=$(select_ace_image)
configure_gpu_args "$ACE_IMAGE"

NUM_CPUS=$(nproc)
DOCKER_ARGS=(
    --rm
    -v "${DATA_ROOT}:/data"
    -w /ace
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
)

if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    DOCKER_ARGS+=("${GPU_ARGS[@]}")
fi

CONTAINER_SCENE="/data/${SCENE_PATH}"
CONTAINER_MODEL="/data/${MODEL_PATH}"

if [ "$COMMAND" = "train" ]; then
    ACE_ARGS=(
        train_ace.py
        "$CONTAINER_SCENE"
        "$CONTAINER_MODEL"
        --encoder_path /ace/ace_encoder_pretrained.pt
    )
else
    ACE_ARGS=(
        test_ace.py
        "$CONTAINER_SCENE"
        "$CONTAINER_MODEL"
        --encoder_path /ace/ace_encoder_pretrained.pt
    )
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    ACE_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Ensure output directory exists on the host
MODEL_DIR=$(dirname "${DATA_ROOT}/${MODEL_PATH}")
mkdir -p "$MODEL_DIR"

echo "Comando      : $COMMAND"
echo "Escena       : $SCENE_FULL"
echo "Modelo       : ${DATA_ROOT}/${MODEL_PATH}"
echo "GPU          : $USE_GPU"
echo "CPUs         : $NUM_CPUS"
echo ""

run_with_docker_hint() {
    if ! "$@"; then
        echo ""
        echo "La ejecucion en Docker fallo."
        echo "Si usas Docker Desktop, prueba primero: docker context use default"
        exit 1
    fi
}

run_with_docker_hint docker run "${DOCKER_ARGS[@]}" "$ACE_IMAGE" "${ACE_ARGS[@]}"
