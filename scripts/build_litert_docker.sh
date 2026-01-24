#!/bin/bash
# Build LiteRT using Docker hermetic build environment
# This avoids memory issues from compiling TensorFlow on resource-constrained servers
#
# Usage:
#   ./build_litert_docker.sh [--jobs N] [--memory LIMIT]
#
# Options:
#   --jobs N       Number of parallel build jobs (default: 4)
#   --memory LIMIT Docker memory limit, e.g. "8g" (default: no limit)

set -e

# Parse arguments
JOBS=4
MEMORY_LIMIT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --memory|-m)
            MEMORY_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--jobs N] [--memory LIMIT]"
            echo "  --jobs N       Parallel build jobs (default: 4)"
            echo "  --memory LIMIT Docker memory limit, e.g. '8g'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LITERT_DIR="${PLUGIN_DIR}/third_party/LiteRT"
PREBUILT_DIR="${PLUGIN_DIR}/prebuilt"

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64)
        LINUX_ARCH="x64"
        ;;
    aarch64|arm64)
        LINUX_ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Building LiteRT for Linux ${LINUX_ARCH} using Docker..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Check if LiteRT submodule exists
if [ ! -f "${LITERT_DIR}/docker_build/hermetic_build.Dockerfile" ]; then
    echo "Error: LiteRT submodule not initialized"
    echo "Run: git submodule update --init plugins/flutter_recorder/third_party/LiteRT"
    exit 1
fi

cd "${LITERT_DIR}"

# Build Docker image (reuse if exists)
IMAGE_NAME="litert_build_env"
if docker images -q "${IMAGE_NAME}" 2>/dev/null | grep -q .; then
    echo "Using existing Docker image: ${IMAGE_NAME}"
else
    echo "Building Docker image (this may take a while on first run)..."
    docker build -t "${IMAGE_NAME}" -f docker_build/hermetic_build.Dockerfile docker_build/
fi

# Create output directory
DEST_DIR="${PREBUILT_DIR}/linux/${LINUX_ARCH}"
mkdir -p "${DEST_DIR}"

# Run build inside Docker
CONTAINER_NAME="litert_build_$(date +%s)"
echo "Building libLiteRt.so inside Docker container..."
echo "  Jobs: ${JOBS}"
[ -n "${MEMORY_LIMIT}" ] && echo "  Memory limit: ${MEMORY_LIMIT}"

DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --security-opt seccomp=unconfined
    --user "$(id -u):$(id -g)"
    -e HOME=/litert_build
    -e USER="$(id -un)"
    -v "${LITERT_DIR}:/litert_build"
    -v "${DEST_DIR}:/output"
    -v "${HOME}/.cache/bazel:/root/.cache/bazel"
)

# Add memory limit if specified
if [ -n "${MEMORY_LIMIT}" ]; then
    DOCKER_ARGS+=(--memory "${MEMORY_LIMIT}")
fi

# Build and copy inside container (bazel-bin is a symlink, must copy before exit)
docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" \
    bash -c "./configure && bazel build //litert/c:litert_runtime_c_api_so --jobs=${JOBS} && cp -L bazel-bin/litert/c/libLiteRt.so /output/"

# Verify the output
if [ ! -f "${DEST_DIR}/libLiteRt.so" ]; then
    echo "Error: Built library not found at ${DEST_DIR}/libLiteRt.so"
    exit 1
fi

echo ""
echo "Success! LiteRT library built and cached."
echo "Output: ${DEST_DIR}/libLiteRt.so"
echo ""
echo "Future Flutter builds will use this prebuilt library automatically."
