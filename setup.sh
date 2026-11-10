#!/usr/bin/env bash
set -e

echo "Setting up BioT environment"

ENV_NAME="BioT"
PYTHON_VERSION="3.10"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists"
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
    conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
fi

echo "Activating environment '${ENV_NAME}'"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Installing PyTorch (CUDA 12.6)"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Point CUDA_HOME at PyTorch's bundled CUDA so C++ extensions compile
# against 12.6, not whatever version the system nvcc happens to be.
TORCH_CUDA_DIR=$(python -c "import torch.utils.cpp_extension as e; import os; print(os.path.dirname(e._find_cuda_home()))" 2>/dev/null || true)
if [ -z "${TORCH_CUDA_DIR}" ]; then
    # Fallback: locate the nvidia/cuda_runtime package that pip installed
    TORCH_CUDA_DIR=$(python -c "
import nvidia.cuda_runtime as cr, os
print(os.path.join(os.path.dirname(os.path.dirname(cr.__file__)), 'cuda_runtime'))
" 2>/dev/null || true)
fi

# The most reliable path: build the CUDA_HOME from PyTorch's own nvcc
NVCC_PATH=$(python -c "
import torch, pathlib, glob, os
base = pathlib.Path(torch.__file__).resolve().parent
candidates = sorted(glob.glob(str(base / '..') + '/nvidia/cuda_nvrtc/lib*'))
if candidates:
    print(os.path.dirname(os.path.dirname(candidates[0])))
" 2>/dev/null || true)

# Construct a minimal CUDA_HOME that has bin/nvcc and include/
# from the pip-installed nvidia packages
CUDA_SHIM="${PWD}/.cuda_shim"
echo "Building CUDA shim at ${CUDA_SHIM}"
rm -rf "${CUDA_SHIM}"
mkdir -p "${CUDA_SHIM}/bin" "${CUDA_SHIM}/lib64" "${CUDA_SHIM}/include"

# Symlink nvcc from the nvcc pip package
NVCC_BIN=$(python -c "
import glob, os, site
for d in site.getsitepackages():
    hits = glob.glob(os.path.join(d, 'nvidia', 'cuda_nvcc', 'bin', 'nvcc'))
    if hits:
        print(hits[0]); break
" 2>/dev/null || true)

if [ -n "${NVCC_BIN}" ] && [ -f "${NVCC_BIN}" ]; then
    ln -sf "${NVCC_BIN}" "${CUDA_SHIM}/bin/nvcc"
else
    # Use system nvcc as last resort but override the version check
    echo "  warning: pip nvcc not found, using system nvcc"
    ln -sf "$(which nvcc)" "${CUDA_SHIM}/bin/nvcc"
fi

# Symlink headers and libs from the pip CUDA packages
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
for pkg in cuda_runtime cuda_nvrtc cublas cusolver cusparse cufft curand; do
    pkg_dir="${SITE_PKGS}/nvidia/${pkg}"
    [ -d "${pkg_dir}/include" ] && cp -rsn "${pkg_dir}/include/"* "${CUDA_SHIM}/include/" 2>/dev/null || true
    [ -d "${pkg_dir}/lib" ]     && cp -rsn "${pkg_dir}/lib/"*     "${CUDA_SHIM}/lib64/"   2>/dev/null || true
done

export CUDA_HOME="${CUDA_SHIM}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
echo "  CUDA_HOME=${CUDA_HOME}"

SE3CNN_DIR="se3cnn-patch"

if [ -d "${SE3CNN_DIR}" ]; then
    echo "se3cnn repo already exists, skipping clone"
else
    echo "Cloning custom se3cnn fork"
    git clone https://github.com/patjiang/se3cnn-patch.git
fi

echo "Installing se3cnn"
cd ${SE3CNN_DIR}
pip install .
cd ..

echo "Installing helper libraries"
pip install appdirs lie_learn scipy torchinfo tensorboard scikit-learn pandas

echo "Installing core training dependencies"
pip install \
    pytorch-lightning==2.5.2 \
    pot==0.9.5 \
    einops==0.8.1 \
    matplotlib==3.10.3

echo "Verifying SE3CNN installation"
python - << 'EOF'
import torch
from se3cnn.image.convolution import SE3Convolution

size = 32
scalar_field = torch.randn(1, 1, size, size, size)
conv = SE3Convolution(Rs_in=[(1, 0)], Rs_out=[(1, 1)], size=5)
vector_field = conv(scalar_field)
print("SE3CNN OK")
EOF

echo "Installing package (editable)"
pip install -e .

if [ -f "test.py" ]; then
    echo "Running test.py"
    python test.py
else
    echo "test.py not found, skipping"
fi

echo "BioT setup complete!"
