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

# Install pip nvcc matching the PyTorch CUDA version so C++ extensions
# compile against 12.6 regardless of the system CUDA.
pip install nvidia-cuda-nvcc-cu12==12.6.77

# Build a minimal CUDA_HOME from the pip-installed nvidia packages.
# This keeps the se3cnn C++ extension from seeing system CUDA 13.x.
CUDA_SHIM="${PWD}/.cuda_shim"
echo "Building CUDA shim at ${CUDA_SHIM}"
rm -rf "${CUDA_SHIM}"
mkdir -p "${CUDA_SHIM}/bin" "${CUDA_SHIM}/lib64" "${CUDA_SHIM}/include"

SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")

# nvcc binary
ln -sf "${SITE_PKGS}/nvidia/cuda_nvcc/bin/nvcc" "${CUDA_SHIM}/bin/nvcc"

# Headers and libs from every pip nvidia-cuda-* package
for pkg_dir in "${SITE_PKGS}"/nvidia/cuda_*/ "${SITE_PKGS}"/nvidia/cublas/ \
               "${SITE_PKGS}"/nvidia/cusolver/ "${SITE_PKGS}"/nvidia/cusparse/ \
               "${SITE_PKGS}"/nvidia/cufft/ "${SITE_PKGS}"/nvidia/curand/; do
    [ -d "${pkg_dir}/include" ] && cp -rsn "${pkg_dir}/include/"* "${CUDA_SHIM}/include/" 2>/dev/null || true
    [ -d "${pkg_dir}/lib" ]     && cp -rsn "${pkg_dir}/lib/"*     "${CUDA_SHIM}/lib64/"   2>/dev/null || true
done

export CUDA_HOME="${CUDA_SHIM}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
echo "  CUDA_HOME=${CUDA_HOME}  (nvcc: $(${CUDA_SHIM}/bin/nvcc --version | tail -1))"

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
