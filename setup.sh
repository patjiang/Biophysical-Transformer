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

echo "Installing CUDA 12.6 nvcc compiler to match PyTorch"
conda install -y -c nvidia cuda-nvcc=12.6.*

echo "Installing PyTorch (CUDA 12.6)"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

SE3CNN_DIR="se3cnn-patch"

if [ -d "${SE3CNN_DIR}" ]; then
    echo "se3cnn repo already exists, skipping clone"
else
    echo "Cloning custom se3cnn fork"
    git clone https://github.com/patjiang/se3cnn-patch.git
fi

echo "Installing se3cnn"
cd ${SE3CNN_DIR}

export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export FORCE_CUDA=1

python setup.py install
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
conda activate BioT
if [ -f "test.py" ]; then
    echo "Running test.py"
    python test.py
else
    echo "test.py not found, skipping"
fi

echo "BioT setup complete!"
