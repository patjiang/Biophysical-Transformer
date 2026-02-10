## 🛠 Environment Setup (manual, you can also use the build.sh as well)

### 1. Create Conda Environment

First, initialize a clean Python 3.10 environment.

```bash
conda create -n EBPT python=3.10
source activate EBPT

```

### 2. Install SE3-CNN Dependencies

The  Convolutional Neural Network requires specific versions of PyTorch and a custom fork of the `se3cnn` library.

**A. Install PyTorch (CUDA 12.6)**

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

```

**B. Install Custom SE3-CNN Fork**

> [!NOTE]
> This fork includes a necessary fix for `scipy.linalg.eigh` calls in `SO3.py`.

```bash
git clone https://github.com/patjiang/se3cnn.git
cd se3cnn
python setup.py install
cd ..

```

**C. Install Helper Libraries**

```bash
pip install appdirs lie_learn scipy

```

### 3. Verify SE3-CNN Installation

Run this script in your Python REPL to ensure the  convolutions are working correctly:

```python
import torch
from se3cnn.image.convolution import SE3Convolution

size = 32
scalar_field = torch.randn(1, 1, size, size, size)
Rs_in = [(1, 0)]
Rs_out = [(1, 1)]
conv = SE3Convolution(Rs_in, Rs_out, size=5)

vector_field = conv(scalar_field)
print('Success! No errors!')

```

### 4. Install Final Dependencies

Install the remaining core libraries for training and data manipulation.

```bash
pip install pytorch-lightning==2.5.2 pot==0.9.5 einops==0.8.1 matplotlib==3.10.3

```

### 5. Final Compilation Test

Run the test suite to ensure everything is linked correctly:

```bash
python test.py

```
