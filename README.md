# Electrostatics-BiophysicalTransformer
WIP


### Environment setup instructions
#### conda instructions
1) Setup conda environment
```
conda create -n EBPT python=3.10
source activate EBPT
```
2) Install SE3 cnn dependencies (need to install pytorch first)

- Currently hosted under me due to a small change needed for the scipy.linalg.eigh call in SO3.py. I have submitted a pull request, but until then, this will work as well.
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/patjiang/se3cnn.git
cd se3cnn
python setup.py install
```

- Additional dependencies need to be installed as well:
```
pip install appdirs lie_learn scipy
```
a) To test proper installation of SE3CNN, run the following script

- It is fine to paste in python REPL within the conda environment is fine; i.e. ```source activate EBPT```; ```python```
```{python}
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
3) Install pytorch-lightning, other dependencies
```
pip install pytorch-lightning==2.5.2 pot==0.9.5 einops==0.8.1 matplotlib==3.10.3
```

4) Test that it will compile by running
```
python test.py
```
- And you're done!
