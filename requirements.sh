# This document creates a enviroment for running TAPIR MICCAI 2022

# Create environment
# conda create --name tapir python=3.9
# conda activate tapir

# # Install pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Fvcore
pip install 'git+https://github.com/facebookresearch/fvcore'

# simplejson
pip install simplejson

# PyAV
conda install av -c conda-forge

# Iopath
pip install -U iopath

# psutil
pip install psutil

# opencv
pip install opencv-python

# tensorboard
pip install tensorboard

# moviepy
pip install moviepy

# PyTorchVideo
pip install pytorchvideo

# Fairscale
pip install 'git+https://github.com/facebookresearch/fairscale'

pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

pip install pandas
pip install Pillow
pip install scikit-learn
pip install psutil

# RoCL
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install diffdist==0.1