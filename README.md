# vibe-audio-algorithm
Traditional audio algorithm validation...
sudo apt-get update
sudo apt-get install libspeexdsp-dev  build-essential swig


conda create -n my-env python=3.10 -y
conda activate my-env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install numpy=1.24.4 -c conda-forge -y


pip install -r reauirement.txt  -i https://pypi.mirrors.ustc.edu.cn/simple 