# vibe-audio-algorithm
Traditional audio algorithm validation...
## Environment setup
```shell
sudo apt-get update
sudo apt-get install libspeexdsp-dev  build-essential swig

conda create -n my-env python=3.10 -y
conda activate my-env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install numpy=1.24.4 -c conda-forge -y

pip install -r reauirement.txt  -i https://pypi.mirrors.ustc.edu.cn/simple 
```
## Usage
```shell
cd vibe-audio-algorithm
python main.py
```

## Complie webrtc aec3  for python
### 1.Prepare dependency
```shell
sudo apt update
sudo apt install -y git python3 python3-pip python3-setuptools \
    pkg-config ninja-build build-essential libglib2.0-dev \
    clang cmake libtool yasm libasound2-dev

git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
echo 'export PATH=$PATH:$HOME/depot_tools' >> ~/.bashrc
source ~/.bashrc
```
### 2. Fetch webrtc code
```shell
mkdir webrtc-checkout && cd webrtc-checkout
fetch --nohooks webrtc
gclient sync
```

### 3. Config GN 
```shell
cd src
gn gen out/aec3 --args='
  is_debug=false
  rtc_include_tests=false
  rtc_build_examples=false
  target_os="linux"
  target_cpu="x64"
  use_custom_libcxx=false
'
```
Need to commomed some error lines.

### 4. Compile code
```shell
ninja -C out/aec3 modules/audio_processing:audio_processing
```
Under folder out/aec3/obj and out/aec3/, there are important out files: libaudio_processing.a, audio_processing.h 

### 5. Write and package the C-API wrapper and compile it as libaec3.so
```shell
g++ -fPIC -shared -o libaec3.so aec_wrapper.c -I./ -L./out/aec3/obj/audio_processing -laudio_processing
```
With ctypes.CDLL("libaec3.so"), python could call the API
