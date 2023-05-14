# TJU计算机视觉课程设计——人脸检测及表情识别

天津大学计算机视觉课程设计。

基于深度卷积神经网络的人脸检测及表情识别。

[表情识别模块作者主页 ](https://github.com/VMnK-Run); [仓库地址](https://github.com/VMnK-Run/ImageProcessFinalProject)

## 环境配置

本项目使用到的语言、框架和开发环境为：

* Ubuntu20.04
* CUDA 11.7
* CUDNN 8

* Python 3.10
* PyTorch 1.13.1
* OpenCV

可通过以下命令配置环境：

```bash
conda create -n face2emoji python=3.10
conda activate face2emoji
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-contrib-python
pip install scikit-image
```

注：训练环境需要安装其他依赖，根据报错信息安装相应依赖即可。



## 使用方法：

如果你的当前设备有摄像头，并且处于空闲状态，你可以输入命令开始：

```bash
python run.py
```

如果你想以一个.mp4文件作为输入可以输入以下命令:

```bash
python ./run.py --source <path_to_your_.mp4_file>
```



