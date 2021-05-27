#System requirements and installation
## 1 System requirements
### 1.1 Hardware Requirements
#### 1.1.1 Recommended spec

Our program requires a Graphics Processing Unit (GPU) to accelerate the image processing. A large random access memory (RAM) is also preferred as 3D images usually have large sizes. 

We recommend a minimum spec as below: 

GPU: 8+ GB, supporting CUDA

RAM: 64+ GB

#### 1.1.2 Our spec:
Our program was developed and tested under following spec:

CPU: Intel CoreTM i7-6800K, 3.4GHz x 12 processor / Intel CoreTM i7-7800K, 3.5GHz x 12 processor

RAM: 16 GB / 64 GB

GPU: NVIDIA GeForce GTX 1080 (8GB)

### 1.2 Software Requirements:
Our program was developed under environment #1. To prove its compatibility with the most recent software versions, we also tested our program under environment #2.

|  | Environment #1 | Environment #2 |
| ------------- | ------------- | ------------- |
| OS | Linux (Ubuntu 16.04) | Linux (Ubuntu 16.04) |
| Programming language | Python 2.7 | Python 3.7 |
| For GPU acceleration | CUDA 8.0, cuDNN 6.0 | CUDA 10.0.130, cuDNN 7.3.1 |
| Distribution of Python | Anaconda 4.3.30 | Anaconda 2019.03 |
| IDE | Spyder 3.2.4 | Spyder 3.3.3 |
| Prerequisite Python packages | keras 2.0.8<br>tensorflow-gpu 1.4.0rc0<br>numpy 1.13.0<br>opencv 3.1.0<br>matplotlib 2.1.0<br>scikit-image 0.13.1<br>scikit-learn 0.19.1<br>scipy 1.0.0rc2<br>pillow 4.3.0<br>h5py 2.7.1 |keras 2.2.4<br>tensorflow-gpu 1.13.1<br>numpy 1.16.2<br>opencv 3.4.2<br>matplotlib 3.0.3<br>scikit-image 0.14.2<br>scikit-learn 0.20.3<br>scipy 1.2.1<br>pillow 5.4.1<br>h5py 2.8.0|

### 1.3 Other software required:
#### 1.3.1 Fiji: 
Fiji is a distribution of ImageJ. We used it for preprocessing (alignment) and confirmation of tracking results. It can be downloaded here: [https://imagej.net/Fiji/Downloads](https://imagej.net/Fiji/Downloads)

For alignment, we also used the plugin “StackReg”, which can be downloaded here: [http://bigwww.epfl.ch/thevenaz/stackreg/](http://bigwww.epfl.ch/thevenaz/stackreg/). In most cases, this step (alignment) can be skipped.
#### 1.3.2 ITK-SNAP: 
ITK-SNAP was used for manual correction of the segmentation, and for generating 3D movies. It can be downloaded here: [http://www.itksnap.org/](http://www.itksnap.org/). Users may use other tools for the correction, such as "napari" package in Python.

## 2. Installation guide
Here we describe the procedures for installation of environment #2:
### 2.1 Ubuntu
The installers for the latest version and for older releases can be downloaded here: [https://www.ubuntu.com/#download](https://www.ubuntu.com/#download). The link to tutorials for installing Ubuntu can be found on the same page. 

In this test, we installed Ubuntu 16.04 from a bootable USB.
### 2.2 Driver for an NVIDIA GPU
#### 2.2.1 Choose an appropriate driver version:
Before installing CUDA and cuDNN, users need to install an appropriate version of the GPU driver. The compatibility information can be found below:

Drivers for different GPU and OS: [https://www.geforce.com/drivers](https://www.geforce.com/drivers). 

Drivers for different CUDA version: (Table 1) [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

In our test, we installed nvidia-418.56, which supports our GPU GTX 1080 in Linux, and supports the latest CUDA10 which we installed afterwards (See below). An easy way to install the driver under Ubuntu is to use the Personal Package Archives (PPA). Users should confirm that the required driver is in the following list supported by PPA: [https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa)

#### 2.2.2 Install the driver using PPA:
The nvidia-418.56 were installed by running the following commands in the terminal:
```
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt-get update
$ sudo apt-get install nvidia-418
```
After restarting the computer, the driver should have been correctly installed.
Linux users can check the installation of the driver by a command: 
```
$ nvidia-smi
```
If the driver is correctly installed, the driver version should be displayed:
![nvidia-smi](/pictures/nvidia-smi.png)
### 2.3 Software or packages for segmentation and tracking
##### 2.3.1 Choose an appropriate version of Anaconda and install it:
The latest version of the Anaconda installer has included most of the software/packages we need. See here: [https://docs.anaconda.com/anaconda/packages/py3.7_linux-64/](https://docs.anaconda.com/anaconda/packages/py3.7_linux-64/). Other software/packages can also be easily installed from the Anaconda cloud. We therefore recommend users to install the latest Anaconda for convenience.

In this test, we installed Anaconda 2019.03 for linux 64bit with Python 3.7 which included Python, Spyder, numpy, matplotlib, scikit-image, scikit-learn, scipy, pillow, and h5py. The installer can be downloaded here: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

After download, we installed Anaconda by a command, and re-started the terminal (~1min): 
```
$ bash ~/Downloads/Anaconda_xxx.sh
```
Users should replace the above path after “bash” with the correct one containing the installer.

#### 2.3.2 Install required packages:
Users can install the required packages manually or using our 3DCT.yml file.

**Manually install:**
We recommend users to install and use following software/packages in a virtual Anaconda environment for safety. Here are the commands to create and activate a new environment (replace "yourenvname" with a custom name):
```
$ conda create -n yourenvname --clone base
$ source activate yourenvname
```
After installing Anaconda, we can install CUDA, cuDNN, tensorflow-gpu, and keras from the Anaconda cloud by a single command (~1 min): 
```
$ conda install keras-gpu
```
We can install opencv by another command (~1min):
```
$ conda install opencv
```

Finally, we should install our 3DeeCellTracker package by following command (~1min):
```
$ pip install 3DeeCellTracker
```

**Using 3DCT.yml file:**
Users should download the "3DCT.yml" file from this repository. Then they should open the terminal and change the working directory to the folder contain this yml file using cd command.

Then we can install the environment named "3DCT" by following command (~1min):
```
$ conda env create -f 3DCT.yml
```
