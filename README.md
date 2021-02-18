# 3DeeCellTracker - A quick guide

This pipeline was designed for segmenting and tracking cells in 3D + T images in deforming organs. The methods have been explained in Wen et al. bioRxiv 2018 (doi: https://doi.org/10.1101/385567) and in another paper during revision. 
The data for demonstration and the previous demo codes can be found in: http://ssbd.qbic.riken.jp/set/20190602/

## 1 System requirements
### 1.1 Hardware Requirements
#### 1.1.1 Recommended spec

Our program requires a Graphics Processing Unit (GPU) to accelerate the image processing. A large random access memory (RAM) is also preferred as 3D images usually have large sizes. 

We recommend a minimum spec as below: 

GPU: 8+ GB, supporting CUDA

RAM: 16+ GB

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
| Prerequired Python packages | keras 2.0.8<br>tensorflow-gpu 1.4.0rc0<br>numpy 1.13.0<br>opencv 3.1.0<br>matplotlib 2.1.0<br>scikit-image 0.13.1<br>scikit-learn 0.19.1<br>scipy 1.0.0rc2<br>pillow 4.3.0<br>h5py 2.7.1 |keras 2.2.4<br>tensorflow-gpu 1.13.1<br>numpy 1.16.2<br>opencv 3.4.2<br>matplotlib 3.0.3<br>scikit-image 0.14.2<br>scikit-learn 0.20.3<br>scipy 1.2.1<br>pillow 5.4.1<br>h5py 2.8.0|

### 1.3 Other software required (for preprocessing and confirmation/correction of results):
#### 1.3.1 Fiji: 
Fiji is a distribution of ImageJ. We used it for preprocessing (alignment) and confirmation of tracking results. It can be downloaded here: [https://imagej.net/Fiji/Downloads](https://imagej.net/Fiji/Downloads)

For alignment, we also used the plugin “StackReg”, which can be downloaded here: [http://bigwww.epfl.ch/thevenaz/stackreg/](http://bigwww.epfl.ch/thevenaz/stackreg/). In most cases, this step (alignment) can be skipped.
#### 1.3.2 ITK-SNAP: 
ITK-SNAP was used for manual correction of the segmentation, and for generating 3D movies. It can be downloaded here: [http://www.itksnap.org/](http://www.itksnap.org/). Users may use other tools for the correction, such as "napari" package in Python.

## 2. Installation guide
Here we describe the procedures for installation of environment #2:
### 2.1 Ubuntu (~15 mins)
The installers for the latest version and for older releases can be downloaded here: [https://www.ubuntu.com/#download](https://www.ubuntu.com/#download). The link to tutorials for installing Ubuntu can be found on the same page. 

In this test, we installed Ubuntu 16.04 from a bootable USB.
### 2.2 Driver for a NVIDIA GPU
#### 2.2.1 Choose an appropriate driver version:
Before installing CUDA and cuDNN, users need to install an appropriate version of the GPU driver. The compatibility information can be found below:

Drivers for different GPU and OS: [https://www.geforce.com/drivers](https://www.geforce.com/drivers). 

Drivers for different CUDA version: (Table 1) [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

In our test, we installed nvidia-418.56, which supports our GPU GTX 1080 in Linux, and supports the latest CUDA10 which we installed afterwards (See below). An easy way to install the driver under Ubuntu is to use the Personal Package Archives (PPA). Users should confirm that the required driver is in the following list supported by PPA: [https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa)

#### 2.2.2 Install the driver using PPA (~10 mins):
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
### 2.3 Software/packages for segmentation and tracking
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

## 3. Use 3DeeCellTracker (explained by a demo dataset)
We have compressed the demo data, the weights of pre-trained 3D U-net/FFN models, and the segmentation/tracking results into "Demos190610.zip" which can be downloaded from: [http://ssbd.qbic.riken.jp/set/20190602/](http://ssbd.qbic.riken.jp/set/20190602/)
### 3.1 Tracking 
#### 3.1.1 Preparations: 
We suggest that users should use an IDE such as Spyder to run the "cell_segment_track.py" under "./Tracking/".

1. Modify following path and file names in "cell_segment_track.py", including: 
- "folder_path" (containing data, models and segmetation/tracking results),  
- "files_name"(of raw images), 
- "unet_weight_file" (name of 3D U-net weight file)
- "FFN_weight_file" (name of FFN weight file)
2. Put raw images into "data" folder and unet and FFN weight files into "models" folder.
3. Modify global parameters (see the [user-guide for setting parameters](https://github.com/WenChentao/3DeeCellTracker/blob/master/Guide%20for%20parameters.md)).
#### 3.1.2 Procedures for tracking:
##### 3.1.2.1 Segmentation of volume #1
Run the code in "cell_segment_track.py" until finishing "automatic segmentation of volume #1". The resulted segmentation is stored into the folder “auto_vol1”

| Raw image | Segmentation result |
| ------------- | ------------- | 
| ![raw-worm1](/pictures/raw-worm1.png) | ![autoseg-worm1](/pictures/autoseg-worm1.png) |

 (Optional) Users can check the segmentation in Fiji. Here are the 2D projected raw images in volume #1 (left, color = “fire”; images 1-21) and segmentation results (right, color = “3-3-2 RGB”): 
 
##### 3.1.2.2 Manually correct the segmentation in volume #1 
Users should correct the segmentation in other software such as ITK-SNAP. For the demo data, we have included the corrected segmentation in folder “manual_vol1”. Here is the 2D projection of our corrected segmentation:
<img src="/pictures/manualseg-worm1.png" width="400">
##### 3.1.2.3 Track cells in all volumes
Run "cell_segment_track.py" to the end. The tracked labels are stored into the folder “track_results”. 

<img src="/pictures/track-worm1.gif" width="400">

(Optional) Users can check the tracking results in Fiji by comparing the raw images and tracked labels:

Users can use other software for checking the results, such as in IMARIS.

### 3.2 Training 3D U-net
#### 3.2.1 Preparations: 
For training 3D U-net, users should run "unet_training.py" under "./UnetTraining/".

Again, users should modify the "folder_path" (containing following data and results) accordingly.

Similarly, users should create folders for storing 
- weights (in folder "weights"), 
- predictions (in folder "prediction"), 
- 3D image for training (in folder "train_image") and for the corresponding annotated cell image (in folder "train_cells"), 
- another 3D image and corresponding annotation for validation (in folder "valid_image" and "valid_cells"). 

In the demo data, we have supplied necessary image data for training and validation (see below). 

Training data (Projected to 2D plane):
| Raw image (color = "fire") | Cell image (binary value) |
| ------------- | ------------- | 
| ![raw-unet](/pictures/raw-unet.png) | ![seg-unet](/pictures/seg-unet.png) |

Validation data:
| Raw image | Cell image |
| ------------- | ------------- | 
| ![raw-unet-valid](/pictures/raw-unet-valid.png) | ![seg-unet-valid](/pictures/seg-unet-valid.png) |

The structure of the 3D U-net can be modified to existing ones: "unet3_a ", "unet3_b", or "unet3_c". For example, to use the structure "a", we could run following codes: 
```
from CellTracker.unet3d import unet3_a
unet_model = unet3_a() 
```
Users can also define their own structures of 3D U-net in "unet3d.py" under "./UnetTraning/CellTracker/". 
#### 3.2.2 Procedures for training 3D U-net:
##### 3.2.2.1 Train the 3D U-net
Users should run the codes until finishing training the 3D U-net. Here we trained 30 epochs (30 cycles). Users can increase the number of epochs if they wish to obtain a more accurate model. 

To show the history of the errors during training by authors, we plotted the loss function (binary cross-entropy) on training data and validation data (figure below).

Notice the training process will be different every time owning to randomness, but users should observe a quick decrease of loss though early stage of training.

<img src="/pictures/unet-loss.svg" width="400">

##### 3.2.2.2 Check the prediction of the U-net 
The loss function gives us a quantitative evaluation of the error rates, but we also need an intuitive impression to judge the prediction. 

During training, those weights (parameters of the 3D U-net) with decreased loss were stored for different epochs (here 1-30). Users should first confirm in which epochs weights were stored (in folder "weights"). Then users can load weights and predict cell regions in training and validation images (results are saved in folder "prediction"). Here we show predictions corresponding to 3 different epochs (figure below). Users may obtain different predictions but as a trend the accuracy should be improved gradually. 

Cell regions (following images are probability maps: black:0, white:1) predicted from raw images

(1) For training data. 
| epoch = 1(weight = 1) | epoch = 2 | epoch = 28 |
| ------------- | ------------- | ------------- | 
| ![unet-epoch1](/pictures/unet-epoch1.png) | ![unet-epoch2](/pictures/unet-epoch2.png) | ![unet-epoch28](/pictures/unet-epoch28.png) |

(2) For validation data. 
| epoch = 1 | epoch = 2 | epoch = 28 |
| ------------- | ------------- | ------------- | 
| ![unet-epoch1-valid](/pictures/unet-epoch1-valid.png) | ![unet-epoch2-valid](/pictures/unet-epoch2-valid.png) | ![unet-epoch28-valid](/pictures/unet-epoch28-valid.png) |

### 3.3 Training the feedforward network (FFN)
#### 3.3.1 Preparations: 
For training FFN, users should run "FFNTraining.py" under "./FFNTraining/".

Again, users should modify the "folder_path" (containing following data and results) and prepare two point sets including the training data and test data (in folder "data"). In the demo data, we have supplied a point set for training and another point set for test. 

The training data were generated by simulations from the training point set. Here are some typical generated training data examples (Projected to 2D plane): 

Red circles: raw point set. Blue crosses: generated point set with simulated movements.
| example data1 | example data2 | example data3 |
| ------------- | ------------- | ------------- | 
| ![example data1](/pictures/ffn-data1.svg) | ![example data2](/pictures/ffn-data2.svg) | ![example data3](/pictures/ffn-data3.svg) |

#### 3.3.2 Procedures:
##### 3.3.2.1 Train the FFN
Users should run the code "FFNTraining.py" to train the FFN. Please note some parameters used in this demonstration are different with the ones used in our paper, in order to make larger movements and to reduce the training time. The default number of epochs for training is 30. Users can increase the epochs to get a more accurate model. We only measured the loss function (binary cross-entropy) on the generated training data (always different in each epoch due to random simulations). 

<img src="/pictures/ffn-loss.svg" width="400">

##### 3.3.2.2 Check the prediction of FFN on test data
Users can load weights corresponding to different epochs and save predicted matching between the training point set and the test point set (in folder "prediction"). Here we show predictions corresponding to 3 different epochs:

Circles on top: training point set; Crosses on bottom: test point set. Red lines: predicted matching.
| epoch = 1 | epoch = 9 | epoch = 23 |
| ------------- | ------------- | ------------- | 
| ![ffn-epoch1](/pictures/ffn-epoch1.png) | ![ffn-epoch9](/pictures/ffn-epoch9.png) | ![ffn-epoch23](/pictures/ffn-epoch23.png) |
