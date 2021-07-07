# Introduction to 3DeeCellTracker

## What is 3DeeCellTracker
3DeeCellTracker is a Python based program using deep-learning techniques 
which can track moving cells in 3D time-lapse images.
### Things it can do
- It can track cells moving in the 3D time-lapse images from different 
  organs/animals obtained under variant optical conditions.
  
  - Traditional tracking programs were mostly designed for 2D images
    or for 3D images with specific organs/animals/optical conditions. 
    By utilizing multiple techniques including deep learning, our
    3DeeCellTracker is able to track cells under variant conditions.
  
- It can track large cell movements that are coherent.    
  - Here coherent means that each cell movements are not independent 
    but similar with its neighbouring cells. In other words, the 
    spatial pattern of cells should be consistent across time.
  - The coherency assumption allowed 3DeeCellTracker to track very large
    movements with translation/rotation/bending/contraction/extension, 
    such as the nerves cells in freely moving worm.
  - When the movements are small, the independent movements can also be 
    tracked by our method by searching new positions from the nearby cell 
    regions in the raw image. 
### Things it cannot do
- It cannot track large and independent cell movements
  - In such cases, the next positions of cells cannot be correctly 
    predicted because the spatial pattern changed. Also, the new positions 
    cannot be searched in nearby cell regions as the movement is large.
- It cannot handle new generated cells, such as cells moved into the
  visual field during imaging and cells divided into multiple cells
  - In such cases, the new generated cells will change the overall 
    spatial pattern of cells thus cause mistakes in tracking results.
  - In case only few new cells were generated, e.g., one or two new cells 
    in each time points while there are >100 cells in total, the other 
    cells can still be tracked. 

### How it works
#### Train 3D U-Net for a specific optical condition
- Prepare two different 3D images and annotate the corresponding 
  cell/non-cell regions. One image and annotation are used for training 
  the 3D U-Net. Another are used for validating the prediction accuracy. 
- Train the 3D U-Net to predict the cell/non-cell regions, which will
  generate a series of models (with different weights) and predictions 
- Choose and save a model with the best prediction.
- For tracking images obtained under the same condition, the same pre-trained
  model can be reused.
#### Segment cells use 3D U-Net and watershed
- Use the pretrained 3D U-Net to predict the cell/non-cell regions 
  in the images to be tracked
- Separate the cell regions into individual cells by watershed
- Manually correct the segmentation only in the volume 1. The positions 
  of these manually corrected cells will be tracked in the following 
  volumes.
#### Tracking the cell positions by FFN and other techniques
- Update the positions of the cells confirmed in volume 1 in following 
  volumes, i.e. volume 2, 3, 4, ..., until the last volume.
  
- In single mode, the positions were updated subsequently as:
  
  ![](http://latex.codecogs.com/gif.latex?Pos_2=f_{1,2}(Pos_1)),
  ![](http://latex.codecogs.com/gif.latex?Pos_3=f_{2,3}(Pos_2)),
  ![](http://latex.codecogs.com/gif.latex?Pos_{t+1}=f_{t,t+1}(Pos_t)),

  where each function f_t are estimated by FFN and other techniques 
  between volume t and t+1, based on the positions of cells segmented 
  by 3D U-Net and watershed (do not need manual correction)
- In ensemble mode, to improve the accuracy of tracking, 
  the positions at t+1 were instead estimated as the 
  mean of predictions from multiple previous volumes, such like:
  
  ![](http://latex.codecogs.com/gif.latex?Pos_{t+1}=\\frac{1}{N}\\sum_{i}^{N}f_{t-i*\\Delta,t}(Pos_{t-i*\\Delta}))

## Protocol
See Our GitHub repository for more information

### Install the environment
To use 3DeeCellTracker, Users need to install the prerequisite packages
in Python environment. To reduce the runtime related to the deep learning,
we also recommend users to run our programs in a desktop-PC with an 
NVIDIA GPU supporting CUDA. Here we will explain the procedure to install 
the environment in the Ubuntu OS. Users can also try it in 
other OS such as other macOS or Windows.

#### Ubuntu
The installers for the latest version and for older releases can be 
downloaded here: https://ubuntu.com/download/desktop. 
The link to tutorials for installing Ubuntu can be found on the same page.

#### The driver for an NVIDIA GPU (optional)
User can skip this step if their PC does not have GPU. But 
if possible, we recommend using an NVIDIA GPU which supports CUDA, 
because the GPU can largely reduce the runtime of codes involving 
deep learning.

>**Note:** Following instructions were tested in our PC with 
NVIDIA GPU GTX 1080 in the OS Ubuntu 16.04. The same procedures may 
fail in a different hardware/OS environment. We recommend users 
to find the latest solution on Internet. 
> 
> **Some useful information:**
> 1. Driver versions for different GPU and OS: 
https://www.geforce.com/drivers.
> 2. Correspondence between driver versions and CUDA versions: 
> 
>    https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (Table 1)

**Procedures:** To facilitate the deep learning by NVIDIA GPU, user need 
to install a driver for the GPU (here). Here we installed nvidia-418.56 
by running following commands in the terminal:

```
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt-get update
$ sudo apt-get install nvidia-418
```
After restarting the computer, the driver should have been correctly installed. Linux users can check the installation of the driver by a command:
```
$ nvidia-smi
```
If the driver has been correctly installed, the driver version should be displayed.

> **Useful information:**
> 
> An easy way to install the driver under Ubuntu is to use the 
Personal Package Archives (PPA). Here is the list of the available 
drivers in PPA: 
https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa


#### The python packages
**1. Install Anaconda:**
We recommend users to use anaconda to install the prerequisite 
python packages, which can be downloaded here:
https://www.anaconda.com/products/individual-b

After download, installed it by the command 
(replace the path/filename with the correct ones corresponding to the
installer)
```
$ bash ~/Downloads/Anaconda_xxx.sh
```
Then re-started the terminal.

**2. Install prerequisite packages:**
Users can quickly install the packages using our 3DCT.yml file stored in 
our GitHub repository: 
https://github.com/WenChentao/3DeeCellTracker.

After download the "3DCT.yml" file from this repository, 
run following command in the terminal under the directory 
containing "3DCT.yml"
```
$ conda env create -f 3DCT.yml
```
After creating the 3DCT environment, enter this environment by:
```
$ conda activate 3DCT
```

### Train a 3D U-Net model
#### Open the jupyter notebook
Users can download the jupyter notebook file "3D_U_Net_training-clear.ipynb" 
in our GitHub repository and run it in the browser. 

To open the notebook in
your browser, run following commands under the created 3DCT environment:
```
$ jupyter notebook
```
Then open the downloaded 3D_U_Net_training-clear.ipynb in your browser.

#### Run the notebook
To run the code cells (programs) and markdown cells (explanations), click 
the cell and press the run button in the top, 
or press shift + enter alternatively.

Before training the U-Net model, users should run the first code cell to 
import the required packages.

After that, modify and run following code cells subsequently:

**1. Initialize the parameters for training.**
    
Users should modify three parameters:
- noise_level: this value should be roughly the intensity of the 
  background pixels in the raw image. It is used in normalization to
  enhance the cells with weak intensity while ignore the background 
  noises. If the two images (train and validation) have different 
  noise level, choose a value somehow between them.
- folder_path: This path is used to create a folder (if not exist) to 
  store the data and model files. We recommend users to set it as 
  "./xxxx" to create a folder named xxxx under the same directory 
  containing the notebook file. 
- model: This should be a predefined 3D U-Net model. The simplest way is 
  to use the default value unet3_a(). Advanced users can select other 
  predefined model such as unet3_b(), unet3_c(), or a model defined by 
  themselves (need to modify the "Import packages" cell to import the model).
  
**2. Load the train / validation datasets**
- After running 1, the program automatically generated several folders 
  under the folder_path. Users should prepare the training data and 
  validation data (see section "How it works") and store them into the 
  "train_image"(raw 2D image sequence for training data), 
  "train_label"(2D annotation of cell/non-cell regions for training data),
  "valid_image" and "valid_label" (for validation data), respectively.

- Run the code cell. The program will load and draw the images/annotations
  of the training and validation dataset by max-projection.

**3. Preprocess the datasets**
- Run the first code cell. The program will normalize the images and show 
  the normalized images.
- If the normalization looks poor, e.g. the background look too bright, or 
  the intensity of weak cells are not enhanced. Please go back to step 1 
  to modify the parameter "noise_level" and run all codes again.
- Run the second code cell. The program will divide the images into multiple
  sub-images (used as the input of 3D U-Net) and show a part of these sub-images.

**4. Train the 3D U-Net**
- Run the code cell. The program will start to train the 3D U-Net. During 
  the training, the program will draw the updated prediction of cell 
  regions if the loss on validation data is reduced.
- By default, the program will train for 100 epochs. Users can manually 
  stop the training by pressing Ctrl+C if the val_loss no longer 
  decreases anymore.
  
**5.Select the best weights and save the model**
- After training finished or manually stopped. Users should choose the best 
  step which generate the best prediction of cell regions. Usually this should 
  be the one with the lowest val_loss, but uses can select other step instead.
- The program will store the model with the chosen weights into the "models" 
  folder with the name "unet3_pretrained.h5"

### Track the cells (single mode)
Users can download the jupyter notebook file 
"single_mode_worm1-clear.ipynb" and run it with some modifications.

Again, Before start tracking, users should run the first 
code cell to import the required packages.

**1. Initialize the parameters for tracking**

Users should modify several sets of parameters:

**1) Image parameters** 
- volume: number of the volumes (time points) of the images to be tracked 
- siz_xyz: size of each 3D image: (height, width, depth), unit: voxels 
- z_xy_ratio: resolution (um/voxels) ratio between z (depth) and x-y plane 
- z_scaling: (integer), the factor for interpolating the unbalanced z_xy resolution  
  
**2) Segmentation parameters** 
- noise_level: a value close to the averaged non-cell region intensity, which helps the program to ignore the background noises.
- min_size: the possible mimimal cell size, which helps the program to ignore small non-cell regions.
- Note: To change them after this initialization, please use ".set_segmentation()" which will delete the cached segmentation files  
  
**3) Tracking parameters** 
- beta_tk: set it higher/lower to get more coherent/independent predictions for cell positions.
- lambda_tk: set it higher/lower to get more coherent/independent predictions for cell positions.
- maxiter_tk: (integer) the number of iteration for the tracking algorithm (FFN+PR-GLS), the higher, the more accurate (but slower)  
  
**4) Paths** 
- folder_path: the path of the folder to store the data, model, and results.
    - "./xxx" indicates a folder with name "xxx" under the directory containing this jupyter notebook.
- image_name: the names of the images to be tracked.
    - "aligned_t%03i_z%03i.tif" indicates file names like: "aligned_t002_z011.tif", "aligned_t502_z101.tif", etc.
- unet_model_file: the name of the pre-trained unet model file - ffn_model_file: the name of the pre-trained ffn model file

**2. Prepare images to be tracked, and the pre-trained U-Net and FFN models.**

After running step 1, the program automatically created multiple folders 
under the folder_path. Users should move the images to be tracked into the
folder "data", and move the pre-trained 3D U-Net and FFN model into the folder
"models". An FFN model pre-trained by us can be downloaded from xxx. 

**3. Optimize segmentation parameters and segment the image at volume 1.**

- **Modify the segmentation parameters**
  
  This step should be skipped in the first time. 
  Modify the parameters only if the following segmentation result is poor.

- **Segment cells at volume 1**

  Run the code cell to segment the cells in volume 1

- **Draw the results of segmentation (Max projection)**

  Run the code cell to show the segmentation result: the raw image, 
  the cell regions, and the individual cells with max projection.
  
- **Show segmentation in each layer**

  Run the code cell to show the segmentation in each layer. If the 
  result in this cell and last cell are not satisfied, go back to the 
  beginning of step 3 to modify the parameters and run these codes 
  again.

**4. Manually correct the segmentation at volume 1 and load it.**
- **Manual correction**

  The automatically generated segmentation usually contains some mistakes.
  Users do not need to correct them in all volumes but only the first 
  volume in some software such as ITK-SNAP.
  
  Users should delete these fake cell regions and correct the boundaries 
  between cells with serious mistakes. Afterwards, only these 
  manually confirmed cells will be tracked. After correction, save the 
  results (as 2D image sequence) into the folder "manual_vol".
  
- **Load the manually corrected segmentation**

  Run the code cell to load the manually corrected segmentation.

- **Re-train the U-Net using the manual segmentation (optional)**
  
  This step can be skipped if the segmentation is already good 
  (i.e. all cells have been detected, and most of them are correctly 
  separated).
  
  To retrain the U-Net, run the first code cell, which will start the 
  training using the manual segmentation data within 10 steps.
  
  Once the retraining finished (or manually stopped). Run the second 
  code cell to select the best step (if the prediction is not improved, 
  set step=0 to restore the initial model)

- **Interpolate cells to make more accurate/smooth cell boundary**

  Just run the code cell to interpolate/smoothen the segmentation.
  
- **Initiate variables required for tracking**

  Just run the code cell to initial some variables required by tracking.

**5. Optimize tracking parameters.**

- **Modify tracking parameters if the test result is not satisfied (optional)**
  
  This step should be skipped in the first time.
  Modify the parameters only if the following matching result is poor.

- **Test a matching between volume 1 and a target volume, and show the FFN + PR-GLS process by an animation (5 iterations)**
  
  Set the target_volume and run the code cell to match the cells in 
  volume 1 with the cell in the target volume. Users can try multiple target
  volumes to find the most proper parameters.
  
  The program will generate an animation to show the 5 iterations of the 
  matching process which gradually improving the matching.
  
- **Show the accurate correction after the FFN + PR-GLS transformation**

  Run the code cell to show the tiny correction based on the intensities of 
  the raw images and the cell regions by 3D U-Net.
  
- **Show the superimposed cells + labels before/after tracking**

  Run the code cell to show the labels before and after tracking 
  superimposed on the cell regions (detected by 3D U-Net)
  
  If the matching results is poor with many mistakes, go back to the begining 
  of step 5 to modify the parameters and run these codes again.
  
**6. Track following volumes.**

- **Track and show the processes**

  Just run the code cell to track the following volumes. The program will
  update the tracking process in each volume during tracking.
  
  The tracked labels will be stored in the folder 
  "track_results_SingleMode" for further purposes.

- **Show the processes as an animation (for diagnosis)**

  After tracking finished, the tracking processes can be reloaded from 
  the folder "anim" by running this code cell for diagnosis.
  
### Track the cells (ensemble mode) 
Users can download the jupyter notebook file  
"ensemble_mode_worm4-clear.ipynb" and run it with some modifications.

The ensemble mode can be used similarly as in single mode except:
1. Initialize the parameters for tracking with additional parameter:
   ensemble=20, to predicts the cell positions as the average of 
   20 predictions.
2. This notebook also set a new parameter: miss_frame=[79], which means 
   the time point 79 will be skipped during tracking. This is because 
   volume 79 has a serious problem during imaging. In case not necessary, 
   this parameter can be skipped.
