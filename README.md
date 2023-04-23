# 3DeeCellTracker
[![PyPI](https://img.shields.io/pypi/v/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)
[![Youtube](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=ctt6o3DY2bA&list=PLGY0oNQomrHERP08iEj-MsluFW8xQJujP)

**3DeeCellTracker** is a deep-learning based pipeline for tracking cells in 3D time lapse images of deforming/moving organs ([eLife, 2021](https://elifesciences.org/articles/59187)).

## Updates:

**3DeeCellTracker v0.5.0-alpha will be released soon**
- Allows you to use [StarDist3D](https://github.com/stardist/stardist) for segmentation
- Reduces the requirements for fine-tuning parameters
- Decouples the code to facilitate reuse by third-party developers.

## Installation

To install 3DeeCellTracker, please follow the instructions below:

### Prerequisites
- A computer with an NVIDIA GPU that supports CUDA.
- [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://conda.io/miniconda.html) installed.

### Steps
1. Create a new conda environment and activate it by running the following commands in your terminal:

   ```console
   $ conda create -n track python=3.8 pip
   $ conda activate track
   ```
   
2. Install [TensorFlow](https://www.tensorflow.org/install).
3. Install 3DeeCellTracker by running the following command in your terminal:
   ```console
   $ pip install 3DeeCellTracker
   ```
   After completing the installation steps, you can start using 3DeeCellTracker for your 3D cell tracking tasks within 
   the jupyter notebooks we have provided (See below). 
   If you encounter any issues or have any questions, please refer to the project's documentation 
   or raise an issue in the GitHub repository.

## Quick Start
To learn how to track cells using 3DeeCellTracker, please refer to the following notebooks for examples. 
We recommend using StarDist for segmentation, as we have optimized the StarDist-based tracking programs for more convenient and quick cell tracking.
1. Train a custom deep neural network for segmenting cells in new optical conditions: 
    - [**Train 3D StarDist (notebook with results)**](Examples/use_stardist/train_stardist.ipynb)
    - [**Train 3D U-Net (clear notebook)**](Examples/use_unet/3D_U_Net_training-clear.ipynb).
    - [**Train 3D U-Net (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/use_unet/3D_U_Net_training.html).
 
2. Track cells in deforming organs: 
    - [**Single mode + StarDist (notebook with results)**](Examples/use_stardist/track_worm1_stardist_single_mode.ipynb);
    - [**Single mode + UNet (clear notebook)**](Examples/use_unet/single_mode_worm1-clear.ipynb);
    - [**single mode + UNet (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/use_unet/single_mode_worm1.html)

3. Track cells in freely moving animals: 
    - [**Ensemble mode + StarDist (notebook with results)**](Examples/use_stardist/track_worm4_stardist_ensemble_mode.ipynb);
    - [**Ensemble mode + UNet (clear notebook)**](Examples/use_unet/ensemble_mode_worm4-clear.ipynb)
    - [**Ensemble mode + UNet (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/use_unet/ensemble_mode_worm4.html)

   
The data and model files for demonstrating above notebooks can be downloaded here: 
- [**StarDist-based notebooks**](https://osf.io/pgr95/).
- [**UNet-based notebooks**](https://osf.io/dt76c/).


**Note**: Codes above were based on the latest version. 
For old programs used in eLife 2021, please check the "[**Deprecated_programs**](Deprecated_programs)" folder.

## Frequently Reported Issue and Solution (for v0.4)

Multiple users have reported encountering a `ValueError` of shape mismatch when running the `tracker.match()` function. 
After investigation, it was found that the issue resulted from an incorrect setting of `siz_xyz`, 
which should be set to the dimensions of the 3D image as (height, width, depth). 


## Video Tutorials (for v0.4)
We have made tutorials explaining how to use our software. See links below (videos in Youtube):

[Tutorial 1: Install 3DeeCellTracker and train the 3D U-Net](https://www.youtube.com/watch?v=ctt6o3DY2bA)

[Tutorial 2: Tracking cells by 3DeeCellTracker](https://www.youtube.com/watch?v=KZ03Y8u8UK0)

[Tutorial 3: Annotate cells for training 3D U-Net](https://www.youtube.com/watch?v=ONSOLJQaq28)

[Tutorial 4: Manually correct the cell segmentation](https://www.youtube.com/watch?v=e7xWaccH63o)

## A Text Tutorial (for v0.4)
We have written a tutorial explaining how to install and use 3DeeCellTracker. See [Bio-protocol, 2022](https://bio-protocol.org/e4319)

## How it works
We designed this pipeline for segmenting and tracking cells in 3D + T images in deforming organs. The methods have been explained in [Wen et al. bioRxiv 2018]( https://doi.org/10.1101/385567) and in [Wen et al. eLife, 2021](https://elifesciences.org/articles/59187).

**Overall procedures of our method** ([Wen et al. eLife, 2021–Figure 1](https://elifesciences.org/articles/59187/figures#content))

<img src="https://iiif.elifesciences.org/lax:59187%2Felife-59187-fig1-v1.tif/full/1500,/0/default.jpg" width="400">

**Examples of tracking results** ([Wen et al. eLife, 2021–Videos](https://elifesciences.org/articles/59187/figures#content))

| [Neurons in a ‘straightened’ <br />freely moving worm](https://static-movie-usa.glencoesoftware.com/mp4/10.7554/5/4ce9eaa4a84bf7847c99c81a13ccafd797b40218/elife-59187-fig6-video1.mp4)| [Cardiac cells in a zebrafish larva](https://static-movie-usa.glencoesoftware.com/mp4/10.7554/5/4ce9eaa4a84bf7847c99c81a13ccafd797b40218/elife-59187-fig7-video2.mp4) | [Cells in a 3D tumor spheriod](https://static-movie-usa.glencoesoftware.com/mp4/10.7554/5/4ce9eaa4a84bf7847c99c81a13ccafd797b40218/elife-59187-fig8-video2.mp4) |
| ------------- | ------------- | ------------- | 
| <img src="https://user-images.githubusercontent.com/27986173/115169952-63b4e600-a0fa-11eb-9b85-91292bc9d419.gif" width="340">| <img src="https://user-images.githubusercontent.com/27986173/115170418-90b5c880-a0fb-11eb-9382-13690c3375dc.gif" width="400">| <img src="https://user-images.githubusercontent.com/27986173/115170434-9ad7c700-a0fb-11eb-9004-2e4cff86f7ab.gif" width="200">|


## Citation

If you used this package in your research and is interested in citing it here's how you do it:

```
@article{
author = {Wen, Chentao and Miura, Takuya and Voleti, Venkatakaushik and Yamaguchi, Kazushi and Tsutsumi, Motosuke and Yamamoto, Kei and Otomo, Kohei and Fujie, Yukako and Teramoto, Takayuki and Ishihara, Takeshi and Aoki, Kazuhiro and Nemoto, Tomomi and Hillman, Elizabeth MC and Kimura, Koutarou D},
doi = {10.7554/eLife.59187},
journal = {eLife},
month = {mar},
title = {{3DeeCellTracker, a deep learning-based pipeline for segmenting and tracking cells in 3D time lapse images}},
volume = {10},
year = {2021}
}
```

## Acknowledgements
We wish to thank **JetBrains** for supporting this project 
with free open source **Pycharm** license.

[![Pycharm Logo](pictures/jetbrains_small.png)](https://www.jetbrains.com/) 
[![Pycharm Logo](pictures/icon-pycharm_small.png)](https://www.jetbrains.com/pycharm/)
