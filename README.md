# 3DeeCellTracker
[![PyPI](https://img.shields.io/pypi/v/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

**3DeeCellTracker** is a deep-learning based pipeline for tracking cells in 3D time lapse images of deforming/moving organs ([eLife, 2021](https://elifesciences.org/articles/59187)).

## Updates:
### 2021.06.17
We have updated our program to 3DeeCellTracker v0.4.0:
- We modified the notebooks, and the underlying package to simplify the use.
- The intermediate results of segmentation and tracking can be visualized easily to assist the parameter optimization. 

### 2021.03.29
We have updated our program to 3DeeCellTracker v0.3:
- By using vectorization, we remarkably reduced the runtime for tracking cells.

## Installation

* Create a conda environment for a PC with GPU including prerequisite packages using the 3DCT.yml file:

```console
$ conda env create -f 3DCT.yml
```

* (NOT RECOMMEND) Users can create a conda environment for a PC with only CPU, but it will be slow and may fail.
```console
$ conda env create -f 3DCT-CPU.yml
```

* Install the 3DeeCellTracker package solely by pip

```console
$ pip install 3DeeCellTracker
```

For detailed instructions, see [here](Doc/Enviroment.md).
## Quick Start
To learn how to track cells use 3DeeCellTracker, see following notebooks for examples:
1. Track cells in deforming organs: 
    - [**Single mode (clear notebook)**](Examples/single_mode_worm1-clear.ipynb);
    - [**single mode (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/single_mode_worm1.html)


2. Track cells in freely moving animals: 
    - [**Ensemble mode (clear notebook)**](Examples/ensemble_mode_worm4-clear.ipynb)
    - [**Ensemble mode (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/ensemble_mode_worm4.html)


3. Train a new 3D U-Net for segmenting cells in new optical conditions: 
    - [**Train 3D U-Net (clear notebook)**](Examples/3D_U_Net_training-clear.ipynb).
    - [**Train 3D U-Net (results)**](https://wenchentao.github.io//3DeeCellTracker/Examples/3D_U_Net_training.html).
   
The data and model files for demonstrating above notebooks can be downloaded [**here**](https://osf.io/dt76c/).

**Note**: Codes above were based on the latest version. 
For old programs used in eLife 2021, please check the "[**Deprecated_programs**](Deprecated_programs)" folder.

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
We wish to thank **JetBrainsfor** for supporting this project 
with free open source **Pycharm** license.

[![Pycharm Logo](pictures/jetbrains_small.png)](https://www.jetbrains.com/) 
[![Pycharm Logo](pictures/icon-pycharm_small.png)](https://www.jetbrains.com/pycharm/)