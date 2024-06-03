# 3DeeCellTracker
[![PyPI](https://img.shields.io/pypi/v/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/3DeeCellTracker)](https://pypi.org/project/3DeeCellTracker/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)
[![Youtube](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=ctt6o3DY2bA&list=PLGY0oNQomrHERP08iEj-MsluFW8xQJujP)

**3DeeCellTracker** is a deep-learning based pipeline for tracking cells in 3D time-lapse images of deforming/moving organs ([eLife, 2021](https://elifesciences.org/articles/59187)).

## Installation

To install 3DeeCellTracker, please follow the instructions below:

> Note: We have tested the installation and the tracking programs in two environments:
> 1. (Local) Ubuntu 20.04; NVIDIA GeForce RTX 3080Ti; Tensorflow 2.5.0
> 2. (Google Colab) Tensorflow 2.12.0 (You need to upload your data for tracking)

### Prerequisites
- A computer with an NVIDIA GPU that supports CUDA.
- [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://conda.io/miniconda.html) installed.
- TensorFlow 2.x installed.

### Steps
1. Create a new conda environment and activate it by running the following commands in your terminal:

   ```console
   $ conda create -n track python=3.8 pip
   $ conda activate track
   ```

2. Install TensorFlow 2.x by following the instructions provided in the [TensorFlow installation guide](https://www.tensorflow.org/install).

3. Install the 3DeeCellTracker package by running the following command in your terminal:

   ```console
   $ pip install 3DeeCellTracker==0.5.2a0
   ```
4. Once the installation is complete, you can start using 3DeeCellTracker for your 3D cell tracking tasks within the Jupyter notebooks provided in the GitHub repository.

If you encounter any issues or have any questions, please refer to the project's documentation or raise an issue in the GitHub repository.

## Reporting Issues
If you encounter any issues or have suggestions for improvements, please let me know by creating an issue in this repository. To report an issue, follow these steps:

1. Go to the [Issues tab](https://github.com/WenChentao/3DeeCellTracker/issues) on this repository.
2. Click on the **New Issue** button.
3. Provide a descriptive title and a clear description of the issue or suggestion.
4. Include any relevant information, such as error messages, steps to reproduce the issue, or screenshots.
5. Apply appropriate labels or tags to categorize the issue.
6. Click on the **Submit New Issue** button.

Thank you for helping us improve this project!

## Citation

If you used this package in your research, please cite our paper:

- Chentao Wen, Takuya Miura, Venkatakaushik Voleti, Kazushi Yamaguchi, Motosuke Tsutsumi, Kei Yamamoto, Kohei Otomo, Yukako Fujie, Takayuki Teramoto, Takeshi Ishihara, Kazuhiro Aoki, Tomomi Nemoto, Elizabeth MC Hillman, Koutarou D Kimura (2021) 3DeeCellTracker, a deep learning-based pipeline for segmenting and tracking cells in 3D time lapse images eLife 10:e59187

Depending on the segmentation method you used (StarDist3D or U-Net3D), you may also cite either of 
following papers:
- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.
Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020

- Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In: Ourselin, S., Joskowicz, L., Sabuncu, M., Unal, G., Wells, W. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016. MICCAI 2016. Lecture Notes in Computer Science(), vol 9901. Springer, Cham.

## Acknowledgements
We wish to thank **JetBrains** for supporting this project 
with free open source **Pycharm** license.

[![Pycharm Logo](pictures/jetbrains_small.png)](https://www.jetbrains.com/) 
[![Pycharm Logo](pictures/icon-pycharm_small.png)](https://www.jetbrains.com/pycharm/)
