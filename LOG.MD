### 2023.04.24
3DeeCellTracker v0.5.1-alpha was released with following issues fixed
- updated versions of some packages in the requirements.txt and setup.cfg

### 2023.04.23
3DeeCellTracker v0.5.0-alpha was released with following new features
- Provides a wrapper allows you to use [StarDist3D](https://github.com/stardist/stardist) for segmentation
- Reduces the requirements for fine-tuning parameters
- Decouples the code to facilitate reuse by third-party developers.

### 2021.12.02
3DeeCellTracker v0.4.1 was released with following issues fixed
- Solved the bug "the directions of arrows showing the accurate correction are opposite"
- Added functions for extracting/drawing activities from cell images based on tracked labels
- Added the function for storing coordinates of tracked cells
- Added support for both tensorflow 1.x and 2.x 

### 2021.06.17
We have updated our program to 3DeeCellTracker v0.4.0:
- We modified the notebooks, and the underlying package to simplify the use.
- The intermediate results of segmentation and tracking can be visualized easily to assist the parameter optimization. 

### 2021.03.29
We have updated our program to 3DeeCellTracker v0.3:
- By using vectorization, we remarkably reduced the runtime for tracking cells.