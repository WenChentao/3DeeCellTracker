### Tracking Using IDE (Spyder):
#### 1 Preparations: 
We suggest that users should use an IDE such as Spyder to run the "cell_segment_track.py" under "./Tracking/".

1. Modify following path and file names in "cell_segment_track.py", including: 
- "folder_path" (containing data, models and segmetation/tracking results),  
- "files_name"(of raw images), 
- "unet_weight_file" (name of 3D U-net weight file)
- "FFN_weight_file" (name of FFN weight file)
2. Put raw images into "data" folder and unet and FFN weight files into "models" folder.
3. Modify global parameters (see the [user-guide for setting parameters](https://github.com/WenChentao/3DeeCellTracker/blob/master/Guide%20for%20parameters.md)).
#### 2 Procedures for tracking:
##### 2.1 Segmentation of volume #1
Run the code in "cell_segment_track.py" until finishing "automatic segmentation of volume #1". The resulted segmentation is stored into the folder “auto_vol1”

| Raw image | Segmentation result |
| ------------- | ------------- | 
| ![raw-worm1](/pictures/raw-worm1.png) | ![autoseg-worm1](/pictures/autoseg-worm1.png) |

 (Optional) Users can check the segmentation in Fiji. Here are the 2D projected raw images in volume #1 (left, color = “fire”; images 1-21) and segmentation results (right, color = “3-3-2 RGB”): 
 
##### 2.2 Manually correct the segmentation in volume #1 
Users should correct the segmentation in other software such as ITK-SNAP. For the demo data, we have included the corrected segmentation in folder “manual_vol1”. Here is the 2D projection of our corrected segmentation:
<img src="/pictures/manualseg-worm1.png" width="400">
##### 2.3 Track cells in all volumes
Run "cell_segment_track.py" to the end. The tracked labels are stored into the folder “track_results”. 

<img src="/pictures/track-worm1.gif" width="400">

(Optional) Users can check the tracking results in Fiji by comparing the raw images and tracked labels:

Users can use other software for checking the results, such as in IMARIS.
