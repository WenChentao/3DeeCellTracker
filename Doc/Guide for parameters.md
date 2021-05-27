# Guide for setting parameters in 3DeeCellTracker

3DeeCellTracker has some parameters which need to be properly set by end-users. This guide aims to explain how to set these parameters.

## General flow:
(1) The five parameters for segmentation/tracking are decided in a trial-and-error way:
```
# parameters manually determined by experience
noise_level = 200 # a threshold to discriminate noise/artifacts from cells 
min_size = 40 # a threshold to remove small objects which may be noise/artifacts  
BETA = 50 # control coherence using a weighted average of movements of nearby points; 
                   # larger BETA includes more points, thus generates more coherent movements
LAMBDA = 0.1 # control coherence by adding a loss of incoherence; large LAMBDA 
             # generates larger penalty for incoherence.
max_iteration = 50 # maximum number of iterations; large values generate more accurate tracking.
```
The “noise_level” and “min_size” should be set according to the segmentation result. Users should first run the codes from the start until the beginning of the following codes, then run the following codes every time after modifying the “noise_level” or “min_size”.
```
# segment 3D image of volume #1
segmentation(1)
# save the segmented cells of volume #1
for z in range(1, layer_num+1):
   auto_segmentation = (segmentation_auto[:, :, z-1]).astype(np.uint8)
Image.fromarray(auto_segmentation).save(auto_segmentation_vol1_path+"t%03i_z%03i.tif"%(1,z))
```

The “BETA”, “LAMBDA”, and “max_iteration” should be set according to the result of PR-GLS. Users should first run codes from the start until the beginning of the following codes. Then run the following codes every time after changing these three parameters.
```
test_tracking(96,97) # choose two neighboring time points with challenging (large) movements.
```
For detailed instructions for each parameter, see below.

(2) There are other parameters which do not need a trial-and-error procedure. They can be decided according to imaging conditions or user’s preference.
```
# parameters according to imaging conditions
volume_num = 1000 # number of volumes in the 3D + T image
x_siz, y_siz, z_siz = 260, 180, 165 # size of each 3D image
z_xy_resolution_ratio = 2 # the resolution ratio between the z axis and the x-y plane (does not need to be very accurate)
z_scaling = 1 # (integer) for interpolating images along z. z_scaling = 1 makes no interpolation.
                       # z_scaling > 1 generates smoother images.
shrink = (24,24,24) # pad and shrink for u-net prediction, corresponding to (x,y,z). Large values   
     #lead to more accurate segmentations, but it should be less than (input sizes of u-net)/2
```
## Detailed instructions
### Parameter: Noise level
- When to set it?
  - Set it for every new dataset.
- Why to set it?
  - To let the segmentation program detect target cells while ignoring noise/artifacts.
- How to set it?
  1.	Try a value and run the segmentation part of the codes. 
  2.	Check segmentation result in folder “automatic_vol1”.
  3.	Decrease the value if there are any target cells lost. Or increase the value if series noise/artifacts are detected as cells. 
  
  \* The segmentation result does not need to be perfect. A few errors (e.g. 10% or even more) will be acceptable.
- Examples:
<table>
    <thead>
        <tr>
            <th>Raw image</th>
            <th colspan="3">Automatic segmentation</th>
        </tr>
        <tr>
            <th>(zebrafish: z=88) </th>
            <th>Noise level: 50 (artifacts)</th>
            <th>Noise level: 200 (good)</th>
            <th>Noise level: 500 (cells lost)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="/pictures/zebrafish_raw_z88.png"></td>
            <td><img src="/pictures/zebrafish_noiselevel_50_z88.png"></td>
            <td><img src="/pictures/zebrafish_noiselevel_200_z88.png"></td>
            <td><img src="/pictures/zebrafish_noiselevel_500_z88.png"></td>
        </tr>
    </tbody>
</table>

### Parameter: Minimum size of cells
- When to set it?
  - Only when optical setups or cell properties changed. (e.g. resolution or cell sizes are obviously different)
- Why to set it?
  - To remove cell-like noise/artifacts of small size.
- How to set it?
  1.	Try a value and run the segmentation part of the codes. 
  2.	Check segmentation result in folder “automatic_vol1”.
  3.	Decrease the value if real cells are incorrectly removed. Or increase the value if non-cell signals are detected as cells.
  
  \* Again the segmentation result does not need to be perfect.
- Examples:
<table>
    <thead>
        <tr>
            <th>Raw image</th>
            <th colspan="3">Automatic segmentation</th>
        </tr>
        <tr>
            <th>(zebrafish: z=88) </th>
            <th>Minsize: 10 (artifacts)</th>
            <th>Minsize: 40 (good)</th>
            <th>Minsize: 200 (cells removed)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="/pictures/zebrafish_raw_z88.png"></td>
            <td><img src="/pictures/auto_vol1_minsize_10_z88.png"></td>
            <td><img src="/pictures/zebrafish_noiselevel_200_z88.png"></td>
            <td><img src="/pictures/auto_vol1_minsize_200_z88.png"></td>
        </tr>
    </tbody>
</table>
             
### Parameter: Coherence level β
- When to set it?
  - Only when optical setups or cell properties changed. (e.g. resolution or coherence level are obviously different)
- Why to set it?
  - To set the level of coherence by smoothing cell movements in a small (small β) or wide (large β) region.
- How to set it?
  1.	Try a value and test the tracking by PR-GLS in two neighboring volumes which are challenging.  
  2.	If the cell movements are too coherent (usually with tiny or similar movements), reduce β; If the cell movements are too incoherent (usually with obvious mistakes), increase β.
- Examples:
<table>
    <thead>
        <tr>
            <th>Superimposed image</th>
            <th colspan="3">Registration by PR-GLS</th>
        </tr>
        <tr>
            <th>(t = 96 & t = 97)</th>
            <th>β: 10 (mistakes)</th>
            <th>β: 50 (good)</th>
            <th>β: 500 (tiny movements)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="/pictures/zebrafish_raw_t9697.png"></td>
            <td><img src="/pictures/Beta10_xy.svg"></td>
            <td><img src="/pictures/Beta50_xy.svg"></td>
            <td><img src="/pictures/Beta500_xy.svg"></td>
        </tr>
    </tbody>
</table>
                     
### Parameter: Coherence level λ
- When to set it?
  - Only when optical setups or cell properties changed. (e.g. resolution or coherence level are obviously different)
- Why to set it?
  - To set the level of coherence by imposing a small or large penalty for incoherent movements.
  
  \* Users could simply change β and keep λ unchanged, which worked in our worms/zebrafish datasets.
- How to set it?
  - 1.	Try a value and test the tracking by PR-GLS in two neighboring volumes which are challenging.  
  - 2.	If the cell movements are too coherent (usually with tiny or similar movements), reduce λ; If the cell movements are too incoherent (usually with obvious mistakes), increase λ.
- Examples:
<table>
    <thead>
        <tr>
            <th>Superimposed image</th>
            <th colspan="3">Registration by PR-GLS</th>
        </tr>
        <tr>
            <th>(t = 96 & t = 97)</th>
            <th>λ: 0.01 (mistakes)</th>
            <th>λ: 0.1 (good)</th>
            <th>λ: 2 (tiny movements)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="/pictures/zebrafish_raw_t9697.png"></td>
            <td><img src="/pictures/Lambda_0.01_xy.svg"></td>
            <td><img src="/pictures/Beta50_xy.svg"></td>
            <td><img src="/pictures/Lambda_2_xy.svg"></td>
        </tr>
    </tbody>
</table>
                     
### Parameter: Maximum iteration
- When to set it?
  - Only when movement properties changed. (e.g. coherency level is too low)
- Why to set it?
  - To improve the accuracy of registration by iteratively apply the PR-GLS method.
- How to set it?
  1.	Try a value and test the tracking by PR-GLS in two neighboring volumes which are challenging.  
  2.	If the registration between the two volumes is poor in most cells, increase the number.
  
  \* When iteration is large enough, further increasing it will not improve registration much while increasing the runtime.
- Examples:
<table>
    <thead>
        <tr>
            <th>Superimposed image</th>
            <th colspan="3">Registration by PR-GLS</th>
        </tr>
        <tr>
            <th>(t = 96 & t = 97)</th>
            <th>iteration: 5 (poor)</th>
            <th>iteration: 50 (enough)</th>
            <th>iteration: 200 (not necessary)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="/pictures/zebrafish_raw_t9697.png"></td>
            <td><img src="/pictures/iteration_5_xy.svg"></td>
            <td><img src="/pictures/Beta50_xy.svg"></td>
            <td><img src="/pictures/iteration_200_xy.svg"></td>
        </tr>
    </tbody>
</table>
