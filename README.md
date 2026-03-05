A collective of python tools for image sharpening, denoising and mask generation, created with OpenCV and BM3D, used in pair with LaMa for advanced inpainting, CodeFormer for face restoration and DDColor for color restoration

The goal of this project was to use free tools to restore old photographs that have faded colors/physical damage

ScratchDetection.py script to detect scratches using OpenCV operations

- Usage:
```bash

python ScratchDetection.py -i [PATH-TO-IMAGE] -o [OUTPUT-FOLDER] -m [MODE]

```
- Arguments:
```md

- `-i, --input`   Input image path
- `-o, --output`  Output image path
- `-m, --mode`    Processing mode:
  - `l` = find straight lines using hough line transform, useful for images with linear defects
  - `t` = tophat filtering to find defects and keep those with an area smaller than a threshold, usefull for small noise
  - `m` = multiscale detection, good for general purpose mask generation
```



ImageProcessing.py script for image processing using OpenCV and BM3D operations

- Usage:
```bash

python ImageProcessing.py -i [PATH-TO-IMAGE] -o [OUTPUT-FOLDER] -m [MODE]

```
- Arguments:
```md

- `-i, --input`   Input image path
- `-o, --output`  Output image path
- `-m, --mode`    Processing mode:
  - `d` = denoise, uses bm3d to denoise an image
  - `s` = applies sharpening to the image 
```

Results:
(All images are sourced from r/TheWayWeWere)
Original Image:
<img width="1895" height="2272" alt="OldPhoto_1" src="https://github.com/user-attachments/assets/784dcf51-0496-4538-bb10-94d770af09a0" />

Mask (using multiscale detections):
<img width="1895" height="2272" alt="OldPhoto_1_mask" src="https://github.com/user-attachments/assets/d8d31da9-45a2-4aab-916c-1c42f33b58fc" />



