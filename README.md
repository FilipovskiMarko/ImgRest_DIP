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



test.py Deep Image Processing with torch CNN's,
made following this tutorial -> https://www.youtube.com/watch?v=FPzi8cUhNNY
Computationally expensive to recreate the whole image from scratch, so better to use inpainting

