ScratchDetection.py script to detect scratches using OpenCV operations

test.py Deep Image Processing with torch CNN's,
made following this tutorial -> https://www.youtube.com/watch?v=FPzi8cUhNNY
Computationally expensive to recreate the whole image from scratch, so better to use inpainting

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
  - `t` = use tophat filtering to find defects and then only keep those with an area smaller than a certain threshold, usefull for small noise
  - `m` = multiscale detection, good for general purpose mask generation
```


