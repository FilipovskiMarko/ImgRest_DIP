ScratchDetection.py script to detect scratches using OpenCV operations

test.py Deep Image Processing with torch CNN's,
made following this tutorial -> https://www.youtube.com/watch?v=FPzi8cUhNNY
Computationally expensive to recreate the whole image from scratch, so better to use inpainting

- Usage
```bash

python ScratchDetection_test.py -i [PATH-TO-IMAGE] -o [OUTPUT-FOLDER] -m [MODE]


```
- Arguments:
```md

- `-i, --input`   Input image path
- `-o, --output`  Output image path
- `-m, --mode`    Processing mode:
  - `l` = lines
  - `t` = tophat
  - `m` = multiscale detection
```


