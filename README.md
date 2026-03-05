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
<img width="1320" height="1604" alt="OldPhoto_2" src="https://github.com/user-attachments/assets/9121928d-e372-40fe-809d-ca4d3bde8671" />



Mask (using multiscale detections):
<img width="1320" height="1604" alt="OldPhoto_2_mask" src="https://github.com/user-attachments/assets/70abb7b3-d84a-48eb-8c0f-e3e7db8e7db5" />


LaMa:
<img width="1320" height="1604" alt="OldPhoto_2_lama" src="https://github.com/user-attachments/assets/180570e1-fdeb-46a8-878d-7029cd4c73e9" />



CodeFormer:
![Old_Photo_2_codeformer](https://github.com/user-attachments/assets/cd341623-c742-47e2-acac-6dc875c1e95a)

DDColor:
![Old_Photo_2_ddcolor](https://github.com/user-attachments/assets/ae596b43-a9ca-4ab2-9bfe-ae5f147cc340)





Original Image:
<img width="1230" height="876" alt="9" src="https://github.com/user-attachments/assets/98ff1de1-0767-49de-b30f-b5b88549e4b4" />

Mask (TopHat):
<img width="1230" height="876" alt="9_mask" src="https://github.com/user-attachments/assets/213ac827-04de-435f-8c52-453d21cc1d3b" />


LaMa:
<img width="1230" height="876" alt="9_lama" src="https://github.com/user-attachments/assets/4954ed48-ad93-43b1-8a28-b6cbeddf5961" />


CodeFormer:
<img width="2460" height="1752" alt="9_codeformer" src="https://github.com/user-attachments/assets/a00fddbd-db92-4a7d-a69c-31b81d01e04d" />


DDColor:
<img width="2460" height="1752" alt="9_ddcolor" src="https://github.com/user-attachments/assets/4b6a3235-68d4-4c3d-8efe-2a6bbe551775" />













