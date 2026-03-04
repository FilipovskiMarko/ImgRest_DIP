from functions import *
import argparse
from pathlib import Path
import cv2



# Parse Part
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=Path, required=True, help="Input image")
parser.add_argument("-o", "--output",type=Path, required=True, help="Output image")
parser.add_argument("-m", "--mode", type=str, help="How to generate mask, (l)ines,(t)ophat, (m)ultiscale detection...")

args = parser.parse_args()

input_path = args.input
name = input_path.stem
format = input_path.suffix.lower()
img_path = input_path

output_path = str(args.output) + "/"
mode = str(args.mode).lower()

# Processing Part
img, gray = load_and_preprocess(img_path)

if mode == "l":
    denoised = denoise(gray, method="gaussian")
    mask = morphological_operations(denoised, kernel_size=9,operation="tophat")
    _, mask = cv2.threshold(mask,20,255, cv2.THRESH_BINARY)
    mask, count = detect_scratches_hough(mask)
    mask = morphological_operations(mask,3,operation="dilate")

elif mode == "m":
    mask = multiscale_mask(gray,method="gaussian")

elif mode == "t":
    denoised = denoise(gray,method="gaussian")
    mask = tophat_mask(denoised)
else:
    raise ValueError("Unknown mode")

# inpaint with cv
# result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

# cv2.imshow("Original", img)
# cv2.imshow("mask", mask)
# cv2.imshow("result", result)

# cv2.imwrite("Mask/" + name + "_result.png", result)
cv2.imwrite(output_path + name + "_mask.png", mask)
# cv2.imwrite("Mask/" + name + ".png", img)


# cv2.waitKey(0)
# cv2.destroyAllWindows()