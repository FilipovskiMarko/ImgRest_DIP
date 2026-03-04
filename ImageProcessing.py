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

if mode == "d":
    result = denoise_bm3d(img, color=True)
elif mode == "s":
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharp_amnt = 1.5
    sub_weight = -0.5
    result = cv2.addWeighted(img,sharp_amnt,blurred,sub_weight,0)
else:
    raise ValueError("Unknown mode")

# inpaint with cv
# result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

cv2.imshow("Original", img)
cv2.imshow("Result", result)
# cv2.imshow("result", result)

# cv2.imwrite("Mask/" + name + "_result.png", result)
# cv2.imwrite(output_path + name + "_mask.png", mask)
# cv2.imwrite("Mask/" + name + ".png", img)


cv2.waitKey(0)
cv2.destroyAllWindows()