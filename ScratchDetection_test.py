import cv2
import numpy as np
from scipy.sparse.csgraph import laplacian


def load_and_preprocess(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise IOError("Unable to load image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray

def denoise(gray, method="gaussian"):
    """
    remove noise while preserving edges

    gray: grayscale input image
    method: gaussian,bilateral,median
    """

    if method == "gaussian":
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    elif method == "median":
        denoised = cv2.medianBlur(gray, 5)

    return denoised

def edge_detection(image, method="canny"):

    if method == "canny":
        edges = cv2.Canny(image, 50, 150)
    elif method == "sobel":
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges/edges.max() * 255)
    elif method == "laplacian":
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        edges = np.uint8(np.absolute(laplacian))
    elif method == "scharr":
        # More accurate than sobel
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        edges = np.sqrt(scharrx**2 + scharry**2)
        edges = np.uint8(edges/edges.max()*255)

    return edges

def high_pass(img, blurred):
    """

    :param img: Greyscale image
    :param blurred: image blurred
    :return: subtracted image
    """

    high_pass = cv2.subtract(img, blurred)

    return detect_scratches_adaptive(high_pass)

def morphological_operations(edges, kernel_size=3, operation="close"):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == "close":
        # Close gaps in scratches
        result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    elif operation == "open":
        # Remove small noise
        result = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    elif operation == "dilate":
        result = cv2.dilate(edges, kernel,iterations=1)
    elif operation == "erode":
        result = cv2.erode(edges, kernel,iterations=1)
    elif operation == "tophat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.morphologyEx(edges, cv2.MORPH_TOPHAT, kernel)

    #TODO blackhat operation for pictures with black defects


    return result

def detect_scratches_hough(img, edges, min_line_length=50, max_line_gap=10):
    """
    Args
    :param img: input image
    :param edges: edge detected binary image
    :param min_line_length: minimum line length to detect
    :param max_line_gap: max gap between line segments
    """

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Draw the lines

    result = img.copy()
    scratch_count = 0

    if lines is None:
        return result, scratch_count

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        # Filter by angle and length
        ####

        if length > min_line_length:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            scratch_count += 1

    return result, scratch_count

def detect_scratches_contour(img, edges, min_area = 50, max_area=10000):
    """

    :param img:
    :param edges:
    :param min_area:
    :param max_area:
    :return:
    """

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros_like(img)

    cv2.drawContours(output, contours, -1, (255, 255, 255), -1)

    return output

    # result = img.copy()
    # scratch_contours = []
    #
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #
    #     if min_area < area < max_area:
    #         # Aspect Ratio
    #         x, y, w, h = cv2.boundingRect(contour)
    #         aspect_ratio = max(w,h) / min(w,h)
    #
    #         if aspect_ratio > 3:
    #
    #             scratch_contours.append(contour)
    #
    #             # Draw bounding box
    #             cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #             # Draw contour
    #             cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
    #
    # return result, scratch_contours

def detect_scratches_adaptive(img, block_size=11, C=2):
    """

    :param img: Grayscale image
    :param block_size:
    :param C:
    """

    binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block_size,C)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Close gaps in scratches
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    return result

def multiscale_detections(img):
    """

    :param img: greyscale image
    :return:
    """
    minor_blur = cv2.GaussianBlur(img, (5, 5), 0)
    major_blur = cv2.GaussianBlur(img, (25, 25), 0)

    diff = cv2.absdiff(minor_blur, major_blur)

    return diff



img_path = "OldPhoto_1.jpeg"

img, gray = load_and_preprocess(img_path)

# Denoise image
denoised = denoise(gray, method="gaussian")

# Generate mask
# mask = morphological_operations(gray, kernel_size=9, operation="tophat")
# _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
mask = multiscale_detections(gray)
_, mask = cv2.threshold(mask, 30,255, cv2.THRESH_BINARY)

# Clean and fill in scratches
cleaned = morphological_operations(mask, 3, operation="close")
cleaned = detect_scratches_contour(gray, cleaned)
cleaned = morphological_operations(cleaned, 3, operation="dilate")
cleaned = morphological_operations(cleaned, 3, operation="dilate")

# inpaint with cv
result = cv2.inpaint(img,cleaned,3,cv2.INPAINT_TELEA)

cv2.imshow("Original", img)
cv2.imshow("mask", cleaned)
cv2.imshow("restored", result)

cv2.imwrite("test_result.png", gray)
cv2.imwrite("test_result_mask.png", cleaned)


cv2.waitKey(0)
cv2.destroyAllWindows()