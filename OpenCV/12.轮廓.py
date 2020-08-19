import numpy as np
import cv2 as cv

src = cv.imread("./image/枫叶.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
blur = cv.GaussianBlur(src, (5, 5), 0)
canny = cv.Canny(blur, 100, 200)
image, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
dst = np.zeros(src.shape, np.uint8)
np.copyto(dst, src)
for i, contour in enumerate(contours):
    cv.drawContours(dst, contours, i, (0, 255, 0), 2, cv.LINE_AA)
cv.imshow("dst", dst)
cv.waitKey(0)
