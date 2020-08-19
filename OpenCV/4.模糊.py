import numpy as np
import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
blur = cv.blur(src, (5, 5))  # 均值模糊
blur_med = cv.medianBlur(src, 5)  # 中值模糊 椒盐噪声
blur_gau = cv.GaussianBlur(src, (5, 5), 0)  # 高斯模糊
blur_bil = cv.bilateralFilter(src, 0, 100, 15)  # 双边模糊 美颜

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
enhance = cv.filter2D(src, -1, kernel)

cv.imshow("blur", blur)
cv.imshow("blur_med", blur_med)
cv.imshow("blur_gau", blur_gau)
cv.imshow("blur_bil", blur_bil)
cv.imshow("enhance", enhance)
cv.waitKey(0)
