import numpy as np
import cv2 as cv

src = cv.imread("./image/乱糟糟.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
hls = cv.cvtColor(src, cv.COLOR_BGR2HLS)
ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
cv.imshow("gray", gray)
cv.imshow("hsv", hsv)
cv.imshow("hls", hls)
cv.imshow("ycrcb", ycrcb)
cv.imshow("yuv", yuv)

hsv_lower = np.array([0, 43, 46])
hsv_upper = np.array([10, 255, 255])
hsv_red = cv.inRange(hsv, hsv_lower, hsv_upper)
cv.imshow("mask", hsv_red)

b, g, r = cv.split(src)
cv.imshow("b", b)
cv.imshow("g", g)
cv.imshow("r", r)

src[:, :, 0] = 0  # 把第一通道像素设为0
cv.imshow("src[:,:,0] = 0", src)
cv.waitKey(0)
