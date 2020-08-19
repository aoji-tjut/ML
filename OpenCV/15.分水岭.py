import numpy as np
import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, bin = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
open = cv.morphologyEx(bin, cv.MORPH_OPEN, kernel)
back = cv.dilate(open, kernel)  # 背景
cv.imshow("back", back)
dist = cv.distanceTransform(bin, cv.DIST_L2, 3)
dist = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
ret, front = cv.threshold(dist, dist.max() * 0.1, 255, cv.THRESH_BINARY)  # 前景
cv.imshow("front", front)
front = np.uint8(front)
unknow = cv.subtract(back, front)  # 前景背景重合区域
cv.imshow("unknow", unknow)
ret, markers = cv.connectedComponents(front, connectivity=8)
markers += 1  # 使标注大于1
markers[unknow == 255] = 0  # 使255的像素点变为0 去掉背景
markers = cv.watershed(src, markers)  # 分水岭后所有标注变为-1
src[markers == -1] = [255, 255, 255]
cv.imshow("water", src)
cv.waitKey(0)
