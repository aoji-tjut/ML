import numpy as np
import cv2 as cv

src = cv.imread("./image/building.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
canny = cv.Canny(src, 75, 150)
cv.imshow("canny", canny)
lines = cv.HoughLinesP(canny, 1.0, np.pi / 180, 150,
                       minLineLength=20, maxLineGap=10)
dst = np.zeros((row, col, channel), np.uint8)
np.copyto(dst, src)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1, cv.LINE_AA)
cv.imshow("dst", dst)
print(lines)
cv.waitKey(0)
