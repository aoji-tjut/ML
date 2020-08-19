import numpy as np
import cv2 as cv

src = cv.imread("./image/board.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
blur = cv.GaussianBlur(src, (7, 7), 0)
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                          param1=268, param2=21, minRadius=5, maxRadius=30)
dst = np.zeros((row, col, channel), np.uint8)
np.copyto(dst, src)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv.circle(dst, (i[0], i[1]), i[2], (0, 0, 255), 1, cv.LINE_AA)
    cv.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 1, cv.LINE_AA)
# for i in range(len(circles[0])):
#     center = (circles[0][i][0],circles[0][i][1])
#     r = circles[0][i][2]
#     cv.circle(dst, center, r, (0, 0, 255), 1, cv.LINE_AA)
#     cv.circle(dst, center, 1, (0, 0, 255), 1, cv.LINE_AA)
cv.imshow("dst", dst)
print(circles)
cv.waitKey(0)
