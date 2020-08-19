import numpy as np
import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]

dst1 = cv.resize(src, (int(col / 2), int(row / 2)))  # Size(width,height)
cv.imshow("resize", dst1)

M = np.float32([[1, 0, -200], [0, 1, 100]])  # row移动-200 col移动100
dst2 = cv.warpAffine(src, M, (col, row))
cv.imshow("move", dst2)

M_src = np.float32([[0, 0], [0, row - 1], [col - 1, 0]])  # 左上 左下 右上
M_dst = np.float32([[30, 30], [100, 300], [400, 100]])  # 新坐标
M = cv.getAffineTransform(M_src, M_dst)  # 获得新矩阵
dst3 = cv.warpAffine(src, M, (col, row))
cv.imshow("stretch", dst3)

M = cv.getRotationMatrix2D((col / 2, row / 2), 90, 0.5)  # 中心点(x,y) 角度 缩放
dst4 = cv.warpAffine(src, M, (col, row))
cv.imshow("rotate", dst4)

cv.waitKey(0)
