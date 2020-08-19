import numpy as np
import cv2 as cv

corner_value = 30
corner_max = 500


def corner_callback(value):
    corner = cv.getTrackbarPos("corners", "Shi-Thomas")
    if corner <= 1:
        corner = 1
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = np.zeros(src.shape[:], np.uint8)
    np.copyto(dst, src)
    corners = cv.goodFeaturesToTrack(gray, corner, 0.01, 10)
    corners = np.uint(corners)  # 将浮点型数组转换为整型数组
    for i in corners:
        x, y = i.ravel()  # 将二维数组转换为一维数组
        cv.circle(dst, (x, y), 2, (0, 0, 255), -1)
    cv.imshow("Shi-Thomas", dst)


src = cv.imread("./image/house.JPG")
cv.namedWindow("Shi-Thomas")
cv.imshow("Shi-Thomas", src)
cv.moveWindow("Shi-Thomas", 0, 0)
cv.createTrackbar("corners", "Shi-Thomas", corner_value, corner_max, corner_callback)
corner_callback(0)
cv.waitKey(0)
