import numpy as np
import cv2 as cv

src = cv.imread("./image/乱糟糟.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
blur = cv.GaussianBlur(src, (3, 3), 0)
canny = cv.Canny(blur, 50, 100)
image, contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
dst = np.zeros(src.shape, np.uint8)
np.copyto(dst, src)
for i, contour in enumerate(contours):
    area = cv.contourArea(contour)  # 面积
    rect_x, rect_y, rect_w, rect_h = cv.boundingRect(contour)  # 外接矩形
    mm = cv.moments(contour)  # 重心
    center_x = mm['m10'] / mm['m00']
    center_y = mm['m01'] / mm['m00']
    curve = cv.approxPolyDP(contour, 4, True)
    shape = ["rectangle", "circle", "polygon"]
    k = 1
    if curve.shape[0] == 4:
        k = 0
    elif 18 == curve.shape[0]:
        k = 2
    elif curve.shape[0] > 5:
        pass
    print(curve.shape[0])
    cv.putText(dst, "contour=" + str(i + 1), (rect_x, rect_y + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(dst, "area=" + str(area), (rect_x, rect_y + 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(dst, "shape=" + str(shape[k]), (rect_x, rect_y + 45),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
    cv.circle(dst, (int(center_x), int(center_y)), 2, (0, 0, 0), -1, cv.LINE_AA)
    cv.rectangle(dst, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h),
                 (255, 255, 255), 1, cv.LINE_AA)
cv.imshow("dst", dst)
cv.waitKey(0)
