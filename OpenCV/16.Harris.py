import cv2 as cv

src = cv.imread("./image/house.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
harris = cv.cornerHarris(gray, 2, 3, 0.04)
harris = cv.dilate(harris, None)  # 使检测的角点膨胀
src[harris > 0.2 * harris.max()] = [0, 0, 255]  # 0.2为阈值
cv.imshow("harris", src)
cv.waitKey(0)
