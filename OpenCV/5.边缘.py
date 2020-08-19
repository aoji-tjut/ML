import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 0)

canny = cv.Canny(blur, 50, 100)

sobel_x = cv.Sobel(blur, cv.CV_32F, 1, 0)
sobel_y = cv.Sobel(blur, cv.CV_32F, 0, 1)
sobel_xy = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel_x = cv.convertScaleAbs(sobel_x)
sobel_y = cv.convertScaleAbs(sobel_y)
sobel_xy = cv.convertScaleAbs(sobel_xy)

scharr_x = cv.Scharr(blur, -1, 1, 0)
scharr_y = cv.Scharr(blur, -1, 0, 1)
scharr_xy = cv.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
scharr_x = cv.convertScaleAbs(scharr_x)
scharr_y = cv.convertScaleAbs(scharr_y)
scharr_xy = cv.convertScaleAbs(scharr_xy)

laplacian = cv.Laplacian(blur, cv.CV_32F)
laplacian = cv.convertScaleAbs(laplacian)

cv.imshow("canny", canny)
cv.imshow("sobel_x", sobel_x)
cv.imshow("sobel_y", sobel_y)
cv.imshow("sobel_xy", sobel_xy)
cv.imshow("scharr_x", scharr_x)
cv.imshow("scharr_y", scharr_y)
cv.imshow("scharr_xy", scharr_xy)
cv.imshow("laplacian", laplacian)
cv.waitKey(0)
