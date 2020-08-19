import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret_otsu, bin_otsu = cv.threshold(gray, 0, 255,
                                  cv.THRESH_BINARY | cv.THRESH_OTSU)
ret_triangle, bin_triangle = cv.threshold(gray, 0, 255,
                                          cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
bin_adapt_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                      cv.THRESH_BINARY, 25, 10)
bin_adapt_gau = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY, 25, 10)
print("ret_otsu value = " + str(ret_otsu))
print("ret_triangle value = " + str(ret_triangle))
cv.imshow("bin_otsu", bin_otsu)
cv.imshow("bin_triangle", bin_triangle)
cv.imshow("bin_adapt_mean", bin_adapt_mean)
cv.imshow("bin_adapt_gau", bin_adapt_gau)
cv.waitKey(0)
