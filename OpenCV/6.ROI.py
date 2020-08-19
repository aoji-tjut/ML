import cv2 as cv

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
cv.imwrite("/Users/aoji/Pictures/opencv/write.png", src, [cv.IMWRITE_PNG_COMPRESSION, 0])
row, col, channel = src.shape[:]
print("row：", row, "col：", col, "channel：", channel)
face = src[200:380, 200:380]  # src[row,col]
cv.imshow("face", face)
face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
face = cv.cvtColor(face, cv.COLOR_GRAY2BGR)  # BGR图像之间才能合并
src[200:380, 200:380] = face
cv.imshow("dst", src)
cv.waitKey(0)
