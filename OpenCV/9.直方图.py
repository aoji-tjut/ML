import cv2 as cv
import matplotlib.pyplot as plt

src = cv.imread("./image/girl.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
row, col, channel = src.shape[:]
color = ("blue", "green", "red")
for i, color in enumerate(color):
    hist = cv.calcHist([src], [i], None, [256], [0, 256])
    plt.plot(hist, color=(color))
    plt.xlim([0, 256])  # 设置x轴显示范围
plt.show()
cv.waitKey(0)
