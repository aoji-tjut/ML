import numpy as np
import cv2 as cv

src1 = cv.imread("./image/linux.JPG")
src2 = cv.imread("./image/windows.JPG")
cv.namedWindow("src1")
cv.imshow("src1", src1)
cv.moveWindow("src1", 0, 0)
cv.namedWindow("src2")
cv.imshow("src2", src2)
cv.moveWindow("src2", 0, 0)

add = cv.add(src1, src2)  # 加
subtract = cv.subtract(src1, src2)  # 减
multiply = cv.multiply(src1, src2)  # 乘
divide = cv.divide(src1, src2)  # 除

cv.imshow("add", add)
cv.imshow("subtract", subtract)
cv.imshow("multiply", multiply)
cv.imshow("divide", divide)

bit_and = cv.bitwise_and(src1, src2)  # 与
bit_or = cv.bitwise_or(src1, src2)  # 或
bit_not = cv.bitwise_not(src2)  # 非
bit_xor = cv.bitwise_xor(src1, src2)  # 异或
cv.imshow("bit_and", bit_and)
cv.imshow("bit_or", bit_or)
cv.imshow("bit_not", bit_not)
cv.imshow("bit_xor", bit_xor)

clip = np.uint8(np.clip((1.5 * src2 + 10), 0, 255))  # 对比度 亮度
cv.imshow("clip", clip)

add_weight = cv.addWeighted(src1, 0.5, src2, 0.5, 100)  # 所有像素+100
cv.imshow("add_weight", add_weight)

cv.waitKey(0)
