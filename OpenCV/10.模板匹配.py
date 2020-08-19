import cv2 as cv

src = cv.imread("./image/海绵宝宝.JPG")
temp = cv.imread("./image/模板.JPG")
cv.namedWindow("src")
cv.imshow("src", src)
cv.moveWindow("src", 0, 0)
cv.namedWindow("temp")
cv.imshow("temp", temp)
cv.moveWindow("temp", 0, 0)
row1, col1, channel1 = src.shape[:]
row2, col2, channel2 = temp.shape[:]
methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  # 三种匹配方法
for i in methods:
    result = cv.matchTemplate(src, temp, i)  # 计算匹配数值
    # 最小值 最大值 最小值索引(x,y) 最大值索引(x,y)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    if i == cv.TM_SQDIFF_NORMED:  # 此方法特殊处理
        x1y1 = min_loc
    else:
        x1y1 = max_loc
    x2y2 = (x1y1[0] + col2, x1y1[1] + row2)
    cv.rectangle(src, x1y1, x2y2, (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("result:" + str(i), result)
    cv.imshow("dst:" + str(i), src)
cv.waitKey(0)
