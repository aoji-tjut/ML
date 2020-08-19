import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 图片
image_value = 0
image_max = 5
# 色彩空间
color_value = 0
color_max = 4
# x方向移动
x_value = 400
x_max = 1000
# y方向移动
y_value = 400
y_max = 1000
# 旋转角度y
angle_value = 90
angle_max = 360
# 缩放比例
scale_value = 5
scale_max = 20
# 内核大小
size_value = 5
size_max = 30
# 内核形状
shape_value = 0
shape_max = 2
# 模糊种类
blur_value = 3
blur_max = 3
# 边缘种类
canny_value = 0
canny_max = 3
# 霍夫变换
hough_value = 0
hough_max = 1
# 霍夫阈值
param_value = 0
param_max = 500
# 阈值
threshold_value = 128
threshold_max = 255
# 二值化种类
bin_value = 0
bin_max = 4
# 形态学种类
morph_value = 0
morph_max = 5
# 轮廓模式
retr_value = 0
retr_max = 3
# 轮廓方法
chain_value = 0
chain_max = 3


def callback_image(value):
    global image
    image_pos = cv.getTrackbarPos("image", "operation")  # 图片0-5
    if image_pos == 0:
        image = cv.imread("./image/girl.JPG")
    elif image_pos == 1:
        image = cv.imread("./image/dog.JPG")
    elif image_pos == 2:
        image = cv.imread("./image/building.JPG")
    elif image_pos == 3:
        image = cv.imread("./image/board.JPG")
    elif image_pos == 4:
        image = cv.imread("./image/线.JPG")
    elif image_pos == 5:
        image = cv.imread("./image/乱糟糟.JPG")
    cv.imshow("image", image)
    callback_color(0)
    callback_move(0)
    callback_rota(0)
    callback_blur(0)
    callback_canny(0)
    callback_kernel(0)
    callback_hough(0)
    callback_threshold(0)
    callback_morphology(0)
    callback_contours(0)


def callback_color(value):
    color = 0
    color_pos = cv.getTrackbarPos("color", "operation")  # 色彩空间0-4
    if color_pos == 0:
        color = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif color_pos == 1:
        color = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    elif color_pos == 2:
        color = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    elif color_pos == 3:
        color = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    elif color_pos == 4:
        color = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("color", color)


def callback_move(value):
    x_pos = cv.getTrackbarPos("move_x", "operation")  # x方向移动0-1000
    y_pos = cv.getTrackbarPos("move_y", "operation")  # y方向移动0-1000
    x = x_pos - 500
    y = y_pos - 500
    M = np.float32([[1, 0, x], [0, 1, y]])
    move = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv.imshow("move", move)


def callback_rota(value):
    angle_pos = cv.getTrackbarPos("angle", "operation")  # 旋转角度0-360
    scale_pos = cv.getTrackbarPos("scale", "operation")  # 缩放比例10-20
    scale = scale_pos / 10.0
    M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle_pos, scale)
    rota = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv.imshow("rota", rota)


def callback_kernel(value):
    global kernel
    shape = 0
    size_pos = cv.getTrackbarPos("size", "operation")  # 内核大小0-20
    shape_pos = cv.getTrackbarPos("shape", "operation")  # 内核形状0-2
    if shape_pos == 0:
        shape = cv.MORPH_RECT
    elif shape_pos == 1:
        shape = cv.MORPH_ELLIPSE
    elif shape_pos == 2:
        shape = cv.MORPH_CROSS
    if size_pos % 2 == 0:
        size_pos += 1
    kernel = cv.getStructuringElement(shape, (size_pos, size_pos))
    callback_blur(0)
    callback_canny(0)
    callback_hough(0)
    callback_morphology(0)


def callback_blur(value):
    global blur
    blur_pos = cv.getTrackbarPos("blur", "operation")  # 模糊种类0-3
    size_pos = cv.getTrackbarPos("size", "operation")  # 内核大小0-20
    if size_pos % 2 == 0:
        size_pos += 1
    if blur_pos == 0:
        blur = cv.blur(image, (size_pos, size_pos))
    elif blur_pos == 1:
        blur = cv.GaussianBlur(image, (size_pos, size_pos), 0)
    elif blur_pos == 2:
        blur = cv.medianBlur(image, size_pos)
    elif blur_pos == 3:
        blur = cv.bilateralFilter(image, 0, 100, size_pos)
    cv.imshow("blur", blur)


def callback_canny(value):
    global canny
    canny_pos = cv.getTrackbarPos("canny", "operation")  # 边缘种类0-3
    threshold_pos = cv.getTrackbarPos("threshold", "operation")  # 阈值0-255
    if canny_pos == 0:
        canny = cv.Canny(blur, threshold_pos, threshold_pos * 2)
    elif canny_pos == 1:
        canny = cv.Laplacian(blur, cv.CV_32F)
        canny = cv.convertScaleAbs(canny)
    elif canny_pos == 2:
        sobel_x = cv.Sobel(blur, cv.CV_32F, 1, 0)
        sobel_y = cv.Sobel(blur, cv.CV_32F, 0, 1)
        canny = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        canny = cv.convertScaleAbs(canny)
    elif canny_pos == 3:
        scharr_x = cv.Scharr(blur, cv.CV_32F, 1, 0)
        scharr_y = cv.Scharr(blur, cv.CV_32F, 0, 1)
        canny = cv.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
        canny = cv.convertScaleAbs(canny)
    cv.imshow("canny", canny)


def callback_hough(value):
    hough = 0
    hough_pos = cv.getTrackbarPos("hough", "operation")  # 霍夫变换0-1
    param_pos = cv.getTrackbarPos("param", "operation")  # 霍夫阈值0-500
    threshold_pos = cv.getTrackbarPos("threshold", "operation")  # 阈值0-255
    if hough_pos == 0:
        lines = cv.HoughLinesP(canny, 1.0, np.pi / 180, param_pos, minLineLength=20, maxLineGap=10)
        hough = np.zeros(image.shape[:], np.uint8)
        np.copyto(hough, image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif hough_pos == 1:
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=threshold_pos * 2, param2=param_pos,
                                  minRadius=5, maxRadius=30)
        circles = np.uint16(np.around(circles))
        hough = np.zeros(image.shape[:], np.uint8)
        np.copyto(hough, image)
        for circle in circles[0, :, :]:
            cv.circle(hough, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            cv.circle(hough, (circle[0], circle[1]), 2, (0, 0, 255), 2)
    cv.imshow("hough", hough)


def callback_threshold(value):
    bin = 0
    bin_pos = cv.getTrackbarPos("bin", "operation")  # 二值化种类0-4
    threshold_pos = cv.getTrackbarPos("threshold", "operation")  # 阈值0-255
    canny_pos = cv.getTrackbarPos("canny", "operation")  # 边缘种类0-3
    hough_pos = cv.getTrackbarPos("hough", "operation")  # 霍夫变换0-1
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if bin_pos == 0:
        ret, bin = cv.threshold(gray, threshold_pos, 255, cv.THRESH_BINARY)
    elif bin_pos == 1:
        ret, bin = cv.threshold(gray, threshold_pos, 255, cv.THRESH_BINARY_INV)
    elif bin_pos == 2:
        ret, bin = cv.threshold(gray, threshold_pos, 255, cv.THRESH_TRUNC)
    elif bin_pos == 3:
        ret, bin = cv.threshold(gray, threshold_pos, 255, cv.THRESH_TOZERO)
    elif bin_pos == 4:
        ret, bin = cv.threshold(gray, threshold_pos, 255, cv.THRESH_TOZERO_INV)
    cv.imshow("bin", bin)
    if canny_pos == 0:
        callback_canny(0)
    callback_hough(0)


def callback_morphology(value):
    morph = 0
    morph_pos = cv.getTrackbarPos("morphology", "operation")  # 形态学种类0-5
    if morph_pos == 0:
        morph = cv.erode(image, kernel)
    elif morph_pos == 1:
        morph = cv.dilate(image, kernel)
    elif morph_pos == 2:
        morph = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    elif morph_pos == 3:
        morph = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    elif morph_pos == 4:
        morph = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
    elif morph_pos == 5:
        morph = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("morph", morph)


def callback_contours(value):
    contours = 0
    retr_pos = cv.getTrackbarPos("retr", "operation")  # 模式模式0-3
    chain_pos = cv.getTrackbarPos("chain", "operation")  # 轮廓方法0-3
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, bin = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    if retr_pos == 0:
        if chain_pos == 0:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        elif chain_pos == 1:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        elif chain_pos == 2:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        elif chain_pos == 3:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    elif retr_pos == 1:
        if chain_pos == 0:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        elif chain_pos == 1:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        elif chain_pos == 2:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
        elif chain_pos == 3:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    elif retr_pos == 2:
        if chain_pos == 0:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        elif chain_pos == 1:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        elif chain_pos == 2:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
        elif chain_pos == 3:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
    elif retr_pos == 3:
        if chain_pos == 0:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        elif chain_pos == 1:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        elif chain_pos == 2:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        elif chain_pos == 3:
            img, contours, hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
    contour = np.zeros(image.shape[:], np.uint8)
    np.copyto(contour, image)
    for i, i_contour in enumerate(contours):
        cv.drawContours(contour, contours, i, (0, 0, 255), 2)
    cv.imshow("contour", contour)


def hist():
    colors = ("blue", "green", "red")
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.figure("Hist")
        plt.plot(hist, color=color)
    plt.show()


def OnMouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        hist()
    elif event == cv.EVENT_RBUTTONDOWN:
        plt.close("Hist")


operation = np.zeros((1, 500, 3), np.uint8)
cv.namedWindow("operation")
cv.moveWindow("operation", 940, 0)
cv.imshow("operation", operation)
cv.namedWindow("image")
cv.moveWindow("image", 0, 0)
cv.createTrackbar("image", "operation", image_value, image_max, callback_image)
cv.createTrackbar("color", "operation", color_value, color_max, callback_color)
cv.createTrackbar("move_x", "operation", x_value, x_max, callback_move)
cv.createTrackbar("move_y", "operation", y_value, y_max, callback_move)
cv.createTrackbar("angle", "operation", angle_value, angle_max, callback_rota)
cv.createTrackbar("scale", "operation", scale_value, scale_max, callback_rota)
cv.createTrackbar("size", "operation", size_value, size_max, callback_kernel)
cv.createTrackbar("shape", "operation", shape_value, shape_max, callback_kernel)
cv.createTrackbar("blur", "operation", blur_value, blur_max, callback_blur)
cv.createTrackbar("canny", "operation", canny_value, canny_max, callback_canny)
cv.createTrackbar("hough", "operation", hough_value, hough_max, callback_hough)
cv.createTrackbar("param", "operation", param_value, param_max, callback_hough)
cv.createTrackbar("threshold", "operation", threshold_value, threshold_max, callback_threshold)
cv.createTrackbar("bin", "operation", bin_value, bin_max, callback_threshold)
cv.createTrackbar("morphology", "operation", morph_value, morph_max, callback_morphology)
cv.createTrackbar("retr", "operation", retr_value, retr_max, callback_contours)
cv.createTrackbar("chain", "operation", chain_value, chain_max, callback_contours)
cv.setMouseCallback("image", OnMouse)
callback_image(0)
while cv.waitKey(0) == 27:
    break
