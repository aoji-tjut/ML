from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# 曲线图
x = np.linspace(-10, 10, 1000)
sinx = np.sin(x)
cosx = np.cos(x)
plt.figure("plot")
plt.plot(x, sinx, label="sin(x)")  # 横坐标(缺省) 纵坐标 标签
plt.plot(x, cosx, label="cos(x)", color='m', linestyle='--')  # 颜色 形状
plt.legend()  # 显示图示
# plt.xlim(0, 20)#横坐标范围
# plt.ylim(-1, 1)#纵坐标范围
plt.axis([-12, 12, -1.5, 1.5])  # x左 x右 y下 y上
plt.xlabel("x")  # 横坐标说明
plt.ylabel("y")  # 纵坐标说明
plt.xticks(np.linspace(-10, 10, 10))  # 设置x轴数据密度
plt.yticks([-1.0, 1.0], ["min", "max"])  # 设置y轴数据、文字
plt.text(-0.5, -1.2, "balabala", fontdict={"size": 20, "color": "red"})  # 绘制文字
plt.title("title")  # 标题
plt.show()  # 显示

# 散点图
x = np.random.normal(0, 1, 10000)
y = np.random.normal(0, 1, 10000)
plt.figure("scatter")
plt.scatter(x, y, alpha=0.2, marker='^')  # 透明度 形状
plt.show()

# 子图
x1 = x2 = np.linspace(-10, 10, 1000)
y1 = x1 ** 2
y2 = x2 ** -1
plt.figure("subplot", (10, 5))
ax1 = plt.subplot(1, 2, 1)  # 1行2列第1个
ax2 = plt.subplot(1, 2, 2)  # 1行2列第2个
plt.sca(ax1)  # 选中ax1
plt.plot(x1, y1)
plt.sca(ax2)  # 选中ax2
plt.plot(x2, y2)
plt.show()

# plt.ioff()  # 关闭交互模式 遇到show()卡住
# plt.ion()  # 打开交互模式 遇到show()继续向下执行

# 动态图
plt.figure("dynamic")
for i in range(10):
    # plt.cla()  # 清除当前画布
    plt.scatter(i + 1, np.random.random(1), c='b')
    plt.axis([0, 11, -3, 3])
    plt.pause(0.1)
plt.show()


# 等高线
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)  # z轴


plt.figure("contourf")
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)  # 均匀生成[256,256]矩阵
plt.contourf(X, Y, f(X, Y), 10, alpha=0.8, cmap="cool")  # 填充颜色 X Y Z 密度 透明度 色彩空间
C = plt.contour(X, Y, f(X, Y), 10, colors='k')  # 绘制轮廓 X Y Z 密度
plt.clabel(C, inline=True, fontsize=10)  # 标注文字 轮廓 线上标注 字号
plt.show()

# 三维图
iris = datasets.load_iris()
X = iris.data
y = iris.target
iris_3d = plt.figure("3D iris")
ax = Axes3D(iris_3d)
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='b')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='g')
ax.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], c='r')
plt.show()

contourf_3d = plt.figure("3D contourf")
ax = Axes3D(contourf_3d)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)  # 均匀生成矩阵
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, edgecolor='k', rstride=1, cstride=1, cmap="rainbow")  # X Y Z 间隔线 row步长 col步长 色彩空间
ax.contourf(X, Y, Z, zdir='z',offset=-2)  # 沿Z轴向XOY投影等高线 画在Z=-2的平面
ax.set_zlim(-2,2)#Z轴范围
plt.show()

# 显示图片
plt.figure("image")
image = np.loadtxt("./image.txt")
plt.imshow(image, cmap="gray")  # 灰色空间
plt.colorbar()  # 绘制色标
plt.show()
