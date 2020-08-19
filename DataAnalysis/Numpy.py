import numpy as np

#axis=n，改变n所在的维度，其他维度不变

print("-------------------------------------------------------------------------创建")
np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("np_array =\n", np_array)
np_zeros = np.zeros([3, 3])
print("np_zeros =\n", np_zeros)
np_zeros_int = np.zeros([3, 3], dtype=int)
print("np_zeros_int =\n", np_zeros_int)
np_ones = np.ones([3, 3])
print("np_ones =\n", np_ones)
np_full = np.full([3, 3], 666)
print("np_full =\n", np_full)
np_arange = np.arange(0, 1, 0.2)  # [0,1) 步长0.2
print("np_arange =\n", np_arange)
np_linspace = np.linspace(0, 20, 10, dtype=int)  # [0,20] 15个
print("np_linspace =\n", np_linspace)
# np.random.seed(666)#随机种子
np_rand_int = np.random.randint(0, 10, [3, 3])  # [0,10) 3*3个
print("np_rand_int =\n", np_rand_int)
np_rand_rand = np.random.random([3, 3])
print("np_rand_rand =\n", np_rand_rand)
np_rand_normal = np.random.normal(10, 100, [3, 3])  # 正态分布 均值10 标准差100
print("np_rand_normal =\n", np_rand_normal)

print("-------------------------------------------------------------------------基本操作")
x = np.arange(20).reshape([4, 5])
print("x =\n", x)
print("x.ndim =", x.ndim)
print("x.shape =", x.shape)
print("x.size =", x.size)
print("x[1,2] =", x[1, 2])
print("x[1,:] =", x[1, :])
print("x[:2,:3] =\n", x[:2, :3])
print("x.reshape([5, 3]) =\n", x.reshape([5, 4]))
print("x.reshape([2, -1]) =\n", x.reshape([2, -1]))  # 固定2行
print("x.reshape([-1, 2]) =\n", x.reshape([-1, 2]))  # 固定2列
print("x.ravel() =\n", x.ravel())  # 降为一维数组 新数组改变原数组也改变
print("x.flatten() =\n", x.flatten())  # 降为一维数组 新数组改变原数组不改变

print("-------------------------------------------------------------------------合并/分割")
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[3, 2, 1], [6, 5, 4]])
z1 = np.array([7, 8, 9])
z2 = np.array([[1, 2], [3, 4]])
print("x =\n", x)
print("y =\n", y)
print("z1 =\n", z1)
print("z2 =\n", z2)
print("np.concatenate([x, y]) =\n", np.concatenate([x, y]))  # 垂直拼接 行数增加
print("np.concatenate([x, y], axis=1) =\n", np.concatenate([x, y], axis=1))  # 水平拼接 列数增加
print("np.hstack([x, z2]) =\n", np.hstack([x, z2]))  # 水平拼接 列数增加
print("np.vstack([x, z1]) =\n", np.vstack([x, z1]))  # 垂直拼接 行数增加
x1, x2 = np.split(x, [1])
print("np.split(x, [1]) =\n", x1, "\n", x2)  # 0 1
y1, y2 = np.split(y, [1], axis=1)
print("np.split(y, [1], axis=1) =\n", y1, "\n", y2)  # 0 1-2
x1, x2 = np.hsplit(x, [1])
print("np.hsplit(x, [1]) =\n", x1, "\n", x2)  # 水平分割
y1, y2 = np.vsplit(x, [1])
print("np.vsplit(y, [1]) =\n", y1, "\n", y2)  # 垂直分割

print("-------------------------------------------------------------------------矩阵运算")
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x.copy()
z = np.array([[1, 2], [3, 4], [5, 6]])
print("x =\n", x)
print("y =\n", y)
print("z =\n", z)
inv = np.arange(1, 10).reshape([3, 3])
print("x*2 =\n", x * 2)
print("x**2 =\n", x ** 2)
print("x + y =\n", x + y)
print("x*y =\n", x * y)  # 对应元素相乘
print("x.dot(z) =\n", x.dot(z))  # 矩阵乘法
# print("np.linalg.inv(inv) =\n", np.linalg.inv(inv))  # 逆矩阵
print("np.linalg.pinv(x) =\n", np.linalg.pinv(x))  # 伪逆矩阵

print("-------------------------------------------------------------------------聚合运算")
x = np.arange(1, 17).reshape([4, 4])
print("x =\n", x)
print("np.min(x) =\n", np.min(x))
print("np.max(x) =\n", np.max(x))
print("np.sum(x) =\n", np.sum(x))
print("np.sum(x), axis=0 =\n", np.sum(x, axis=0))  # 列相加
print("np.sum(x), axis=1 =\n", np.sum(x, axis=1))  # 行相加
print("np.prod(x) =\n", np.prod(x))  # 乘积
print("np.mean(x) =\n", np.mean(x))  # 平均值
print("np.median(x) =\n", np.median(x))  # 中位数
for i in [0, 25, 50, 75, 100]:
    print("np.percentile(x, %d) =\n" % i, np.percentile(x, i))  # 百分数点
print("np.var(x) =\n", np.var(x))  # 方差
print("np.std(x) =\n", np.std(x))  # 标准差

print("-------------------------------------------------------------------------arg运算")
x = np.arange(1, 17)
print("x =\n", x)
np.random.shuffle(x)  # 乱序 shuffle将原数组乱序，permutation不改变原数组，返回乱序后的数组
print("np.argmin(x) =\n", np.argmin(x))  # 最小值索引
print("np.argmax(x) =\n", np.argmax(x))  # 最大值索引
print("np.sort(x) =\n", np.sort(x))  # x本身没有改变
print("np.argsort(x) =\n", np.argsort(x))  # x排序的索引
print("np.partition(x, 5) =\n", np.partition(x, 5))  # 以5为分界点
print("np.argpartition(x, 5) =\n", np.argpartition(x, 5))  # 以5为分界点的索引
x = x.reshape([4, -1])
print("x =\n", x)
print("np.sort(x, axis=0) =\n", np.sort(x, axis=0))  # 列排序
print("np.sort(x, axis=1) =\n", np.sort(x, axis=1))  # 行排序

print("-------------------------------------------------------------------------Fancy indexing")
x = np.arange(1, 17)
print("x =\n", x)
index = [3, 5, 8]  # 索引值
print("x[index] =\n", x[index])
index = np.array([[0, 2], [1, 3]])
print("x[index] =\n", x[index])
row = np.array([0, 1])
col = np.array([2, 3])
x = x.reshape([4, -1])
print("x =\n", x)
print("x[row, col] =\n", x[row, col])  # [0,2],[1,3]
bool = np.array([True, False, True, False])
print("x[1:3,bool] =\n", x[1:3, bool])  # [1,0],[1,2],[2,0],[2,2]

print("-------------------------------------------------------------------------比较")
x = np.arange(0, 16).reshape([4, -1])
print("x =\n", x)
print("x < 10 =\n", x < 10)
print("np.any(x == 0) =\n", np.any(x == 0))  # x中是否有0
print("np.all(x == 0) =\n", np.all(x == 0))  # x中是否都为0
print("np.any(x%2 == 0) =\n", np.sum(x % 2 == 0))  # x偶数个数
print("np.sum((x > 3) & (x < 10)) =\n", np.sum((x > 3) & (x < 10)))  # 3<x<10的个数 用位运算
print("np.sum(~(x==0)) =\n", np.sum(~(x == 0)))  # x非0的个数

print("-------------------------------------------------------------------------最重要的")
# 用比较返回的bool值作为索引
x = np.arange(0, 16).reshape([4, -1])
print("x[x<10] =\n", x[x < 10])  # 取出x中x<10的数组
print("x[x[:, 3] % 3 == 0, :] =\n", x[x[:, 3] % 3 == 0, :])  # 取出x中第四列可以被3整除的行向量

# print("-------------------------------------------------------------------------txt文件操作")
# # 路径 数据类型 跳过开头为'#'的行 数据隔符 跳过前3行
# load = np.loadtxt("dir", dtype=float, comments='#', delimiter=None, skiprows=3)
# # 路径 数据 数据格式 数据分隔符 换行符 写入文件开头的字符串 写入文件末尾的字符串 将header/footer的字符串开头加上'#'
# np.savetxt("dir", x, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
