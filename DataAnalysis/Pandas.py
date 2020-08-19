import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------创建")
print("pd.Series(np.linspace(1, 5, 5))")
s = pd.Series(np.linspace(1, 5, 5))  # 一维 只有index没有column
print(s, "\n")

print("""pd.date_range("20200101", periods=3)""")
date = pd.date_range("20200101", periods=3)
print(date, "\n")

print("pd.DataFrame(np.random.randint(0, 10, [3, 4]), index=date, columns=['a', 'b', 'c', 'd'])")
df = pd.DataFrame(np.random.random([3, 4]), index=date, columns=['a', 'b', 'c', 'd'])  # 二维
print(df, "\n")

print(
    """pd.DataFrame({'A': np.linspace(1, 3, 3), 'B': ['a', 'b', 'c'], 'C': np.random.random([3]), 'D': ["AAA", "BBB", "CCC"]}, index=date)""")
dict = {'A': np.linspace(1, 3, 3),
        'B': ['a', 'b', 'c'],
        'C': np.random.random([3]),
        'D': ["AAA", "BBB", "CCC"]}
df = pd.DataFrame(dict, index=date)
print(df, "\n")

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------基本操作")

print("df.shape, df.dtypes, df.ndim, df.index, df.columns, df.values")
print(df.shape, df.dtypes, df.ndim, df.index, df.columns, df.values, '\n', sep='\n')  # 各种属性

print("df.info()")
print(df.info(), '\n')  # 基本信息

print("df.head(1)")
print(df.head(1), '\n')  # 前1行

print("df.tail(1)")
print(df.tail(1), '\n')  # 后1行

print("df[:2]['B']")
print(df[:2]['B'], '\n')  # 某行某列

print("df.describe()")
print(df.describe(), "\n")  # 只能计算数字

print("print(df.T)")
print(df.T, "\n")  # 转置

print("df.sort_index(axis=0, ascending=False)")
print(df.sort_index(axis=0, ascending=False), "\n")  # 按行index排序 降序

print("df.sort_index(axis=1, ascending=False)")
print(df.sort_index(axis=1, ascending=False), "\n")  # 按列columns排序 降序

print("df.sort_values(by='C', ascending=True)")
print(df.sort_values(by='C', ascending=True), "\n")  # 按B值排序 升序

print("df.iloc[2, 0] = 123")
df.iloc[2, 0] = 123  # 按下标改值
print(df, "\n")

print("""df.loc["20200101", 'C'] = 1""")
df.loc["20200101", 'C'] = 1  # 按标签改值
print(df, "\n")

print("df.B[df.C < 1.0] = 'z'")
df.B[df.C < 1] = 'z'  # 把C中小于1的B值改为z
print(df, "\n")

print("""df['E'] = ["北京", "上海", "广州"]""")
df['E'] = ["北京", "上海", "广州"]  # 新增一列E
print(df, "\n")

print("""df.append(pd.Series([3, 'o', 0.233333, "DDD", "深圳"], index=['A', 'B', 'C', 'D', 'E']), ignore_index=True)""")
row = pd.Series([3, 'o', 0.233333, "DDD", "深圳"], index=['A', 'B', 'C', 'D', 'E'])
df = df.append(row, ignore_index=True)  # 新增一行 重新排列index
df = df.set_index(pd.date_range("20200101", periods=df.shape[0]))  # 重设index为日期
print(df, "\n")

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------处理丢失数据")
df = df.copy()
df.iloc[2, 0] = np.nan
df.iloc[1, :] = np.nan
print("temp.iloc[2, 0] = np.nan, temp.iloc[1, :] = np.nan")
print(df, "\n")

print("np.any(temp.isnull()==True)")
print(np.any(df.isnull() == True), "\n")  # 检查是否有丢失数据

print("""temp.dropna(axis=0, how="any")""")
print(df.dropna(axis=0, how="any"), "\n")  # 删除有NaN的行

print("""temp.dropna(axis=0, how="all")""")
print(df.dropna(axis=0, how="all"), "\n")  # 删除全为NaN的行

print("temp.fillna(value=0)")
print(df.fillna(value=0), "\n")  # 将NaN设为0

print("df.drop(['B', 'C'], axis=1)")
print(df.drop(['B', 'C'], axis=1), "\n")  # 删除B、C列

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------导入导出")
# names：自定义columns表头
# index_col：使用第index_col列作为index表头 不忽略任何列
# header：使用第header行作为columns表头 忽略小于header的行
# usecols：只读入第usecols列
# nrows：只读取前nrows行
print("""pd.read_csv("./student.csv")""")
std = pd.read_csv("./student.csv")
print(std, "\n")
std.to_csv("./student_save.csv")

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------concat")
# index、columns相同
df1 = pd.DataFrame(np.zeros((3, 4), dtype=np.float) + 1)
df2 = pd.DataFrame(np.zeros((3, 4), dtype=np.float) + 2)
print("df1")
print(df1, "\n")
print("df2")
print(df2, "\n")
print("pd.concat([df1, df2], axis=0)")
result = pd.concat([df1, df2], axis=0)  # 纵向合并
print(result, "\n")
print("pd.concat([df1, df2], axis=0, ignore_index=True)")
result = pd.concat([df1, df2], axis=0, ignore_index=True)  # 忽略各自index 重新排列index
print(result, "\n")
print("pd.concat([df1, df2], axis=1)")
result = pd.concat([df1, df2], axis=1)  # 横向合并
print(result, "\n")

# index、columns不同
df1 = pd.DataFrame(np.zeros((3, 4), dtype=np.float) + 1, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.zeros((3, 4), dtype=np.float) + 2, columns=['c', 'd', 'e', 'f'], index=[2, 3, 4])
print("df1")
print(df1, "\n")
print("df2")
print(df2, "\n")
print("""pd.concat([df1, df2], axis=0, join="outer")""")
result = pd.concat([df1, df2], axis=0, join="outer")  # 合并所有columns 未知部分NaN填充
print(result, "\n")
print("""pd.concat([df1, df2], axis=0, join="inner")""")
result = pd.concat([df1, df2], axis=0, join="inner")  # 合并相同columns
print(result, "\n")
print("""pd.concat([df1, df2], axis=1, join_axes=None)""")
result = pd.concat([df1, df2], axis=1, join_axes=None)  # 合并所有index 未知部分NaN填充
print(result, "\n")
print("""pd.concat([df1, df2], axis=1, join_axes=[df1.index])""")
result = pd.concat([df1, df2], axis=1, join_axes=[df1.index])  # 以df1.index为准 未知部分NaN填充
print(result, "\n")

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------merge")
# 唯一K
left = pd.DataFrame({'A': ["A0", "A1", "A2", "A3"],
                     'B': ["B0", "B1", "B2", "B3"],
                     'K': ["K0", "K1", "K2", "K3"]})
right = pd.DataFrame({'K': ["K0", "K1", "K2", "K3"],
                      'C': ["C0", "C1", "C2", "C3"],
                      'D': ["D0", "D1", "D2", "D3"]})
print("left")
print(left, "\n")
print("right")
print(right, "\n")
print("""pd.merge(left, right, on='K', indicator="方式")""")
result = pd.merge(left, right, on='K', indicator="方式")
print(result, "\n")

# 不唯一K
left = pd.DataFrame({"K_1": ["K0", "K0", "K1", "K2"],
                     "K_2": ["K0", "K1", "K0", "K1"],
                     'A': ["A0", "A1", "A2", "A3"],
                     'B': ["B0", "B1", "B2", "B3"]})
right = pd.DataFrame({"K_1": ["K0", "K1", "K1", "K2"],
                      "K_2": ["K0", "K0", "K0", "K0"],
                      'C': ["C0", "C1", "C2", "C3"],
                      'D': ["D0", "D1", "D2", "D3"]})
print("left")
print(left, "\n")
print("right")
print(right, "\n")
print("""pd.merge(left, right, on=["K_1", "K_2"], how="outer", indicator="method")""")
result = pd.merge(left, right, on=["K_1", "K_2"], how="outer", indicator="method")
print(result, "\n")
print("""pd.merge(left, right, on=["K_1", "K_2"], how="inner", indicator="method")""")
result = pd.merge(left, right, on=["K_1", "K_2"], how="inner", indicator="method")
print(result, "\n")
print("""pd.merge(left, right, on=["K_1", "K_2"], how="left", indicator="method")""")
result = pd.merge(left, right, on=["K_1", "K_2"], how="left", indicator="method")
print(result, "\n")
print("""pd.merge(left, right, on=["K_1", "K_2"], how="right", indicator="method")""")
result = pd.merge(left, right, on=["K_1", "K_2"], how="right", indicator="method")
print(result, "\n")

# index作为索引
left = pd.DataFrame({'A': ["A0", "A1", "A2"],
                     'B': ["B0", "B1", "B2"]},
                    index=["K0", "K1", "K2"])
right = pd.DataFrame({'C': ["C0", "C2", "C3"],
                      'D': ["D0", "D2", "D3"]},
                     index=["K0", "K2", "K3"])
print("left")
print(left, "\n")
print("right")
print(right, "\n")
print("""pd.merge(left, right, left_index=True, right_index=True, how="outer", indicator="method")""")
result = pd.merge(left, right, left_index=True, right_index=True, how="outer", indicator="method")
print(result, "\n")
print("""pd.merge(left, right, left_index=True, right_index=True, how="inner", indicator="method")""")
result = pd.merge(left, right, left_index=True, right_index=True, how="inner", indicator="method")
print(result, "\n")

# 区分相同属性
boys = pd.DataFrame({'K': ["K0", "K1", "K2"], "age": [1, 2, 3]})
girls = pd.DataFrame({'K': ["K0", "K0", "K3"], "age": [4, 5, 6]})
print("boys")
print(boys, "\n")
print("girls")
print(girls, "\n")
print("""pd.merge(boys, girls, on='K', suffixes=["_boy", "_girl"], how="inner")""")
result = pd.merge(boys, girls, on='K', suffixes=["_boy", "_girl"], how="inner")
print(result, "\n")
