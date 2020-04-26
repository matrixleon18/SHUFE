# encoding=GBK

"""
NumPy是Python中科学计算的基础软件包。 它是一个提供多了维数组对象，多种派生对象（如：掩码数组、矩阵）以及用于快速操作数组的函数及API，
它包括数学、逻辑、数组形状变换、排序、选择、I/O 、离散傅立叶变换、基本线性代数、基本统计运算、随机模拟等等。
NumPy中的核心是ndarray对象。ndarray对象是具有相同类型和大小（通常是固定大小）项目的多维容器。可以对其进行索引、切片、形状变换等操作。

ndarry基本使用方法
"""

import numpy as np

data = [[1, 2, 3],               # 这就是一个list。虽然看起来像个矩阵
       [4, 5, 6]]

nd_arr = np.array(data)          # 转换成ndarray

print(nd_arr)                   # 现在就是个2x3矩阵了
print(type(data))                # <class 'list'>  这就是个list
print(type(nd_arr))             # <class 'numpy.ndarray'> 这是ndarray对象


# ndarray对象的属性
print(nd_arr.ndim)              # 数组纬度的个数。纬度被成为了rank （秩）

print(nd_arr.shape)             # 数组的形状。也就是各个纬度上数组的大小。tuple格式

print(nd_arr.size)              # 数组中元素总数。也就是shape的tuple每个数的乘积

print(nd_arr.dtype)             # 数组中元素类型

print(nd_arr.itemsize)          # 数组中每个元素的字节大小

print(nd_arr.real)              # 数组每个元素只取实数部分

print(nd_arr.imag)              # 数组每个元素只取虚数部分


# ndarray创建方式

data = [[1, 2, 3], [4, 5, 6]]

nd_arr = np.array(data)         # 将输入的list, tuple, 数组等序列类型数据转换成ndaray对象
print(nd_arr)

print(np.asarray(nd_arr))       # array和asarray都可以将数据结构转化成ndarray，
                                # 区别是当输入ndarry时，array会生成出一个内存里的copy，asarray不会；

print(np.zeros([2, 3]))         # 创建一个全是0的ndarray；输入list指定每个rank的shape

print(np.zeros_like(data))      # 创建一个具有和data一样shape的全是0的ndarray

print(np.ones([2, 3]))          # 创建一个全是1的ndarray；输入的list指定了每个rank的shape

print(np.ones_like(data))       # 创建一个具有和data一样shape的全是1的ndarray

print(np.empty([2, 3]))         # 根据指定的形状创建ndarray对象，该矩阵采用随机元素填充

print(np.empty_like(data))      # 同上

print(np.full([2, 3], fill_value='a'))  # 创建一个指定形状的ndarray对象，并按照指定值进行填充

print(np.full_like(data, 8))    # 同上

print(np.identity(4))           # 创建一个nxn的单位方阵

print(np.random.rand(2, 3))     # 创建一个指定形状的ndarray对象，对象其中的值是[0,1)之间的随机数

print(np.random.randn(2, 3))    # 创建一个指定形状的ndarray对象，对象其中的值是服从标准正态分布的随机数

print(np.random.randint(low=0, high=5, size=(2, 3), dtype='I'))       # 在[low, high)之间取值创建指定形状的对象

print(np.random.random(size=(2, 3)))      # 生成随机浮点数的ndarray

print(np.random.choice((4, 3, 2, 1), size=None, replace=True, p=None))       # 从序列里按照p的概率选择

print(np.random.seed(100))  # 指定了种子数，每次random出来的随机数一致。用于验证和测试。

"""
numpy.eye(N, M=None, k=0, dtype=<class‘float’>, order=’C’)
N: 行数
M：列数，默认等于N
k：对角线索引：0（默认）为主对角线，正值是指上对角线，负值是指到下对角线
order: {'C', 'F'}输出是否应存储在主要行（C样式）中或内存中的列主（Fortran样式）顺序
"""
print(np.eye(5))     # 这样生成了一个size为5的单位矩阵

print(np.linspace(start=0, stop=10, num=5))         # 在[start, stop]之间创建5个间隔均匀的数字的array

# 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
# 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
# 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
print(np.arange(0, 100, 3))

np.arange(0, 100, 3).tofile('tmp.txt')           # 把产生的array写入到文件中

print(np.fromfile('tmp.txt', dtype=np.int))      # 从文件中读取array。一定要指定数据类型，否则不一定正确。默认float

