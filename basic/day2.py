# encoding=GBK

"""
函数的使用
"""
# 导入包。目录里面有__init__.py文件就是包。
import math

# 调用包的函数
print(math.floor(2.34566))

# 直接导入包里的函数
from math import floor
print(floor(2.34))

# 列出包里所提供的函数
print(dir(math))

print("======================")

# 打印出包提供的帮助信息
print(help(math))

print("======================")

# 打印出包的说明文档
print(math.__doc__)

print("======================")

# 打印出函数说明文档。建议直接google
print(help(math.log))

print("======================")

# 调用函数
print(math.pow(2, 3))

# 调用内置函数
a = [1, 2, 3, 4, 5]
print(len(a))
print(abs(-3))
print(round(2.3456))

# 函数重命名
p = math.pow
print(p(2, 3))

# print语句基本用法
a = '22222222222'
print('22222222222222')
print("this is a string")
print("this is a \"string\"")
print("{} is a string".format(a))


# 定义函数
def func1():
    print("This is a func")


# 调用函数
func1()


# 定义函数和参数
def area(width, length):
    print(width*length)


# 带参数调用函数
area(3, 5)

# 对比数字, tuple, list可以看到数字和tuple不能修改元素的值。list可以修改
b = tuple((2, 3))
c = list([1, 24, 6])
print(b)
c[0] = 9
print(c)

print("======================")


# 函数数字型，tuple型参数是传值的
def func(a):
    a = 20
    print(a)


x = 1
func(x)
print(x)


# 函数的list型参数是传地址的
def func(a):
    a[0] = 9


ll = [1, 2, 3]
func(ll)
print(ll)

print("======================")


def func(a, b):
    print(a-b)


# 函数参数的指定赋值
func(b=2, a=32)


# 函数的参数默认赋值
def func(a, b=0):
    print(a+b)


func(33)


# 函数的参数tuple访问
def func(arg1, *args):
    print(args)
    print(args[0])
    print(args[1])
    print(args[2])


func(1, 23, 45, 6)


# 函数的可变参数
def printinfo(arg1, **vardict):
    print(arg1)
    print(vardict)


printinfo(1, a=2, b=3)

# lambda函数的使用
s = lambda arg1, arg2: arg1 * arg2
print("Result is : ", s(5, 6))

# lambda结合map使用。效率更高。不用循环。
ll = [1, 2, 3, 5]
print(list(map(lambda x: x*2, ll)))

# 打印函数常用方法
a = 32
print("this is ")
print("this is ", 32)
print("this is "+str(a))
print("this is {} number {}".format(a, 333))