# encoding=GBK

"""
������ʹ��
"""
# �������Ŀ¼������__init__.py�ļ����ǰ���
import math

# ���ð��ĺ���
print(math.floor(2.34566))

# ֱ�ӵ������ĺ���
from math import floor
print(floor(2.34))

# �г��������ṩ�ĺ���
print(dir(math))

print("======================")

# ��ӡ�����ṩ�İ�����Ϣ
print(help(math))

print("======================")

# ��ӡ������˵���ĵ�
print(math.__doc__)

print("======================")

# ��ӡ������˵���ĵ�������ֱ��google
print(help(math.log))

print("======================")

# ���ú���
print(math.pow(2, 3))

# �������ú���
a = [1, 2, 3, 4, 5]
print(len(a))
print(abs(-3))
print(round(2.3456))

# ����������
p = math.pow
print(p(2, 3))

# print�������÷�
a = '22222222222'
print('22222222222222')
print("this is a string")
print("this is a \"string\"")
print("{} is a string".format(a))


# ���庯��
def func1():
    print("This is a func")


# ���ú���
func1()


# ���庯���Ͳ���
def area(width, length):
    print(width*length)


# ���������ú���
area(3, 5)

# �Ա�����, tuple, list���Կ������ֺ�tuple�����޸�Ԫ�ص�ֵ��list�����޸�
b = tuple((2, 3))
c = list([1, 24, 6])
print(b)
c[0] = 9
print(c)

print("======================")


# ���������ͣ�tuple�Ͳ����Ǵ�ֵ��
def func(a):
    a = 20
    print(a)


x = 1
func(x)
print(x)


# ������list�Ͳ����Ǵ���ַ��
def func(a):
    a[0] = 9


ll = [1, 2, 3]
func(ll)
print(ll)

print("======================")


def func(a, b):
    print(a-b)


# ����������ָ����ֵ
func(b=2, a=32)


# �����Ĳ���Ĭ�ϸ�ֵ
def func(a, b=0):
    print(a+b)


func(33)


# �����Ĳ���tuple����
def func(arg1, *args):
    print(args)
    print(args[0])
    print(args[1])
    print(args[2])


func(1, 23, 45, 6)


# �����Ŀɱ����
def printinfo(arg1, **vardict):
    print(arg1)
    print(vardict)


printinfo(1, a=2, b=3)

# lambda������ʹ��
s = lambda arg1, arg2: arg1 * arg2
print("Result is : ", s(5, 6))

# lambda���mapʹ�á�Ч�ʸ��ߡ�����ѭ����
ll = [1, 2, 3, 5]
print(list(map(lambda x: x*2, ll)))

# ��ӡ�������÷���
a = 32
print("this is ")
print("this is ", 32)
print("this is "+str(a))
print("this is {} number {}".format(a, 333))