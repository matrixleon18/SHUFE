# encoding=GBK

"""
类的基本使用方法
"""


class MyClass(object):
    __name__ = "THIS IS MYCLASS NAME"           # 定义类的内部变量

    def __init__(self, number):                 # 定义初始化函数
        print("this is __init__")
        self.count = number
        pass

    def __new__(cls, number):                   # 定义new函数
        print("this is __new__")
        inst = object.__new__(cls)              # 调用父类的new
        print(inst)
        return inst
        pass

    def __del__(self):                          # 定义类的内部函数
        print("this is del")
        pass

    def __lt__(self, other):                    # 重载了小于操作符
        print("this is <")
        print(self.count)
        print(other.count)
        return self.count < other.count

    def addone(self):                           # 定义类类的外部方法
        self.count += 1

    def __str__(self):                          # 实现了类的内置方法
        return "here is __str__"


print("Now Begins")
m1 = MyClass(1)                                 # 生成两个类的实例
m2 = MyClass(2)

print(m1 < m2)                                  # 用重载过的小于操作符进行比较

print(isinstance(m1, MyClass))                  # 判断实例是否属于某个类
print(type(m1))                                 # 打印出实例的类名称

myclassone = MyClass(1)
myclasstwo = MyClass(2)

print(myclassone)                               # 打印出了__str__()方法返回值

print(MyClass)                                  # 打印出了类的名称

myclasstwo.addone()
myclasstwo.addone()
print(myclassone < myclasstwo)

print(myclassone.__name__)                      # 打印出了__name__的值


class X(object):
    def __init__(self, a, b, range):
        self.a = a
        self.b = b
        self.range = range
        print('__init__ function')

    def __call__(self, a, b):
        self.a = a
        self.b = b
        print('__call__ function')

    def __del__(self):                          # 类似与C++的析构函数
        del self.a
        del self.b
        del self.range
        print('__del__ function')


x_instance = X(1, 2, 3)                         # 这是调用了__init__函数
x_instance(1, 2)                                # 这是调用了__call__函数


class A(object):
    def forward(self, inputs):
        print('forward function')
        print('forward param: {}'.format(inputs))
        return inputs

    def __call__(self, param):
        print('__call__ function')
        print('__call__ param: {}'.format(param))
        ret = self.forward(param)
        return ret


a = A()
input_param = a('i')                            # 这里才会调用__call__函数
print('parameter of a is {}'.format(input_param))
