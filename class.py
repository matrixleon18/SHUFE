# encoding=GBK

"""
类的基本使用方法
"""

# class MyClass(object):
#
#     __name__ = "THIS IS MYCLASS NAME"
#
#     def __init__(self, number):
#         print("this is __init__")
#         self.count = number
#         pass
#
#     def __new__(cls, number):
#         print("this is __new__")
#         inst = object.__new__(cls)
#         print(inst)
#         return inst
#         pass
#
#     def __del__(self):
#         print("this is del")
#         pass
#
#
#     def __lt__(self, other):
#         print("this is <")
#         print(self.count)
#         print(other.count)
#         return self.count < other.count
#
#     def addone(self):
#         self.count += 1
#
#     def __str__(self):
#         return "here is __str__"
#
#
#
#
# print("Now Begins")
# m1 = MyClass(1)
# m2 = MyClass(2)
#
# print(m1<m2)
#
# print(isinstance(m1, MyClass))
# print(type(m1))
#
# myclassone = MyClass(1)
# myclasstwo = MyClass(2)
#
# print(myclassone)
#
# print(MyClass)
#
# myclasstwo.addone()
# myclasstwo.addone()
# print(myclassone < myclasstwo)
#
# print(myclassone.__name__)
#
#


# __init__ & __call__
# class X(object):
#     def __init__(self, a, b, range):
#         self.a = a
#         self.b = b
#         self.range = range
#         print('__init__ function')
#
#     def __call__(self, a, b):
#         self.a = a
#         self.b = b
#         print('__call__ function')
#
#     def __del__(self):
#         del self.a
#         del self.b
#         del self.range
#         print('__del__ function')
#
#
# x_instance = X(1, 2, 3)
# x_instance(1, 2)


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
input_param = a('i')
print('parameter of a is {}'.format(input_param))