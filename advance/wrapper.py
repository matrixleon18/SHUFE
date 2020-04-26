# coding = GBk

import pytest


# def func(x):
#     return x + 1
#
#
# def test_func():
#     assert func(3) == 4
#
#
# class TestClass:
#     def test_one(self):
#         x = "this"
#         assert h in x
#
#     def test_two(self):
#         y = "hello"
#         assert y == "hi"


# def test_main():
#     assert 1==1
#
#
# if __name__ == '__main__':
#     pytest.main()


# class Test(object):
#     def __init__(self, func):
#         print("Test init")
#         print("func name is {}".format(func.__name__))
#         self.__func = func
#
#     def __call__(self, *args, **kwargs):
#         print("call is calling")
#         self.__func()
#         print("call call end")
#
# @Test
# def test2():
#     print("This is test2 func")
#
# test2()

#############################################

import functools

# from functools import partial
#
#
# def add(x, y):
#     return x+y
#
#
# add2 = partial(add, y=2)
#
# print(add2(3))

###############################################

from functools import wraps


def wrapper(func):
    @wraps(func)
    def wrap_func(*args, **kwargs):
        return func(*args, **kwargs)
    return wrap_func

@wrapper
def wrapped():
    print("Is wrapped")


print(wrapped.__doc__)
print(wrapped.__name__)
wrapped()