# encoding=GBK

"""
异常处理的基本使用方法
"""


class MYException(Exception):                           # 从异常基类继承一个异常
    def __init__(self, error_info):
        super().__init__(self)                          # 要调用父类的初始化方法
        self.errorinfo = error_info

    def __str__(self):
        return self.errorinfo
    pass


try:
    # do something                                      # 这里放要实现的功能
    raise MYException("my exception")                   # 如果有问题就产生一个异常
except MYException as e:                                # 捕捉异常
    print(e)                                            # 打印出异常信息
    print(e.errorinfo)
finally:                                                # 无论是否异常都会到这里
    print("Finally")


try:
    # do something
    raise MYException(30)
except MYException as e:
    print("num :"+ str(e.errorinfo) + " is wrong")
finally:
    print("Yes. It's wrong")