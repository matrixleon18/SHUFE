# encoding=GBK

"""
��Ļ���ʹ�÷���
"""


class MyClass(object):
    __name__ = "THIS IS MYCLASS NAME"           # ��������ڲ�����

    def __init__(self, number):                 # �����ʼ������
        print("this is __init__")
        self.count = number
        pass

    def __new__(cls, number):                   # ����new����
        print("this is __new__")
        inst = object.__new__(cls)              # ���ø����new
        print(inst)
        return inst
        pass

    def __del__(self):                          # ��������ڲ�����
        print("this is del")
        pass

    def __lt__(self, other):                    # ������С�ڲ�����
        print("this is <")
        print(self.count)
        print(other.count)
        return self.count < other.count

    def addone(self):                           # ����������ⲿ����
        self.count += 1

    def __str__(self):                          # ʵ����������÷���
        return "here is __str__"


print("Now Begins")
m1 = MyClass(1)                                 # �����������ʵ��
m2 = MyClass(2)

print(m1 < m2)                                  # �����ع���С�ڲ��������бȽ�

print(isinstance(m1, MyClass))                  # �ж�ʵ���Ƿ�����ĳ����
print(type(m1))                                 # ��ӡ��ʵ����������

myclassone = MyClass(1)
myclasstwo = MyClass(2)

print(myclassone)                               # ��ӡ����__str__()��������ֵ

print(MyClass)                                  # ��ӡ�����������

myclasstwo.addone()
myclasstwo.addone()
print(myclassone < myclasstwo)

print(myclassone.__name__)                      # ��ӡ����__name__��ֵ


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

    def __del__(self):                          # ������C++����������
        del self.a
        del self.b
        del self.range
        print('__del__ function')


x_instance = X(1, 2, 3)                         # ���ǵ�����__init__����
x_instance(1, 2)                                # ���ǵ�����__call__����


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
input_param = a('i')                            # ����Ż����__call__����
print('parameter of a is {}'.format(input_param))
