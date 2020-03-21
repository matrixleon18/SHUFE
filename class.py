#encoding=GBK

class MyClass(object):

    __name__ = "THIS IS MYCLASS NAME"

    def __init__(self, number):
        print("this is __init__")
        self.count = number
        pass

    def __new__(cls, number):
        print("this is __new__")
        inst = object.__new__(cls)
        print(inst)
        return inst
        pass

    def __del__(self):
        print("this is del")
        pass


    def __lt__(self, other):
        print("this is <")
        print(self.count)
        print(other.count)
        return self.count < other.count

    def addone(self):
        self.count += 1

    def __str__(self):
        return "here is __str__"




print("Now Begins")
m1 = MyClass(1)
m2 = MyClass(2)

print(m1<m2)

print(isinstance(m1, MyClass))
print(type(m1))

myclassone = MyClass(1)
myclasstwo = MyClass(2)

print(myclassone)

print(MyClass)

myclasstwo.addone()
myclasstwo.addone()
print(myclassone < myclasstwo)

print(myclassone.__name__)


