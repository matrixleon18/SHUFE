

# class Person(object):
#     def __init__(self):
#         self.__age = 18
#
#     def set_age(self, value):
#         self.__age = value
#
#     def get_age(self):
#         return self.__age
#
#     age = property(get_age, set_age)
#
#
# p = Person()
# print(p.age)
# p.age = 20
# print(p.age)


class Age(object):
    def __init__(self):
        print("init")
        self.val = 0
        pass
    def __get__(self, instance, owner):
        print("get")
        return self.val
        pass
    def __set__(self, instance, value):
        print("set")
        self.val = value
        pass
    def __delete__(self, instance):
        print("delete")
        del self.val
        pass


class Person(object):
    yangli_age = Age()
    yingli_age = Age()

p = Person()
p.yangli_age = 21
p.yingli_age = 20
print(p.yangli_age, p.yingli_age)
