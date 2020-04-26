

# class MyClass(object):
#     def __init__(self):
#         pass
#
#     @property
#     def age(self):
#         return self._age
#
#     @age.setter
#     def age(self, value):
#         self._age = value
#
# p = MyClass()
#
# p.age = 100
# print(p.age)
# p.age = 1
# print(p.age)


# class Student(object):
#     def __init__(self):
#         pass
#
#     @property
#     def score(self):
#         return self._score
#
#     @score.setter
#     def score(self, value):
#         self._score = value
#
#
# p = Student()
# p.score = 100
# print(p.score)


# class Student(object):
#     def __init__(self):
#         pass
#
#     @property
#     def score(self):
#         return self._score
#
#     @score.setter
#     def score(self, value):
#         self._score = value
#
#
# p = Student()
# p.score = 100
# print(p.score)


# class Student(object):
#     def __int__(self, value):
#         self._score = 60
#         pass
#
#     @property
#     def score(self):
#         return self._score
#
#     @score.setter
#     def score(self, value):
#         self._score = value
#
#     @score.deleter
#     def score(self):
#         del self._score
#
#
# p1 = Student()
# p1.score = 100
# print(p1.score)
#
# p2 = Student()
# p2.score = 60
# print(p2.score)
# print(p1.score)




class MyClass(object):
    def __init__(self):
        self._name = ""
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.getter
    def name(self):
        return self._name


p = MyClass()
p.name = "AA"
print(p.name)
