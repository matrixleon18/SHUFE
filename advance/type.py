def init(self):
    pass


def set_name(self, value):
    self.name = value


def get_name(self):
    return self.name


MyClass = type("TypeExample", (object,), {"__init__": init, "setname": set_name, "getname": get_name})

p = MyClass()

p.setname("TEST")
print(p.getname())
print(type(p))
print(type(MyClass))
