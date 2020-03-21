# encoding=GBK

def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1


fab(5)


def fab2(max):
    n, a, b = 0, 0, 1
    L = []
    while n < max:
        L.append(b)
        a, b = b, a + b
        n = n + 1
    return L


print(fab2(5))


class Fab3(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()


for i in Fab3(5):
    print(i)


def fab4(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        # print(b)
        a, b = b, a + b
        n = n + 1


for n in fab4(5):
    print(n)

print("=======yield========")
t = fab4(5)
print(t)
print(next(t))
print(next(t))
print(next(t))
print(next(t))

print("========iterator========")
mylist = [x*x for x in range(3)]
for i in mylist:
    print(i)
for i in mylist:
    print(i)

print("===========generator======")
mygenerator = (x*x for x in range(3))
for i in mygenerator:
    print(i)
for i in mygenerator:
    print(i)

print("===========yield======")
def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator()
print(mygenerator)
for i in mygenerator:
    print(i)
for i in mygenerator:
    print(i)


print("========yield2=========")
def test():
    print("Start")
    m = yield 2
    print(m)
    n = yield 12
    print(n)
    x = yield 13
    print("Done")

c = test()
print(c)
m = next(c)
n = c.send("send")
x = c.send("send2")

print(m)
print(n)
print(x)