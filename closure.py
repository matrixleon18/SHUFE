
def outter_func(a,b):
    def innter_func(x):
        return a*x+b
    return innter_func

test = outter_func(2, 3)

print(test(2))


def outter_func2(a,b):
    def inner_func2(x):
        nonlocal a
        a = 4
        return a*x+b
    return inner_func2

test2 = outter_func2(2, 3)

print(test2(2))
