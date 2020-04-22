
# basic data type
# int
print(int(32))
# float
print(float(32))
# string
print(str(32))
# bool: True/False
print(bool(32))
# nothing
print()
# bool
print(32 == 0)
# bool
print(32 is 0)
# bool
print(32 and True)
print(32 and False)
print(0 and True)
print(0 and False)

# Tuple
a = (1, 2, 3)
# index start from 0
print(a.index(2))

# list
a = [1, 2, 3, 4, 5]
b = list([1, 2, 3, 4, 5])
print(a)
print(b)

# string
b = 'a33bcde'
print(b)
# count how many char/str in the string
print(b.count('3'))
# find 1st char/str position in the string
print(b.index('3'))

# convert string to tuple
c = tuple('a33bcde')
print(c)

# convert string to list
print(list(b))

# dict
f = dict({'k1': 1, 'k2': 2, 'k3': 3})
g = {'k1': 1, 'k2': 2, 'k3': 3}
# they are same
print(f)
print(g)
