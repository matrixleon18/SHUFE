# encoding=GBK

"""
自动求导(Autograd)原理解析
"""

import torch
from torch.autograd.variable import Variable

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a + b
d = torch.tensor(3.0, requires_grad=True)
e = c * d

print("=================================================")
print("backward()前：")
print("a.grad: {}".format(a.grad))
print("b.grad: {}".format(b.grad))
print("c.grad: {}".format(c.grad))
print("d.grad: {}".format(d.grad))
print("e.grad: {}".format(e.grad))
e.backward()
print("backward()后：")
print("a.grad: {}".format(a.grad))
print("b.grad: {}".format(b.grad))
print("c.grad: {}".format(c.grad))
print("d.grad: {}".format(d.grad))
print("e.grad: {}".format(e.grad))

print("=================================================")
print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
print("d: {}".format(d))
print("e: {}".format(e))

print("=================================================")
print("a.grad_fn: {}".format(a.grad_fn))
print("b.grad_fn: {}".format(b.grad_fn))
print("c.grad_fn: {}".format(c.grad_fn))
print("d.grad_fn: {}".format(d.grad_fn))
print("e.grad_fn: {}".format(e.grad_fn))

print("=================================================")
print("type(e.grad_fn): {}".format(type(e.grad_fn)))
print("dir(e.grad_fn): {}".format(dir(e.grad_fn)))
print("e.grad_fn.next_functions: {}".format(e.grad_fn.next_functions))
print("dir(e.grad_fn.next_functions[1][0]): {})".format(dir(e.grad_fn.next_functions[1][0])))
print("e.grad_fn.next_functions[1][0].variable: {}".format(e.grad_fn.next_functions[1][0].variable))

print("=================================================")
print("这个 variable 就是 d")
print(id(e.grad_fn.next_functions[1][0].variable))
print(id(d))

print("=================================================")
print(e)
print(e.grad_fn)
print(e.grad_fn.next_functions)

print("=================================================")
((f1, _), (f2, _)) = e.grad_fn.next_functions
print(f1)                                       # 这是 c.grad_fn
print(f2)
print(f2.variable)