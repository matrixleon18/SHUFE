
# def num(numnum):
#     def func_num_in(in_num):
#         return numnum + in_num
#     return func_num_in
#
# a = num(100)
# b = a(100)
# print(a)
# print(b)


def test(name):
    def pre_name(prefix):
        return name+prefix

    return pre_name

t = test("Tom")
x = t("Jerry")
print(x)