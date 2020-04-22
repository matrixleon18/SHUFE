


class MYException(Exception):
    def __init__(self, error_info):
        super().__init__(self)
        self.errorinfo = error_info
    def __str__(self):
        return self.errorinfo
    pass

try:
    # do something
    raise MYException("my exception")
except MYException as e:
    print(e)
    print(e.errorinfo)
finally:
    print("Finally")


try:
    # do something
    raise MYException(30)
except MYException as e:
    print("num :"+ str(e.errorinfo) + " is wrong")
finally:
    print("Yes. It's wrong")