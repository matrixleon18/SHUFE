# encoding=GBK

"""
�쳣����Ļ���ʹ�÷���
"""


class MYException(Exception):                           # ���쳣����̳�һ���쳣
    def __init__(self, error_info):
        super().__init__(self)                          # Ҫ���ø���ĳ�ʼ������
        self.errorinfo = error_info

    def __str__(self):
        return self.errorinfo
    pass


try:
    # do something                                      # �����Ҫʵ�ֵĹ���
    raise MYException("my exception")                   # ���������Ͳ���һ���쳣
except MYException as e:                                # ��׽�쳣
    print(e)                                            # ��ӡ���쳣��Ϣ
    print(e.errorinfo)
finally:                                                # �����Ƿ��쳣���ᵽ����
    print("Finally")


try:
    # do something
    raise MYException(30)
except MYException as e:
    print("num :"+ str(e.errorinfo) + " is wrong")
finally:
    print("Yes. It's wrong")