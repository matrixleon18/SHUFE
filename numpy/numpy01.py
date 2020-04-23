# encoding=GBK

"""
NumPy��Python�п�ѧ����Ļ���������� ����һ���ṩ����ά������󣬶������������磺�������顢�����Լ����ڿ��ٲ�������ĺ�����API��
��������ѧ���߼���������״�任������ѡ��I/O ����ɢ����Ҷ�任���������Դ���������ͳ�����㡢���ģ��ȵȡ�
NumPy�еĺ�����ndarray����ndarray�����Ǿ�����ͬ���ͺʹ�С��ͨ���ǹ̶���С����Ŀ�Ķ�ά���������Զ��������������Ƭ����״�任�Ȳ�����

ndarry����ʹ�÷���
"""

import numpy as np

data = [[1, 2, 3],               # �����һ��list����Ȼ�������������
       [4, 5, 6]]

nd_arr = np.array(data)          # ת����ndarray

print(nd_arr)                   # ���ھ��Ǹ�2x3������
print(type(data))                # <class 'list'>  ����Ǹ�list
print(type(nd_arr))             # <class 'numpy.ndarray'> ����ndarray����


# ndarray���������
print(nd_arr.ndim)              # ����γ�ȵĸ�����γ�ȱ���Ϊ��rank ���ȣ�

print(nd_arr.shape)             # �������״��Ҳ���Ǹ���γ��������Ĵ�С��tuple��ʽ

print(nd_arr.size)              # ������Ԫ��������Ҳ����shape��tupleÿ�����ĳ˻�

print(nd_arr.dtype)             # ������Ԫ������

print(nd_arr.itemsize)          # ������ÿ��Ԫ�ص��ֽڴ�С

print(nd_arr.real)              # ����ÿ��Ԫ��ֻȡʵ������

print(nd_arr.imag)              # ����ÿ��Ԫ��ֻȡ��������


# ndarray������ʽ

data = [[1, 2, 3], [4, 5, 6]]

nd_arr = np.array(data)         # �������list, tuple, �����������������ת����ndaray����
print(nd_arr)

print(np.asarray(nd_arr))       # array��asarray�����Խ����ݽṹת����ndarray��
                                # �����ǵ�����ndarryʱ��array�����ɳ�һ���ڴ����copy��asarray���᣻

print(np.zeros([2, 3]))         # ����һ��ȫ��0��ndarray������listָ��ÿ��rank��shape

print(np.zeros_like(data))      # ����һ�����к�dataһ��shape��ȫ��0��ndarray

print(np.ones([2, 3]))          # ����һ��ȫ��1��ndarray�������listָ����ÿ��rank��shape

print(np.ones_like(data))       # ����һ�����к�dataһ��shape��ȫ��1��ndarray

print(np.empty([2, 3]))         # ����ָ������״����ndarray���󣬸þ���������Ԫ�����

print(np.empty_like(data))      # ͬ��

print(np.full([2, 3], fill_value='a'))  # ����һ��ָ����״��ndarray���󣬲�����ָ��ֵ�������

print(np.full_like(data, 8))    # ͬ��

print(np.identity(4))           # ����һ��nxn�ĵ�λ����

print(np.random.rand(2, 3))     # ����һ��ָ����״��ndarray���󣬶������е�ֵ��[0,1)֮��������

print(np.random.randn(2, 3))    # ����һ��ָ����״��ndarray���󣬶������е�ֵ�Ƿ��ӱ�׼��̬�ֲ��������





