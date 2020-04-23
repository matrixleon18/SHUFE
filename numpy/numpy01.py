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

print(np.random.randint(low=0, high=5, size=(2, 3), dtype='I'))       # ��[low, high)֮��ȡֵ����ָ����״�Ķ���

print(np.random.random(size=(2, 3)))      # ���������������ndarray

print(np.random.choice((4, 3, 2, 1), size=None, replace=True, p=None))       # �������ﰴ��p�ĸ���ѡ��

print(np.random.seed(100))  # ָ������������ÿ��random�����������һ�¡�������֤�Ͳ��ԡ�

"""
numpy.eye(N, M=None, k=0, dtype=<class��float��>, order=��C��)
N: ����
M��������Ĭ�ϵ���N
k���Խ���������0��Ĭ�ϣ�Ϊ���Խ��ߣ���ֵ��ָ�϶Խ��ߣ���ֵ��ָ���¶Խ���
order: {'C', 'F'}����Ƿ�Ӧ�洢����Ҫ�У�C��ʽ���л��ڴ��е�������Fortran��ʽ��˳��
"""
print(np.eye(5))     # ����������һ��sizeΪ5�ĵ�λ����

print(np.linspace(start=0, stop=10, num=5))         # ��[start, stop]֮�䴴��5��������ȵ����ֵ�array

# 1��һ������ʱ������ֵΪ�յ㣬���ȡĬ��ֵ0������ȡĬ��ֵ1��
# 2����������ʱ����һ������Ϊ��㣬�ڶ�������Ϊ�յ㣬����ȡĬ��ֵ1��
# 3����������ʱ����һ������Ϊ��㣬�ڶ�������Ϊ�յ㣬����������Ϊ���������в���֧��С��
print(np.arange(0, 100, 3))

np.arange(0, 100, 3).tofile('tmp.txt')           # �Ѳ�����arrayд�뵽�ļ���

print(np.fromfile('tmp.txt', dtype=np.int))      # ���ļ��ж�ȡarray��һ��Ҫָ���������ͣ�����һ����ȷ��Ĭ��float

