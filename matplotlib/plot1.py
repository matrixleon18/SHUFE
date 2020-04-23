# encoding=GBK

"""
����maptlotlib�Ļ����÷�
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6), dpi=80)    # ����һ�� 8*6 �㣨point����ͼ�������÷ֱ���Ϊ 80

plt.subplot(1, 1, 1)                        # ����һ���µ� 1*1 ����ͼ����������ͼ�����������еĵ� 1 �飨Ҳ��Ψһ��һ�飩

X = np.linspace(                            # ����һ���Ȳ�����
    -np.pi,                                 # ���е����� -3.14 ��ʼ
    np.pi,                                  # ���е��յ㵽 +3.14 ����
    256,                                    # ������������ 256 ����
    endpoint=True                           # ������棬��һ������stop�����ΪFalse��һ��������stop
)

C, S = np.cos(X), np.sin(X)                 # �ֱ��X�� cos,sin ��ֵ���� C,S

plt.plot(                                   # ��ʼ����cos��
    X,                                      # ���xֵ
    C,                                      # ���yֵ
    color='blue',                           # ����ɫ
    linewidth=1.0,                          # �ߵĿ�ȡ�float
    linestyle='-'                           # ʵ��/����/�� {��-��, ���C-��, ��:��, ��-.��, ��None��, ' ', '', 'solid', 'dotted' ��}
)

plt.plot(                                   # ��ʼ����sin��
    X,                                      # ���xֵ
    S,                                      # ���yֵ
    color='green',                          # ����ɫ
    linewidth=1.0,                          # �߿�1.0
    linestyle='-'                           # ʵ��
)

plt.xlim(                                   # ���ú����������
    -4.0,                                   # ����� -4 ��ʼ
    4.0                                     # ���굽 4 ����
)

plt.xticks(                                 # ���ú������ʾ�̶�
    np.linspace(-4, 4, 9, endpoint=True)    # ��-4��4��ʾ9���̶�
)

plt.ylim(                                   # ���������������
    -1.0,                                   # ����� -1 ��ʼ
    1.0                                     # ���굽  1 ����
)

plt.yticks(                                 # �����������ʾ�̶�
    np.linspace(-1, 1, 5, endpoint=True)    # ��-1��1��ʾ5���̶�
)

plt.savefig("exercice.png", dpi=72)         # ��ͼ����ļ�

plt.show()                                  # ����Ļ����ʾ