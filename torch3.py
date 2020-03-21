# encoding=GBK

"""
���ʹ���ݶ��½���������Ԫһ�η�����Ĳ���
���ݸ�����һ��۲����������ŵķ��̲��� w,b
"""

import numpy as np

def calc_loss(b, w, points):
    """
    ���ݸ����Ĳ�����ͣ����й۲�㣬�����㷽�̵��۲������.
    ��������ԭ������ y = w x + b
    :param b:  ��   ������
    :param w: ��    �Ա���ϵ��
    :param points:  �۲⵽�����е�� list
    :return:        ���ض������й۲���ƽ�����
    """
    total_error = 0                      # �����
    for i in range(0, len(points)):     # �������еĹ۲��
        x = points[i, 0]                # ȡ�ù۲��ģ�ֵ
        y = points[i, 1]                # ȡ�ù۲��ģ�ֵ
        total_error = (w*x + b - y) ** 2 # ��������ƽ����

    return total_error/len(points)       # ���ľ�ֵ


def step_gradient(b_current, w_current, points, learning_rate):
    """
    ����ʧ���� (wx+b-y)**2 �������й۲�㣬�ԣ��ͣ����ƽ���ݶȱƽ���
    ��Ҫ���Ż���ʽ���ǣ�
        ����ʧ������ w �� b �ֱ���ƫ��, �õ����ڣ��ͣⷽ���ϵ�ƫ������
            �ԣ���ƫ�� = 2(wx+b-y) * x
            ��b��ƫ�� = 2(wx+b-y)
        �������ʧ�����ģ��ͣ���ݶ�����
        ���ΰѹ۲�����ݴ��룬�����ڹ۲�㴦���ݶ�������
            Ҳ�����ڹ۲�㴦���ݶ��½���췽���������
            ÿ�ΰѣ� �� b �����ݶȷ�����ֵ��������
        ���������еĵ�֮��͵õ��ܵ��ݶȡ�
        ������ݶ��ܺ���ƽ��ֵ���õ��ľ������е��ݶȼ������ƽ���ķ���
        �ٳ���learning_rate,����ȡ���������һ���ǳ�С�Ĳ���������ܹ�ͷ��
        �ã��ͣ��ȥ����ǳ�С�ı仯�����ƽ�һ�Ρ�
    :param b_current:       ����� b ֵ����������һ�ֵģ�
    :param w_current:       ����� w ֵ����������һ�ֵģ�
    :param points:          ȫ���Ĺ۲��
    :param learning_rate:   ����Ĳ���
    :return:                �Ż����� b,w ֵ
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 * (w_current * x + b_current - y))
        w_gradient += (2 * (w_current * x + b_current - y) * x)

    new_b = b_current - learning_rate * b_gradient / N
    new_w = w_current - learning_rate * w_gradient / N

    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    """
    ���ܼ��֣�ÿһ�ֶ���ȫ���۲�����ƽ�gradient�ĺ���
    :param points:          ����ĵ����꼯��
    :param starting_b:      ��ʼ�����ģ�
    :param starting_w:      ��ʼ�����ģ�
    :param learning_rate:   ����
    :param num_iterations:  ��������
    :return:                �Ż������Ժ�õ��ģ�ͣ�
    """
    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)

    return [b, w]


def gen_data():
    """
    ����ԭʼ������ݡ�
    ������һ����ά�����ߣ�����������
    :return:    ����һ�����������У��е�array,ÿһ�о���һ���������
    """
    x = np.random.randint(100, size=(1000,1))        # �ڣ���������Χ�ڵģ��������У��е������
    y = 1.47*x + 0.03 + np.random.rand(1000, 1)     # �ں�������һ������(0,1)֮����̬�ֲ������������

    return np.hstack((x, y))


def run():
    points = gen_data()
    learning_rate = 0.0001
    init_b = 0
    init_w = 0
    num_itera = 10000

    [b, w] = gradient_descent_runner(points, init_b, init_w, learning_rate, num_itera)

    print("b is : {}  w is : {}".format(b, w))

    _loss = calc_loss(b, w, points)

    print("loss is : ", _loss)


if __name__ == '__main__':
    run()