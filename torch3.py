# encoding=GBK

"""
如和使用梯度下降法来求解二元一次方程组的参数
根据给出的一组观测点来求得最优的方程参数 w,b
"""

import numpy as np

def calc_loss(b, w, points):
    """
    根据给出的参数ｂ和ｗ还有观测点，来计算方程到观测点的误差.
    这里假设的原函数是 y = w x + b
    :param b:  　   常数项
    :param w: 　    自变量系数
    :param points:  观测到的所有点的 list
    :return:        返回对于所有观测点的平均误差
    """
    total_error = 0                      # 总误差
    for i in range(0, len(points)):     # 遍历所有的观测点
        x = points[i, 0]                # 取得观测点的ｘ值
        y = points[i, 1]                # 取得观测点的ｙ值
        total_error = (w*x + b - y) ** 2 # 计算误差的平方和

    return total_error/len(points)       # 误差的均值


def step_gradient(b_current, w_current, points, learning_rate):
    """
    对损失函数 (wx+b-y)**2 输入所有观测点，对ｗ和ｂ进行平均梯度逼近：
    主要的优化方式就是：
        让损失函数对 w 和 b 分别求偏导, 得到了在ｗ和ｂ方向上的偏导函数
            对ｗ求偏导 = 2(wx+b-y) * x
            对b求偏导 = 2(wx+b-y)
        这就是损失函数的ｗ和ｂ的梯度向量
        依次把观测点数据带入，就是在观测点处的梯度向量。
            也就是在观测点处的梯度下降最快方向的向量。
            每次把ｗ 和 b 的在梯度方向上值加起来，
        遍历完所有的点之后就得到总的梯度。
        把这个梯度总和求平均值。得到的就是所有点梯度计算出来平均的方向
        再乘以learning_rate,就是取这个方向上一个非常小的步长，免得跑过头了
        用ｗ和ｂ减去这个非常小的变化量。逼近一次。
    :param b_current:       输入的 b 值。可以是上一轮的ｂ
    :param w_current:       输入的 w 值。可以是上一轮的ｗ
    :param points:          全部的观测点
    :param learning_rate:   定义的步长
    :return:                优化过的 b,w 值
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
    多跑几轮，每一轮都是全部观测点来逼近gradient的函数
    :param points:          输入的点坐标集合
    :param starting_b:      起始给出的ｂ
    :param starting_w:      起始给出的ｗ
    :param learning_rate:   步长
    :param num_iterations:  迭代次数
    :return:                优化计算以后得到的ｂ和ｗ
    """
    b = starting_b
    w = starting_w

    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)

    return [b, w]


def gen_data():
    """
    生成原始点的数据。
    这里是一个二维的曲线，加上了噪音
    :return:    返回一个１０００行２列的array,每一行就是一个点的坐标
    """
    x = np.random.randint(100, size=(1000,1))        # 在０～１０范围内的１０００行１列的随机数
    y = 1.47*x + 0.03 + np.random.rand(1000, 1)     # 在后面增加一个符合(0,1)之间正态分布的随机噪音项

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