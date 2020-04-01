# encoding=GBK


#               房型图
##################################################
#
#    -------------+------------+
#   |             |                  5
#   |     0       |      1     +------------+
#   |             |            |            |
#   +-------+   +-+--+  +------+            |
#   |             |            |      2     |
#   |     4       |      3                  |
#   |                          |            |
#   |             |            |            |
#   +--+   +------+------------+------------+
#         5
#
#              　状态／行动图
##################################################
#                    Action
#     state    0   1   2   3   4   5
#       0    |                 0      |
#       1    |             0      100 |
#   R = 2    |             0          |
#       3    |     0   0       0      |
#       4    | 0           0      100 |
#       5    |     0           0  100 |
#
##################################################


import numpy as np

q = np.zeros((6, 6))            # 这就是要求的ｑ矩阵
rewards = np.zeros((6, 6))      # 回报矩阵
rewards[:, 5] = 500             # 最终到房间 5 的回报最高
actions = [[4],                 # 房间 0 只能进入房间 4
           [3, 5],              # 房间 1 只能进入房间 3, 5
           [3],                 # 房间 2 只能进入房间 3
           [1, 2, 4],           # 房间 3 只能进入房间 1, 2, 4
           [0, 3, 5],           # 房间 4 只能进入房间 0, 3, 5
           [1, 4, 5]]           # 房间 5 只能进入房间 1, 4, 5
gama = 0.8                      # 折现系数为0.8
EPOCH = 10


def train():                                            # 训练模型
    s = np.random.randint(6)                            # 开始时候随机给出一个房间号

    while s < 5:
        a = np.random.choice(actions[s])                # 从对应的房间号里随机选一个动作
        s1 = a
        q[s, a] = rewards[s, a] + gama * q[s1].max()    # 这就是转移规则
        s = s1


def test(s):
    print(q)
    print("=> {}".format(s))
    while s < 5:
        s = q[s].argmax()                       # 得到最大值的参数
        print("=> {}".format(s))

for _ in range(EPOCH):         # 迭代训练
    train()

test(0)                 # 看看测试的路径结果
