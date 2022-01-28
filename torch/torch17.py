# encoding=GBK

"""
强化学习
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


# 超参数
BATCH_SIZE = 32             # 每一批训练数据的大小
LR = 0.001                  # 学习效率
EPSILON = 0.9               # 贪心策略
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # 目标更新的频率
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]       # 环境的状态4个：小车位置，小车速率，杆子角度，杆子角速度


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)      # 定义神经网络来输入观测到到状态
        self.fc1.weight.data.normal_(0, 0.1)    # 用正态分布来初始化权重
        self.out = nn.Linear(10, N_ACTIONS)     # 定义神经网络来输出的是下一步的动作
        self.fc1.weight.data.normal_(0, 0.1)    # 用正态分布来初始化权重

    def forward(self, x):
        x = self.fc1(x)                         # 输入
        x = F.relu(x)                           # 激励函数
        action_value = self.out(x)              # 产生输出动作
        return action_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0         # 学了多少步的计数器
        self.memory_counter = 0
        self.memory = np.zeros(MEMORY_CAPACITY, N_STATES * 2 + 2)  # 初始化记忆库来记录所有的状态；加上一个Action一个Reward
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        pass

    def choose_action(self, x):
        """
        选择下一步的动作
        因为不做反向传播，所以借用forward求出x状态下的最优动作
        :param x: 是输入的环境变量
        :return:
        """
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:       # 90%的概率根据以往经验选取最优解
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0,0]
        else:                                   # 10%的概率随机探索其他解是否更好
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_):
        """
        记录环境信息
        :param s:   状态 state
        :param a:   动作 action
        :param r:   环境反馈的奖励 reward
        :param s_:  下一个状态 next_state
        :return:
        """
        transition = np.hstack(self, s, a, r, _s)
        # replace the memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY   # 如果transition数量超过memory内存，就重新开始覆盖掉之前掉记忆数据
        self.memory[index, :] = transition
        self.memory_counter += 1
        pass

    def learn(self):
        """
        大脑学习的过程
        :return:
        """
        # 检查是否需要更新目标网络
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[: -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)


dqn = DQN()

print("Collecting experience...")

for i_episode in ranger(400):
    s = env.reset()                                 # 环境重启

    while True:
        env.render()                                # 绘制环境
        a = dqn.choose_action(s)                    # 根据状态选择动作
        s_, r, done, info = env.step(a)             # 环境根据我的动作给出的反馈
        dqn.store_transition(s, a, r, s_)           # dqn存储状态和动作等信息

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:                                    # 这个回合结束了。跳出这个回合的循环
            break

        s = s_                                      # 把这次得到的下个状态赋值给下一次起始状态