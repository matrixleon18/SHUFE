# encoding=GBK

"""
ǿ��ѧϰ
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


# ������
BATCH_SIZE = 32             # ÿһ��ѵ�����ݵĴ�С
LR = 0.001                  # ѧϰЧ��
EPSILON = 0.9               # ̰�Ĳ���
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # Ŀ����µ�Ƶ��
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]       # ������״̬4����С��λ�ã�С�����ʣ����ӽǶȣ����ӽ��ٶ�


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)      # ����������������۲⵽��״̬
        self.fc1.weight.data.normal_(0, 0.1)    # ����̬�ֲ�����ʼ��Ȩ��
        self.out = nn.Linear(10, N_ACTIONS)     # ���������������������һ���Ķ���
        self.fc1.weight.data.normal_(0, 0.1)    # ����̬�ֲ�����ʼ��Ȩ��

    def forward(self, x):
        x = self.fc1(x)                         # ����
        x = F.relu(x)                           # ��������
        action_value = self.out(x)              # �����������
        return action_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0         # ѧ�˶��ٲ��ļ�����
        self.memory_counter = 0
        self.memory = np.zeros(MEMORY_CAPACITY, N_STATES * 2 + 2)  # ��ʼ�����������¼���е�״̬������һ��Actionһ��Reward
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        pass

    def choose_action(self, x):
        """
        ѡ����һ���Ķ���
        ��Ϊ�������򴫲������Խ���forward���x״̬�µ����Ŷ���
        :param x: ������Ļ�������
        :return:
        """
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:       # 90%�ĸ��ʸ�����������ѡȡ���Ž�
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0,0]
        else:                                   # 10%�ĸ������̽���������Ƿ����
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_):
        """
        ��¼������Ϣ
        :param s:   ״̬ state
        :param a:   ���� action
        :param r:   ���������Ľ��� reward
        :param s_:  ��һ��״̬ next_state
        :return:
        """
        transition = np.hstack(self, s, a, r, _s)
        # replace the memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY   # ���transition��������memory�ڴ棬�����¿�ʼ���ǵ�֮ǰ����������
        self.memory[index, :] = transition
        self.memory_counter += 1
        pass

    def learn(self):
        """
        ����ѧϰ�Ĺ���
        :return:
        """
        # ����Ƿ���Ҫ����Ŀ������
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
    s = env.reset()                                 # ��������

    while True:
        env.render()                                # ���ƻ���
        a = dqn.choose_action(s)                    # ����״̬ѡ����
        s_, r, done, info = env.step(a)             # ���������ҵĶ��������ķ���
        dqn.store_transition(s, a, r, s_)           # dqn�洢״̬�Ͷ�������Ϣ

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:                                    # ����غϽ����ˡ���������غϵ�ѭ��
            break

        s = s_                                      # ����εõ����¸�״̬��ֵ����һ����ʼ״̬