"""
    test
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 创建字典
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
num_dict = {n: i for i, n in enumerate(char_arr)}

# 网络参数
n_step = 5
n_hidden = 128
n_class = len(num_dict)
batch_size = len(seq_data)
NUM_LAYERS = 2


# 准备数据
def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))
        input = [num_dict[n] for n in seq[0]]
        ouput = [num_dict[n] for n in ('S' + seq[1])]
        target = [num_dict[n] for n in (seq[1]) + 'E']

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[ouput])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))


input_batch, output_batch, target_batch = make_batch(seq_data)


# 创建网络
class Seq2Seq(nn.Module):
    """
    要点：
    1.该网络包含一个encoder和一个decoder，使用的RNN的结构相同，最后使用全连接接预测结果
    2.RNN网络结构要熟知
    3.seq2seq的精髓：encoder层生成的参数作为decoder层的输入
    """

    def __init__(self):
        super().__init__()
        # 此处的input_size是每一个节点可接纳的状态，hidden_size是隐藏节点的维度
        self.enc = nn.RNN(input_size=n_class, hidden_size=n_hidden, num_layers=NUM_LAYERS, dropout=0.5)
        self.dec = nn.RNN(input_size=n_class, hidden_size=n_hidden, num_layers=NUM_LAYERS, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # RNN要求输入：(seq_len, batch_size, n_class)，这里需要转置一下
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)
        _, enc_states = self.enc(enc_input, enc_hidden)
        outputs, _ = self.dec(dec_input, enc_states)
        pred = self.fc(outputs)

        return pred


# training
model = Seq2Seq()
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    hidden = Variable(torch.zeros(NUM_LAYERS, batch_size, n_hidden))

    optimizer.zero_grad()
    pred = model(input_batch, hidden, output_batch)
    pred = pred.transpose(0, 1)
    loss = 0
    for i in range(len(seq_data)):
        temp = pred[i]
        tar = target_batch[i]
        loss += loss_fun(pred[i], target_batch[i])
    if (epoch + 1) % 1000 == 0:
        print('Epoch: %d   Cost: %f' % (epoch + 1, loss))
    loss.backward()
    optimizer.step()


# 测试
def translate(word):
    input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])
    # hidden 形状 (2, 1, n_class)
    hidden = Variable(torch.zeros(NUM_LAYERS, 1, n_hidden))
    # output 形状（6，1， n_class)
    output = model(input_batch, hidden, output_batch)
    predict = output.data.max(2, keepdim=True)[1]
    decoded = [char_arr[i] for i in predict]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated.replace('P', '')


print('girl ->', translate('girl'))
