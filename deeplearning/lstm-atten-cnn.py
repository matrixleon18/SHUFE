# encoding=GBK
# 准备数据
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import random
import math
import matplotlib.pyplot as plt

np.random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed(3)

torch.backends.cudnn.deterministic = True

# 设置 GPU 优先
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义 PositionEncoding 模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim_size, num_layers, output_dim_size, seq_len, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_dim_size
        self.max_seq_len = seq_len
        self.num_layers = num_layers
        self.input_dim_size = input_size
        self.output_dim_size = output_dim_size

        self.post=PositionalEncoding(d_model=self.input_dim_size, dropout=0.1, max_len=self.max_seq_len)

        self.conv1 = nn.Conv1d(self.input_dim_size, 512, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(self.input_dim_size, 512, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(self.input_dim_size, 512, kernel_size=7, padding='same')
        self.conv4 = nn.Conv1d(self.input_dim_size, 512, kernel_size=9, padding='same')
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

        self.conv01 = nn.Conv1d(self.input_dim_size, 256, kernel_size=3, padding='same')
        self.conv02 = nn.Conv1d(self.input_dim_size, 256, kernel_size=5, padding='same')
        self.conv03 = nn.Conv1d(self.input_dim_size, 256, kernel_size=7, padding='same')
        self.conv04 = nn.Conv1d(self.input_dim_size, 256, kernel_size=9, padding='same')

        self.conv31 = nn.Conv1d(self.hidden_size, 512, kernel_size=3, padding='same')
        self.conv32 = nn.Conv1d(self.hidden_size, 512, kernel_size=5, padding='same')
        self.conv33 = nn.Conv1d(self.hidden_size, 512, kernel_size=7, padding='same')
        self.conv34 = nn.Conv1d(self.hidden_size, 512, kernel_size=9, padding='same')
        self.linear35 = nn.Linear(64, self.output_dim_size)    # 这个是给最后的多头注意力再乘 W0 的

        self.num_attention_head = 256                                                                 # 64
        self.attention_head_size = int(self.hidden_size/self.num_attention_head)                      # 每个头是32个维度
        self.all_head_size = self.num_attention_head * self.attention_head_size                       # 所有的头的维度合集

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

        self.lstm1 = nn.LSTM(input_size=self.input_dim_size,       hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout)

        self.linear_1 = nn.Linear(self.input_dim_size, self.hidden_size)          # 这个是将输入的维度转换成 hidden_dim 的
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)             # 这个是给最后的多头注意力再乘 W0 的
        self.linear_3 = nn.Linear(self.hidden_size, self.output_dim_size)         # 这个是最后输出时将每个element的hidden_dim转成需要的out_dim
        self.linear_4 = nn.Linear(self.max_seq_len, 1)                            # 这个是最后输出时将整个seq的输出转成一个值

        self.relu = nn.LeakyReLU()                                       # 用 relu 来增强模型非线性
        self.tanh = nn.Tanh()

        # self.query = nn.Linear(self.hidden_size, self.hidden_size)     # 输入768， 输出多头的维度总数。这里还是768.
        # self.key = nn.Linear(self.hidden_size, self.hidden_size)       # 输入768， 输出多头的维度总数。这里还是768.
        # self.value = nn.Linear(self.hidden_size, self.hidden_size)     # 输入768， 输出多头的维度总数。这里还是768.

        self.query = nn.Linear(self.hidden_size, self.all_head_size)     # 输入768， 输出多头的维度总数。这里还是768.
        self.key = nn.Linear(self.hidden_size, self.all_head_size)       # 输入768， 输出多头的维度总数。这里还是768.
        self.value = nn.Linear(self.hidden_size, self.all_head_size)     # 输入768， 输出多头的维度总数。这里还是768.

        self.init_weights3()

    def init_weights1(self):
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def init_weights2(self):
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)

    def init_weights3(self):
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def init_weights4(self):
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, mean=0, std=1)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, mean=0, std=1)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def attention_net(self, in_value, mask=None):
        # print(lstm_output.size())                                                   # [batch_size, seq_len, hidden_dim_size]
        # print(query.size())                                                         # [batch_size, seq_len, hidden_dim_size]
        # key   = lstm_output.transpose(1,2)
        # query = lstm_output
        # value = lstm_output
        batch_size = in_value.shape[0]
        seq_len = in_value.shape[1]
        hidden_dim = in_value.shape[2]

        # Q = self.query(in_value)
        # K = self.key(in_value)
        # V = self.value(in_value)
        # V = in_value

        # Q : [batch_size, seq_len, hidden_dim] ==> [batch_size, seq_len, num_head, head_size] ==> [batch_size, num_head, seq_len, head_size]
        Q = self.query(in_value).reshape(batch_size, seq_len, self.num_attention_head, self.attention_head_size).permute(0, 2, 1, 3)      # 先将 hidden_dim 切成 num_head * head_size ，再将 num_head 和 seq_len互换
        K = self.key(in_value).reshape(batch_size, seq_len, self.num_attention_head, self.attention_head_size).permute(0, 2, 1, 3)        # 先将 hidden_dim 切成 num_head * head_size ，再将 num_head 和 seq_len互换
        V = self.value(in_value).reshape(batch_size, seq_len, self.num_attention_head, self.attention_head_size).permute(0, 2, 1, 3)      # 先将 hidden_dim 切成 num_head * head_size ，再将 num_head 和 seq_len互换

        # d_k = Q.size(-1)                                                                            # d_k为query的维度。避免概率接近0

        # attention_scores = torch.matmul(query, lstm_output.transpose(1, 2)) / math.sqrt(d_k)     #打分机制  [batch_size, seq_len, hid_dim] * [batch_size, hid_dim, seq_len] ==> scores:[batch_size, seq_len, seq_len], 每个值就是两个输入x元素的相似性
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))                                    # [batch_size, num_head, seq_len, head_size] * [batch_size, num_head, head_size, seq_len] ==> [batch_size, num_head, seq_len, seq_len]

        # attention_scores = attention_scores / math.sqrt(d_k)                                       # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)                  # 多头机制下，hidden-dim被划分为 num_head个区域，所以现在要除的就是每个小区域的维度开方

        # alpha = F.softmax(attention_scores, dim = -1)                                            #对最后一个维度归一化得分  [batch_size, seq_len, seq_len] 保证相似性在一行上归一了。
        alpha = nn.Softmax(dim=-1)(attention_scores)                                               # 因为alpha是方阵，0维的seq_len就是真正的序列长度，1维的seq_len是对应每一个element和序列元素相关性。

        # alpha = self.dropout(alpha)

        attention = torch.matmul(alpha, V)                                            # [batch_size, num_head, seq_len, seq_len] * [batch_size, num_head, seq_len, head_size] = [batch_size, num_head, seq_len, head_size]

        attention = attention.permute(0, 2, 1, 3).contiguous()                       # [batch_size, num_head, seq_len, head_size] ==> [batch_size, seq_len, num_head, head_size]
        # new_attention_shape = attention.size()[:-2] + (self.all_head_size,)
        # attention = attention.view(*new_attention_shape)
        attention = attention.reshape(batch_size, seq_len, self.all_head_size)

        attention = self.linear_2(attention)

        return attention

    def forward(self, x, hidden, cell):
        # # 下面这是 position_encoding 实现
        # x = x.permute(1, 0, 2)                                                                 # 转化为 seq_len * batch_size * hidden_dim
        # # 输入的维度： seq_len * batch * hidden_dim
        # x=self.post(x)
        # # 输出的维度： seq_len * batch * hidden_dim
        # x = x.permute(1, 0, 2)                                                                 # 转化为 batch_size * seq_len * hidden_dim

        # 下面这是双 LSTM+Attention 实现
        lstm1_out, (h1_n, c1_n) = self.lstm1(x, (hidden, cell))
        lstm1_out = self.dropout(lstm1_out)
        lstm1_out,  (h2_n, c2_n) = self.lstm2(lstm1_out, (h1_n, c1_n))
        attn_output = self.attention_net(lstm1_out)                                             # 和LSTM的不同就在于这一句   40 x 25 x 1024 [batch_size, seq_len, dim]
        attn_output = attn_output.permute(0, 2, 1)                                              # 40 x 1024 x 25 [batch_size, seq_len, dim]

        # 下面这是 CNN-1D 的实现
        cov_output01 = F.relu(self.conv31(attn_output))                                         # 40 x 512 x 25 [batch_size, dim, seq_len]
        cov_output01 = cov_output01.permute(0, 2, 1)                                            # 40 x 25 x 512 [batch_size, seq_len, dim]
        cov_output01 = F.max_pool1d(cov_output01, 2)                                            # 40 x 25 x 256 [batch_size, seq_len, dim]

        cov_output02 = F.relu(self.conv32(attn_output))
        cov_output02 = cov_output02.permute(0, 2, 1)
        cov_output02 = F.max_pool1d(cov_output02, 2)

        cov_output03= F.relu(self.conv33(attn_output))
        cov_output03 = cov_output03.permute(0, 2, 1)
        cov_output03 = F.max_pool1d(cov_output03, 2)

        cov_output04 = F.relu(self.conv34(attn_output))
        cov_output04 = cov_output04.permute(0, 2, 1)
        cov_output04 = F.max_pool1d(cov_output04, 2)

        # 下面是将 4 个做完 LSTM-CNN 的数据拼起来
        attn_output = torch.cat([cov_output01, cov_output02, cov_output03, cov_output04], 2)    # 40 x 25 x 1024 [batch_size, seq_len, dim]

        # 下面是将拼接后的数据输出为 1 维的预测值
        predictions = self.linear_3(attn_output)                                                # 40 x 25 x 1 [batch_size, seq_len, out_dim]

        # 这段代码将seq_len的数据压缩成1个
        # predictions = predictions.permute(0, 2, 1)
        # predictions = self.linear_4(predictions)
        # predictions = predictions.permute(0, 2, 1)

        return predictions, h2_n, c2_n

# 定义 RMSE 损失函数。因为torch没有。
def RMSE(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


# 定义生成数据函数
def generate_data(in_file_name: str):
    # 加载数据
    dataset = pd.read_csv(in_file_name, index_col=0)
    dataset = dataset.fillna(0)                                                  # 这句话可有可无

    # 将数据按照BATCH_SIZE的窗口进行滑动，每个窗口数据做一组
    TRAIN_VALIDATION_RATIO = 0.9                                                 # 取 90% 的数据为训练集
    TRAIN_BATCH_SIZE = 40                                                        # 设40个seq为一个batch
    TEST_BATCH_SIZE = 1                                                          # 只留一个test的batch
    SEQ_LENGTH = 25                                                              # 定义每个time sequence取前25个序列数据来预测。取太大效果反而不好。
    Y_DIM = 1                                                                    # 预测的值就是1维的一个数字.需要被预测的时间序列就1个。如果需要预测未来多个时间序列，要用Seq2Seq模型
    X_DIM = dataset.shape[1]-Y_DIM                                               # 输入的sequence里每个element的维度数量，也是encoder的input_dim

    # 把原有的序列数据进行格式转换
    # 一共生成了 n 个 batch数据，
    # 每个batch有 batch_size 个 sequence，
    # 每个sequence有 seq_len 个 element，
    # 每个element有 X_DIM 个feature
    rolling_data = pd.DataFrame()
    for i in dataset.rolling(SEQ_LENGTH):                                        # 在原有数据上按照窗口大小为 seq_len 进行平移。窗口数据格式为[seq_len, X_DIM+Y_DIM]
        if i.shape[0] == SEQ_LENGTH:                                             # 如果得到的窗口数据的第一维和 seq_len 一样宽，说明数据完整。
            rolling_data = rolling_data.append(i)                                # 将这个完整的窗口数据追加到 dataframe 后面去。数据格式为 [seq_count, seq_len, x_dim+y_dim]

    rolling_data = rolling_data.values.reshape(-1, SEQ_LENGTH, X_DIM+Y_DIM)      # 变形后的数据一共是 [seq_count x seq_len x xy_dim]

    print("rolling_data shape: {}".format(rolling_data.shape))                   # 现在数据的shape就是 (seq_count, seq_len, xy_dim)
    print("seq count: {}".format(rolling_data.shape[0]))                         # 所以一共有seq_count个二维数据，其中二维数据里有seq_len行，每行中元素是x+1维 （包括y）
    print("seq length: {}".format(SEQ_LENGTH))                                   # 这个seq_len是手动定义的。

    TEST_BATCH_SIZE = (rolling_data.shape[0])%TRAIN_BATCH_SIZE                   # 所有的seq被划分为：train, validate, test 三个部分；那些被BATCH_SIZE除以后余下的就是test；batch_size就是每个batch中seq的数量。
    if TEST_BATCH_SIZE == 0:                                                     # 如果 rolling_data 刚好被整除，那就将最后一个batch算作test
        TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
    TEST_BATCH_COUNT = 1                                                         # 不管test的batch多大，test中只保留一个batch
    TRAIN_BATCH_COUNT = int(((rolling_data.shape[0]-TEST_BATCH_SIZE*TEST_BATCH_COUNT)//TRAIN_BATCH_SIZE) * TRAIN_VALIDATION_RATIO)  # 其他的seq取90%为 train
    VALID_BATCH_COUNT = int(((rolling_data.shape[0]-TEST_BATCH_SIZE*TEST_BATCH_COUNT)//TRAIN_BATCH_SIZE) - TRAIN_BATCH_COUNT)       # 再剩下的seq作为 validation

    print("TRAIN_BATCH_COUNT : {}".format(TRAIN_BATCH_COUNT))
    print("VALID_BATCH_COUNT : {}".format(VALID_BATCH_COUNT))
    print("TEST_BATCH_COUNT  : {}".format(TEST_BATCH_COUNT))

    # 现在所有的 sequence 被划分成了 train, valid, test 三个区域
    train = rolling_data[:TRAIN_BATCH_COUNT*TRAIN_BATCH_SIZE].reshape(TRAIN_BATCH_COUNT, TRAIN_BATCH_SIZE, SEQ_LENGTH, X_DIM+Y_DIM)                                      # 把数据转成 tain_batch_count x TRAIN_BATCH_SIZE x seq_len x in_dim 格式
    valid = rolling_data[TRAIN_BATCH_COUNT*TRAIN_BATCH_SIZE:-TEST_BATCH_COUNT*TEST_BATCH_SIZE].reshape(VALID_BATCH_COUNT, TRAIN_BATCH_SIZE, SEQ_LENGTH, X_DIM+Y_DIM)     # 把数据转成 tain_batch_count x TRAIN_BATCH_SIZE x seq_len x in_dim 格式
    test  = rolling_data[-TEST_BATCH_COUNT*TEST_BATCH_SIZE:].reshape(TEST_BATCH_COUNT, TEST_BATCH_SIZE, SEQ_LENGTH, X_DIM+Y_DIM)                                         # 把数据转成 test_batch_count x TEST_BATCH_SIZE x seq_len x in_dim 格式

    TRAIN_BATCH_COUNT = train.shape[0]                                          # 现在就可以获得train的 batch_count了
    TRAIN_BATCH_SIZE = train.shape[1]
    VALID_BATCH_COUNT = valid.shape[0]                                          # 现在就可以获得valid的 batch_count了
    VALID_BATCH_SIZE = valid.shape[1]
    TEST_BATCH_COUNT = test.shape[0]                                            # 现在就可以获得test的 batch_count了。这个就是1
    TEST_BATCH_SIZE = test.shape[1]                                             # 这个test的batch_size可能不太一样。

    # 要把数据从 fload/double 转换为 tensor
    train = torch.tensor(train)
    valid = torch.tensor(valid)
    test  = torch.tensor(test)

    # 把划分好的 train, valid, test 数据再分别区分为 X, Y 两个部分
    # 有点绕，通常 y 也是取 seq_len 的长度，因为根据seq_len的数据来预测未来，那么，y只取最后一个即可。
    # time-sequence : [time-element0, time-element1 , ... , time-element24]
    # time-element  : [y_dim, x_dim1, x_dim2, ... , x_dim100]
    # 就是要计算 time-element24 的 y_dim 的loss，其他的y_dim都忽略
    train_x, train_y = train[:, :, :, Y_DIM:], train[:, :, -1:, 0:Y_DIM]           # [train_batch_count, batch_size, sequence_length, X-dimension]     [train_batch_count, batch_size, 1, y-dimension]
    valid_x, valid_y = valid[:, :, :, Y_DIM:], valid[:, :, -1:, 0:Y_DIM]           # [valid_batch_count, batch_size, sequence_length, X-dimension]  [valid_batch_count, batch_size, 1, y-dimension]
    test_x,  test_y  = test[:, :, :,  Y_DIM:],  test[:, :, -1:, 0:Y_DIM]           # [train_batch_count, batch_size, sequence_length, X-dimension]  [train_batch_count, batch_size, 1, Y-dimension]

    # train_y = train_y.permute(0, 1, 3, 2)                                          # conver from [train_batch_count, batch_size, seq_length, y_dim]  to [train_batch_count, batch_size, y_seq_len, 1-dim]
    # vald_y = valid_y.permute(0, 1, 3, 2)                                          # conver from [train_batch_count, batch_size, seq_length, y_dim]  to [train_batch_count, batch_size, y_seq_len, 1-dim]
    # itest_y  =  test_y.permute(0, 1, 3, 2)                                          # conver from [test_batch_count, batch_size, seq_length, y_dim]  to  [test_batch_count, batch_size, y_seq_len, 1-dim]

    # 如果用了 GPU, 这些数据就要从CPU-memory中搬移到GPU-memory里
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    valid_x = valid_x.to(device)
    valid_y = valid_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    print("train_x: {}".format(train_x.shape))
    print("train_y: {}".format(train_y.shape))
    print("valid_x: {}".format(valid_x.shape))
    print("valid_y: {}".format(valid_y.shape))
    print("test_x:  {}".format(test_x.shape))
    print("test_y:  {}".format(test_y.shape))
    print("train_batch_count: {}".format(train.shape[0]))
    print("valid_batch_count: {}".format(valid.shape[0]))
    print("test_batch_count:  {}".format(test.shape[0]))

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# 定义生成模型函数
def generate_model(train_x, train_y, valid_x, valid_y):
    """
    用来生成 LSTM 模型的函数。这是要找到valid_loss最小情况下的模型。
    :param train_x: 训练集的输入值。数据是4维的tensor [train_batch_count, train_batch_size, seq_len, x_dim]
    :param train_y: 训练集的实际值。数据是4维的tensor [train_batch_count, train_batch_size, 1,       y_dim]
    :param valid_x: 验证集的输入值。数据室4维的tensor [valid_batch_count, valid_batch_size, seq_len, x_dim]
    :param valid_y: 验证集的实际值。数据室4维的tensor [valid_batch_count, valid_batch_size, 1      , y_dim]
    :return: 根据输入的数据生成的模型
    """
    # 训练 LSTM 模型 ---- 这里的损失函数是计算Sequence最后一个元素的预测数据和真实数据差异
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 2

    model = LSTMModel(input_size=train_x.shape[3], hidden_dim_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, seq_len=train_x.shape[2], output_dim_size=1).double().to(device)
    LR = 1e-5
    # loss_func = nn.MSELoss(reduction="mean")
    loss_func = RMSE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1, last_epoch=-1)

    # 训练 LSTM 模型;  ---- 这里的损失函数是计算Sequence最后一个元素的预测数据和真实数据差异
    model.train()
    epoches = 200
    train_epoch_loss = 0
    train_epoch_loss_list = []
    valid_smallest_loss = 1
    valid_smallest_epoch = 0
    valid_epoch_loss = 0
    valid_epoch_loss_list = []

    train_batch_count = train_x.shape[0]
    valid_batch_count = valid_x.shape[0]

    h0 = torch.zeros(NUM_LAYERS, train_x.shape[1], HIDDEN_SIZE).double().to(device)                     # NUM_LAYERS, TRAIN_BATCH_SIZE, HIDDEN_SIZE
    c0 = torch.zeros(NUM_LAYERS, train_x.shape[1], HIDDEN_SIZE).double().to(device)                     # NUM_LAYERS, TRAIN_BATCH_SIZE, HIDDEN_SIZE

    for epoch in range(epoches):
        batch_loss = []
        train_epoch_loss = 0
        train_pred_value_list = []
        train_real_value_list = []
        train_batch_list = list(range(0,train_batch_count))
        # random.shuffle(train_batch_list)
        for step in train_batch_list:
            train_pred, hn, cn = model(train_x[step], h0, c0)                                                    # pred: [batch_size, seq_len, out_dim]  但被修改成了 [batch_size, 1, out_dim]
            # h0, c0 = hn.detach(), cn.detach()
            loss = loss_func(train_pred[:, -1, -1], train_y[step][:, -1, -1])                                    # 取batch里每个sequence最后一个预测输出来和实际
            train_pred_value_list.extend(list(train_pred[:, -1].cpu().detach().flatten().numpy() ))
            train_real_value_list.extend(list(train_y[step, :, -1, -1].cpu().detach().flatten().numpy() ))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            batch_loss.append(loss.cpu().data.numpy())
        # print(batch_loss)
        train_epoch_loss = np.mean(batch_loss)

        batch_loss = []
        valid_epoch_loss = 0
        valid_pred_value_list = []
        valid_real_value_list = []
        for step in range(valid_batch_count):
            valid_pred, hn, cn = model(valid_x[step], h0, c0)
            loss = loss_func(valid_pred[:, -1, -1], valid_y[step][:, -1, -1])
            valid_pred_value_list.extend(list(valid_pred[:,-1].cpu().detach().flatten().numpy()))
            valid_real_value_list.extend(list(valid_y[step,:,-1,-1].cpu().detach().flatten().numpy()))
            batch_loss.append(loss.cpu().data.numpy())
        # print(batch_loss)
        valid_epoch_loss = np.mean(batch_loss)

        if ((epoch+1) %10) == 0:
            print("{} of {} epoch   train_loss: {:.3f}   valid_loss: {:.3f}".format(epoch, epoches, train_epoch_loss, valid_epoch_loss))

        valid_epoch_loss_list.append(valid_epoch_loss)
        train_epoch_loss_list.append(train_epoch_loss)

    plt.plot(train_epoch_loss_list, 'r-')
    plt.plot(valid_epoch_loss_list, 'b-')
    plt.show()
    print("min train loss: {:.3f}".format(min(train_epoch_loss_list)))
    print("min valid loss: {:.3f}".format(min(valid_epoch_loss_list)))

    # 这是 train 的拟合图形
    plt.plot(train_real_value_list, 'r-')
    plt.plot(train_pred_value_list, 'b-')
    plt.show()

    # 这是 vali的预测图形
    plt.plot(valid_real_value_list, 'r-')
    plt.plot(valid_pred_value_list, 'b-')
    plt.show()

    return model


# 定义利用模型进行预测函数
def model_predict(test_x, test_y):
    # 用模型预测数据
    # 考虑到时序因素，这里的时候误差很大。
    model.eval()
    test_loss = 0

    h0 = torch.zeros(NUM_LAYERS, test_x.shape[1], HIDDEN_SIZE).double().to(device)
    c0 = torch.zeros(NUM_LAYERS, test_x.shape[1], HIDDEN_SIZE).double().to(device)

    for step in range(test_x.shape[0]):
        pred, hn, cn = model(test_x[step], h0, c0)

        loss = loss_func(pred[:,-1,-1], test_y[step][:,-1,-1])               # Compare the all sequences' last element in one batch

        if test_x.shape[0] > 1:
            actual_line.append(test_y[step][-1,-1].item())
            pred_line.append(pred[-1,-1].item())
        elif test_x.shape[0] == 1:
            actual_line = test_y[step].cpu().detach().flatten().numpy()        # Only plot the last sequence of test batch
            pred_line   = pred[:,-1].cpu().detach().flatten().numpy()                # Only plot the last sequence of test batch

    print("Test Loss : {:.3f}".format(loss.data))
    print("Prediction: {:.2f}".format(float(pred[-1,-1].data)))
    print("Actual:     {:.2f}".format(float(test_y[step][-1,-1].data)))


    plt.plot(test_y[step,:,-1,-1].cpu().detach().flatten().numpy(), 'r--')
    plt.plot(pred[:,-1].cpu().detach().flatten().numpy(), 'b-')
    plt.show()
    print(test_y[step,:,-1,-1])
    print(pred[:,-1])


train_X, train_Y, valid_X, valid_Y, test_X, test_Y = generate_data("601229.csv")


dl_model = generate_model(train_X, train_Y, valid_X, valid_Y)


prediction = model_preidct(test_x, test_y)



