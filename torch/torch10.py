# encoding=GBK

"""
神经网络的批训练(batch)
"""

import torch


BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_data_set = torch.utils.data.TensorDataset(x, y)  # x用来训练; y用来计算误差;

loader = torch.utils.data.DataLoader(           # 是一个可迭代对象，使用iter()访问，不能使用next()访问
    dataset=torch_data_set,                     # 从中加载数据的数据集。
    batch_size=BATCH_SIZE,                      # 每一批训练的数据个数。
    shuffle=True,                               # 是否乱序取样（不是随机取样）。
    collate_fn=None,                            # 专门用来把 Dataset 类的返回值拼接成 tensor，我们不设置的时候，会调用 default 的函数，如果我们的训练数据长度不一，default 函数就 hold 不住了，因此我们要自定义一个 collate_fn，并在 DataLoader 中设置这个参数，再运行就不会报错了 https://zhuanlan.zhihu.com/p/59772104
    batch_sampler=None,                         # 每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质。和batch_size、shuffle 、sampler and drop_last不兼容
    sampler=None,                               # 根据定义的策略从数据集中采样输入。如果定义采样规则，则洗牌（shuffle）设置必须为False。
    num_workers=0,                              # 用于数据加载的子进程数。０就是主进程。
    pin_memory=False,                           # 如果为True，数据加载器在返回去将张量复制到CUDA固定内存中。
    drop_last=False,                            # 如果数据集大小不能被batch_size整除， 设置为True可以删除最后一个不完整的批处理。
    timeout=0,                                  # 收集数据的超时值。
    worker_init_fn=None                         # 子进程导入模式，在数据导入前和步长结束后，根据工作子进程的ID逐个按顺序导入数据。
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):  # 对可迭代对象访问
        # training...
        print("epoch: {}; step: {}; batch_x: {}; batch_y: {}".format(epoch, step, batch_x, batch_y))
