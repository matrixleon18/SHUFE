# encoding=GBK

"""
���������ѵ��(batch)
"""

import torch


BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_data_set = torch.utils.data.TensorDataset(x, y)  # x����ѵ��; y�����������;

loader = torch.utils.data.DataLoader(           # ��һ���ɵ�������ʹ��iter()���ʣ�����ʹ��next()����
    dataset=torch_data_set,                     # ���м������ݵ����ݼ���
    batch_size=BATCH_SIZE,                      # ÿһ��ѵ�������ݸ�����
    shuffle=True,                               # �Ƿ�����ȡ�����������ȡ������
    collate_fn=None,                            # ר�������� Dataset ��ķ���ֵƴ�ӳ� tensor�����ǲ����õ�ʱ�򣬻���� default �ĺ�����������ǵ�ѵ�����ݳ��Ȳ�һ��default ������ hold ��ס�ˣ��������Ҫ�Զ���һ�� collate_fn������ DataLoader ��������������������оͲ��ᱨ���� https://zhuanlan.zhihu.com/p/59772104
    batch_sampler=None,                         # ÿ������������������������ģʽ��������ʹ���ݸ����ж������ʡ���batch_size��shuffle ��sampler and drop_last������
    sampler=None,                               # ���ݶ���Ĳ��Դ����ݼ��в������롣����������������ϴ�ƣ�shuffle�����ñ���ΪFalse��
    num_workers=0,                              # �������ݼ��ص��ӽ������������������̡�
    pin_memory=False,                           # ���ΪTrue�����ݼ������ڷ���ȥ���������Ƶ�CUDA�̶��ڴ��С�
    drop_last=False,                            # ������ݼ���С���ܱ�batch_size������ ����ΪTrue����ɾ�����һ����������������
    timeout=0,                                  # �ռ����ݵĳ�ʱֵ��
    worker_init_fn=None                         # �ӽ��̵���ģʽ�������ݵ���ǰ�Ͳ��������󣬸��ݹ����ӽ��̵�ID�����˳�������ݡ�
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):  # �Կɵ����������
        # training...
        print("epoch: {}; step: {}; batch_x: {}; batch_y: {}".format(epoch, step, batch_x, batch_y))
