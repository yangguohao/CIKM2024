# 启动脚本：
# python3 -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset
import numpy as np

mesh0 = dist.ProcessMesh([1], dim_names=['x']) # 创建进程网格
mesh1 = dist.ProcessMesh([0], dim_names=['x']) # 创建进程网格


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        return input

    def __len__(self):
        return self.num_samples


class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = dist.shard_tensor(
                    self.create_parameter(shape=[1024, 4096*8]),
                    mesh0, [dist.Replicate()])
        self.w1 = dist.shard_tensor(
                    self.create_parameter(shape=[4096*8, 1024]),
                    mesh1, [dist.Replicate()])

    def forward(self, x):
        x = dist.shard_tensor(x, mesh0, [dist.Replicate()])
        y = paddle.matmul(x, self.w0)
        # 重切分，将 stage0 上的中间计算结果传输给 stage1
        y = dist.reshard(y, mesh1, [dist.Replicate()])
        print(y.shape)
        z = paddle.matmul(y, self.w1)
        return z


# model = MlpModel()
# # dataset = RandomDataset(128, 1024)
# # sampler = BatchSampler(
# #     dataset,
# #     batch_size=2,
# # )
# # dataloader = DataLoader(
# #     dataset,
# #     batch_sampler=sampler,
# # )
# # dataloader = dist.shard_dataloader(dataloader, meshes=[mesh0, mesh1], shard_dims='x')
#
# opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
# opt = dist.shard_optimizer(opt)
#
# for step in range(10):
#     data = paddle.randn(shape=[128, 1024])
#     logits = model(data)
#     loss = paddle.mean(logits)
#     loss.backward()
#     opt.step()
#     opt.clear_grad()

# x = paddle.randn((1, 2048+256, 70000))
y = 1*(2048+256)*70000*4 /(1024*1024)
print(y)