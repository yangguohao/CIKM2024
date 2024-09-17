import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.spatial import cKDTree


def knn(x, k):

    unique_batches = np.arange(0, x.shape[0], 1)
    all_distances = np.full((x.shape[0], x.shape[2], k), np.inf)
    all_indices = np.full((x.shape[0], x.shape[2], k), -1)

    for batch in unique_batches:
        batch_x_points = x[batch]

        if batch_x_points.size == 0 or batch_x_points.size == 0:
            continue

        tree = cKDTree(batch_x_points.transpose([1, 0]))
        distances, indices = tree.query(batch_x_points.transpose([1, 0]), k=k)

        all_distances[batch] = distances
        all_indices[batch] = indices

    return paddle.to_tensor(all_indices)
#
# def knn(x, k):
#     # Calculate pairwise distance, shape (batch_size, num_points, num_points)
#     step = paddle.to_tensor(np.ceil(x.shape[2] / k), dtype=paddle.int64)
#     idx = paddle.zeros(shape=(x.shape[0], x.shape[2], k), dtype=paddle.int64)
#     x = x.transpose([0, 2, 1])
#     for i in range(k):
#         total_idx = None
#         total_number = None
#         for j in range(k):
#             tmp = x[:, step * i: step * (i + 1), :]
#             tmp2 = x[:, step * j: step * (j + 1), :]
#             pairwise_distance = -paddle.cdist(tmp, tmp2)
#             # Retrieve the indices of the k nearest neighbors
#             tmp_idx = pairwise_distance.topk(k=min(k, pairwise_distance.shape[-1]), axis=-1)
#             if total_number is None and total_idx is None:
#                 total_idx = tmp_idx[1]
#                 total_number = tmp_idx[0]
#             else:
#                 idx_base = paddle.arange(j * step, min((j + 1) * step, x.shape[1]))
#                 total_idx = paddle.concat([idx_base[tmp_idx[1]], total_idx], axis=-1)
#                 total_number = paddle.concat([tmp_idx[0], total_number], axis=-1)
#                 tmp_idx2 = total_number.topk(k=min(k, total_number.shape[-1]), axis=-1)
#                 total_idx = paddle.take_along_axis(total_idx, tmp_idx2[1], axis=-1)
#                 total_number = tmp_idx2[0]
#         idx[:, step * i: step * (i + 1), :] = total_idx
#
#     return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape((batch_size, -1, num_points))

    # Compute k-nearest neighbors if not provided
    if idx is None:
        idx = knn(x, k=k)

    # Prepare indices for gathering

    idx_base = paddle.arange(0, batch_size, ).reshape((-1, 1, 1)) * num_points
    idx = idx + idx_base
    idx = idx.reshape((-1,))

    _, num_dims, _ = x.shape
    x = x.transpose([0, 2, 1])

    # Gather neighbors for each point to construct local regions
    feature = x.reshape((batch_size * num_points, -1))[idx, :]
    feature = feature.reshape((batch_size, num_points, k, num_dims))

    # Expand x to match the dimensions for broadcasting subtraction
    x = x.reshape((batch_size, num_points, 1, num_dims)).tile([1, 1, k, 1])

    # Concatenate the original point features with the relative positions to form the graph features
    feature = paddle.concat((feature - x, x), axis=3).transpose([0, 3, 1, 2])
    return feature


class RegDGCNN(nn.Layer):

    def __init__(self, args, output_channels=1, track=''):
        super(RegDGCNN, self).__init__()
        self.args = args
        self.k = args['k']  # Number of nearest neighbors
        dim_list = [256, 512, 512, 1024]
        # Batch normalization layers to stabilize and accelerate training
        self.conv_list = nn.LayerList([
            nn.Sequential(nn.Conv2D(6, dim_list[0], kernel_size=1, bias_attr=False),
                          nn.BatchNorm2D(dim_list[0]),
                          nn.LeakyReLU(negative_slope=0.2)),
            nn.Sequential(nn.Conv2D(dim_list[0] * 2, dim_list[1], kernel_size=1, bias_attr=False),
                          nn.BatchNorm2D(dim_list[1]),
                          nn.LeakyReLU(negative_slope=0.2)),
            nn.Sequential(nn.Conv2D(dim_list[1] * 2, dim_list[2], kernel_size=1, bias_attr=False),
                          nn.BatchNorm2D(dim_list[2]),
                          nn.LeakyReLU(negative_slope=0.2)),
            nn.Sequential(nn.Conv2D(dim_list[2] * 2, dim_list[3], kernel_size=1, bias_attr=False),
                          nn.BatchNorm2D(dim_list[3]),
                          nn.LeakyReLU(negative_slope=0.2)),
        ])
        self.conv5 = nn.Sequential(nn.Conv1D(sum(dim_list), args['emb_dims'], kernel_size=1, bias_attr=False),
                                   nn.BatchNorm1D(args['emb_dims']),
                                   nn.LeakyReLU(negative_slope=0.2))

        if 'Cd' in track:
            bn = nn.BatchNorm1D
            self.linear_list = nn.LayerList([
                nn.Sequential(
                    nn.Linear(args['emb_dims'] * 2, 128, bias_attr=False),
                    bn(128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(128, 64, bias_attr=False),
                    bn(64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(64, 32, bias_attr=False),
                    bn(32),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(32, 16, bias_attr=False),
                    bn(16),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
            ])

        else:
            bn = nn.Identity
            self.linear_list = nn.LayerList([
                nn.Sequential(
                    nn.Linear(args['emb_dims'], 128, bias_attr=False),
                    bn(),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(128, 64, bias_attr=False),
                    bn(),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(64, 32, bias_attr=False),
                    bn(),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
                nn.Sequential(
                    nn.Linear(32, 16, bias_attr=False),
                    bn(),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(p=args['dropout'])
                ),
            ])

        self.linear5 = nn.Linear(16, output_channels)  # The final output layer
        self.track = track

    # @paddle.amp.debugging.check_layer_numerics
    # @paddle.jit.to_static
    def forward(self, x):
        """
        Forward pass of the model to process input data and predict outputs.

        Args:
            x (paddle.Tensor): Input tensor representing a batch of point clouds.

        Returns:
            paddle.Tensor: Model predictions for the input batch.
        """
        batch_size = x.shape[0]
        feature_list = []

        for conv1 in self.conv_list:
            x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x = conv1(x)  # conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 256, num_points, k)
            x = paddle.max(x, axis=-1)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            feature_list.append(x)

        # Concatenate features from all EdgeConv blocks
        x = paddle.concat(feature_list, axis=1)  # (batch_size, 256+512+512+1024, num_points)
        # Apply the final convolutional block
        x = self.conv5(x)  # (batch_size, 256+512+512+1024, num_points) -> (batch_size, emb_dims, num_points)

        if 'Cd' in self.track:
            # Combine global max and average pooling features
            x1 = F.adaptive_max_pool1d(x, 1).reshape(
                (batch_size, -1))  #(batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
            x2 = F.adaptive_avg_pool1d(x, 1).reshape(
                (batch_size, -1))  #(batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
            x = paddle.concat((x1, x2), 1)  # (batch_size, emb_dims*2,)
        else:
            x = x.transpose([0, 2, 1])  # (batch_size, emb_dims, num_points)

        # Process features through fully connected layers with dropout and batch normalization
        for linear in self.linear_list:
            x = linear(x)

        # Final linear layer to produce the output
        x = self.linear5(x)  # (batch_size, 16) -> (batch_size, 1)

        return x


class RegDGCNNWith3DSamplePoints(RegDGCNN):
    def __init__(self, args, output_channels, config):
        super().__init__(args=args, output_channels=output_channels, track=config.track)

    def data_dict_to_input(self, data_dict):
        input_grid_features = data_dict['sdf'].unsqueeze(axis=1)
        grid_points = data_dict['sdf_query_points']
        input_grid_features = paddle.concat(x=(input_grid_features,
                                               grid_points), axis=1)
        output_points = data_dict['vertices']
        return input_grid_features, output_points

    @paddle.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        _, output_points = self.data_dict_to_input(data_dict)
        pred_var = self.forward(output_points.transpose([0, 2, 1]))
        pred_var = decode_fn(pred_var)
        pred_var_key = None
        if 'pressure' in data_dict.keys():
            pred_var = pred_var.squeeze(0)
            pred_var_key = 'pressure'
        elif "velocity" in data_dict.keys():
            pred_var = pred_var.squeeze(0)
            pred_var_key = 'velocity'
        elif "cd" in data_dict.keys():
            pred_var = paddle.mean(pred_var)
            pred_var_key = 'cd'
        else:
            raise NotImplementedError("only pressure velocity works")
        return {pred_var_key: pred_var}

    def loss_dict(self, data_dict, loss_fn=None, **kwargs):
        _, output_points = self.data_dict_to_input(data_dict)
        pred_var = self.forward(output_points.transpose([0, 2, 1]))
        true_var = None
        if 'pressure' in data_dict.keys():
            true_var = data_dict['pressure'].unsqueeze(-1)
        elif "velocity" in data_dict.keys():
            true_var = data_dict['velocity']
        elif "cd" in data_dict.keys():
            pred_var = paddle.mean(pred_var)
            true_var = data_dict['cd']
        else:
            raise NotImplementedError("only pressure velocity works")
        # print("pred_var = ", pred_var)
        # print("true_var = ", true_var)
        return {'loss': loss_fn(pred_var, true_var)}


if __name__ == '__main__':
    # import operator
    # from functools import reduce
    #
    #
    # def count_params(model):
    #     c = 0
    #     for p in list(model.parameters()):
    #         c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    #     return c
    #
    #
    # import random
    # import paddle.distributed as dist
    # from paddle.distributed import fleet
    # from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model
    #
    # # strategy = fleet.DistributedStrategy()
    # #
    # # # 设置两路张量模型并行
    # # model_parallel_size = 2
    # # data_parallel_size = 1
    # # strategy.hybrid_configs = {
    # #     "dp_degree": data_parallel_size,
    # #     "mp_degree": model_parallel_size,
    # #     "pp_degree": 1
    # # }
    # # # 注意 strategy 是这里传递的，动态图只能这里，静态图还可以在 distributed_optimizer 里传
    # # fleet.init(is_collective=True, strategy=strategy)
    # #
    # #
    # # def set_random_seed(seed, rank_id):
    # #     random.seed(seed)
    # #     np.random.seed(seed)
    # #     paddle.seed(seed + rank_id)
    # # hcg = fleet.get_hybrid_communicate_group()
    # # mp_id = hcg.get_model_parallel_rank()
    # # rank_id = dist.get_rank()
    # # set_random_seed(1024, rank_id)
    #
    # # fleet.init(is_collective=True)
    # # group = paddle.distributed.new_group([0, 1, ])
    #
    # # paddle.set_device('gpu:1')
    # paddle.seed(1024)
    # random.seed(1024)
    # np.random.seed(1024)
    # args = dict(
    #     k=10,
    #     emb_dims=512,
    #     dropout=0.4,
    # )
    # model = RegDGCNN(args, 1, 'Cd')
    # use_pure_fp16 = 1
    #
    # clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    # optimizer = paddle.optimizer.AdamW(
    #     learning_rate=0.00001,
    #     parameters=model.parameters(),
    #     weight_decay=0.00001,
    #     grad_clip=clip,
    #     multi_precision=bool(use_pure_fp16)
    # )
    #
    # scaler = None
    # if use_pure_fp16:
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    #     # level O2 means converting the network to FP16
    #     model, optimizer = paddle.amp.decorate(
    #         models=model,
    #         optimizers=optimizer,
    #         level='O1',
    #         save_dtype='float32')
    #
    # # wrap GroupSharded model, optimizer and scaler
    # # model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g_os", scaler=scaler)
    # # model = fleet.distributed_model(model)
    # # optimizer = fleet.distributed_optimizer(optimizer)
    # for step_id in range(1, 10):
    #     x = paddle.rand(shape=(1, 3, 63094))
    #     x.stop_gradient = False
    #     with paddle.amp.debugging.collect_operator_stats():
    #         with paddle.amp.auto_cast(use_pure_fp16,
    #                                   custom_black_list=[
    #                                       # 'einsum',
    #                                       # 'softmax',
    #                                       # 'matmul'
    #                                       'conv2d',
    #                                       # 'matmul_v2',
    #                                       # 'reduce_sum',
    #                                       # 'cast'
    #                                   ],
    #                                   level='O1'
    #                                   ):
    #             out = model(x)
    #     loss = out.mean()
    #     if use_pure_fp16:
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #     else:
    #         loss.backward()
    #         optimizer.step()
    #     optimizer.clear_grad()
    #     if scaler is not None and scaler._cache_founf_inf:
    #         for p in list(model.parameters()):
    #             p.clear_gradient()
    #     print("=== step_id : {}    loss : {}".format(step_id, loss.numpy()))
    #     break
    # # fleet.meta_parallel
    # # save model and optimizer state_dict
    # # save_group_sharded_model(model, output_dir, optimizer)
    # print(model(x).shape)
    seed = 155  # np.random.randint(0, 1024)
    print(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    x = paddle.rand(shape=(1, 32, 6000))
    print(x[:, 0, :10])
    res = paddle.sort(knn(x, 20).astype(paddle.float32), axis=-1)
    res2 = paddle.sort(knn2(x, 20).astype(paddle.float32), axis=-1)
    for i in range(x.shape[-1]):
        if bool(paddle.allclose(
                res[:, i, :],
                res2[:, i, :]
        )) is not True:
            print(i, res[:, i, :], res2[:, i, :], paddle.allclose(
                res[:, i, :],
                res2[:, i, :]
            ))
            break
