import paddle.nn as nn

import paddle
from .Transolver_conv_proj import Model
from .ConvUNet2 import UNet3D


class TransolverAttnEfficientUNet(nn.Layer):
    def __init__(self, in_channels: int, hidden_channels=64, num_levels=4, use_attn=False, space_dim=1,
                 n_layers=5, n_hidden=256, dropout=0, n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, k=20, unified_pos=False, use_graph=True):
        super().__init__()
        self.unet = UNet3D(in_channels, hidden_channels, final_sigmoid=False,
                           f_maps=64, layer_order='gcr', num_groups=8, num_levels=num_levels,
                           is_segmentation=False, conv_padding=1, use_attn=use_attn, )
        self.transolver = Model(space_dim=space_dim,
                                n_layers=n_layers,
                                n_hidden=n_hidden,
                                dropout=dropout,
                                n_head=n_head,
                                act=act,
                                mlp_ratio=mlp_ratio,
                                fun_dim=fun_dim,
                                out_dim=hidden_channels,
                                slice_num=slice_num,
                                ref=ref,
                                k = k,
                                unified_pos=unified_pos,
                                use_graph=use_graph)
        self.ln = paddle.nn.LayerNorm(normalized_shape=hidden_channels)
        self.out_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_channels * 2, out_features=hidden_channels * 4),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_channels * 4, out_features=out_dim),
        )

    def data_dict_to_input(self, data_dict):
        input_grid_features = data_dict['sdf'].unsqueeze(axis=1)
        # grid_points = data_dict['sdf_query_points']
        # input_grid_features = paddle.concat(x=(input_grid_features,
        #                                        grid_points,
        #                                        ), axis=1)
        output_points = data_dict['vertices']
        return input_grid_features, output_points

    @paddle.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        unet_var = self.unet_forward(input_grid_features, output_points)
        pred_var = self.out_mlp( paddle.concat([self.transolver(output_points), unet_var], axis=-1))

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

    def unet_forward(self, x, output_points):
        x = self.unet(x)
        output_points = output_points.unsqueeze(axis=2).unsqueeze(axis=2)
        x = paddle.nn.functional.grid_sample(x=x, grid=output_points,
                                             align_corners=False)
        x = x.squeeze(axis=3).squeeze(axis=3)
        x = x.transpose(perm=[0, 2, 1])
        return self.ln(x)

    def loss_dict(self, data_dict, loss_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        unet_var = self.unet_forward(input_grid_features, output_points)
        pred_var = self.out_mlp(paddle.concat([self.transolver(output_points), unet_var], axis=-1))

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
    import operator
    from functools import reduce


    def count_params(model):
        c = 0
        for p in list(model.parameters()):
            c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
        return c


    model = TransolverAttnEfficientUNet(in_channels=4, hidden_channels=64, out_dim=1, num_levels=6, fun_dim=0, space_dim=3)
    x = paddle.rand(shape=(8, 4, 64, 64, 64))
    y = paddle.rand(shape=(8, 1765, 3))
    # upsample = nn.Conv3DTranspose(4, 16, kernel_size=4, stride=4,)
    # output = upsample(x)
    print(model.loss_dict(x, y)[0, 0, :].shape)
