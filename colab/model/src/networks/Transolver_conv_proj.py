import numpy as np

import ppsci

# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle
import paddle.nn as nn

ACTIVATION = {
    "gelu": paddle.nn.GELU,
    "tanh": paddle.nn.Tanh,
    "sigmoid": paddle.nn.Sigmoid,
    "relu": paddle.nn.ReLU,
    "leaky_relu": paddle.nn.LeakyReLU(negative_slope=0.1),
    "softplus": paddle.nn.Softplus,
    "ELU": paddle.nn.ELU,
    "silu": paddle.nn.Silu,
}


class Physics_Attention_1D(paddle.nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        out_0 = paddle.create_parameter(
            shape=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).shape,
            dtype=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[1, heads, 1, 1]) * 0.5
            ),
        )
        out_0.stop_gradient = False
        self.temperature = out_0
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = paddle.nn.Linear(
            in_features=dim_head, out_features=slice_num
        )
        for ly in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(ly.weight)
        self.to_q = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        B, N, C = tuple(x.shape)
        fx_mid = (
            self.in_project_fx(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )
        x_mid = (
            self.in_project_x(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None]

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = paddle.matmul(x=q_slice_token * self.scale, y=k_slice_token, transpose_y=True)
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)

        out_x = paddle.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.reshape([out_x.shape[0], out_x.shape[2], -1])  # rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class MLP(paddle.nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_input, out_features=n_hidden), act()
        )
        self.linear_post = paddle.nn.Linear(in_features=n_hidden, out_features=n_output)
        self.linears = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden), act()
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x * int(self.res)
        x = self.linear_post(x)
        return x


class SwiGLU(paddle.nn.Layer):
    def __init__(self,
                 n_input, n_hidden, n_output, act="silu"):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            ACTIVATION[act](),
        )

        self.linear2 = nn.Linear(n_input, n_hidden)
        self.linear = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.linear(self.linear1(x) * self.linear2(x))


class Transolver_block(paddle.nn.Layer):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act="gelu",
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.Attn = Physics_Attention_1D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = (
            # MLP(
            #     hidden_dim,
            #     hidden_dim * mlp_ratio,
            #     hidden_dim,
            #     n_layers=0,
            #     res=False,
            #     act=act,
            # )
            SwiGLU(
                hidden_dim,
                hidden_dim * mlp_ratio,
                hidden_dim,
                act=act,
            )
        )
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, fx):
        x = self.ln_1(fx)
        x = self.Attn(x)
        fx = x + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            fx = self.mlp2(self.ln_3(fx))
        return fx


class Model(paddle.nn.Layer):
    def __init__(
            self,
            space_dim=1,
            n_layers=5,
            n_hidden=256,
            dropout=0,
            n_head=8,
            act="gelu",
            mlp_ratio=1,
            fun_dim=1,
            out_dim=1,
            slice_num=32,
            ref=8,
            unified_pos=False,
    ):
        super(Model, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = paddle.nn.LayerList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=_ == n_layers - 1,
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        param = 1 / n_hidden * paddle.rand(shape=(n_hidden,))
        out_1 = paddle.create_parameter(
            shape=param.shape,
            dtype=param.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(param),
        )
        # paddle.nn.initializer.xavier.XavierNormal(out_1)
        out_1.stop_gradient = False
        self.placeholder = out_1
        self.preprocess = (
            # MLP(
            #     fun_dim + space_dim,
            #     n_hidden * 2,
            #     n_hidden,
            #     n_layers=0,
            #     res=False,
            #     act=act,
            # )
            SwiGLU(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                act=act,
            )
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            m.weight = ppsci.utils.initializer.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    # @paddle.jit.to_static
    def forward(self, x):
        fx = self.preprocess(x) + self.placeholder  # [None, None, :]
        for block in self.blocks[:-1]:
            fx = block(fx)
        for block in self.blocks:
            fx = block(fx)
        # for block in self.blocks:
        #     fx = block(fx)
        return fx


class ModelWith3DSamplePoints(Model):
    def __init__(self, space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act="gelu",
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False):
        super().__init__(space_dim=space_dim,
                         n_layers=n_layers,
                         n_hidden=n_hidden,
                         dropout=dropout,
                         n_head=n_head,
                         act=act,
                         mlp_ratio=mlp_ratio,
                         fun_dim=fun_dim,
                         out_dim=out_dim,
                         slice_num=slice_num,
                         ref=ref,
                         unified_pos=unified_pos,)

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
        pred_var = super(ModelWith3DSamplePoints, self).forward(output_points)
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
        pred_var = super(ModelWith3DSamplePoints, self).forward(output_points)
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

