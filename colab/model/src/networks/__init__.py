# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper


from .ConvUNet2 import UNet3DWithSamplePoints
from .EfficientUNet import EfficientUNet3DWithSamplePoints
from .Transolver_conv_proj import ModelWith3DSamplePoints
from .TranAttnEfficientUNet import TransolverAttnEfficientUNet
from .RegDCCNN import RegDGCNNWith3DSamplePoints
# print the number of parameters
import operator
from functools import reduce


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    return c


def instantiate_network(config):
    if config.model == "UNet":
        model = UNet3DWithSamplePoints(
            in_channels=config.in_channels,  # xyz + sdf
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            num_levels=config.num_levels,
            use_position_input=config.use_position_input,
        )
    elif config.model == 'EfficientUNet':
        model = EfficientUNet3DWithSamplePoints(
            in_channels=config.in_channels,  # xyz + sdf
            out_channels=config.out_channels,
            model_name=config.model_name,
            hidden_channels=config.hidden_channels,
            use_position_input=config.use_position_input,
            use_attn=config.use_attn,
        )
    elif config.model == "Transolver":
        model = ModelWith3DSamplePoints(n_hidden=256,
                                        n_layers=8,
                                        space_dim=3,
                                        fun_dim=0,
                                        n_head=8,
                                        act="silu",
                                        mlp_ratio=2,
                                        out_dim=config.out_channels,
                                        slice_num=32,
                                        unified_pos=False,
                                        use_graph=False,
                                        )
    elif config.model == 'RegDGCNN':
        args = dict(
            k=40,
            emb_dims=512,
            dropout=0.4,
        )
        model = RegDGCNNWith3DSamplePoints(args, config.out_channels, config)
    else:
        raise ValueError("Network not supported")

    print("The model size is ", count_params(model))
    return model
