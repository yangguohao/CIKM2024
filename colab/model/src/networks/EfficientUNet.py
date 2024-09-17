import paddle
import paddle.nn as nn
import math
from collections import OrderedDict, namedtuple
import re
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = paddle.stack((sin_inp.sin(), sin_inp.cos()), axis=-1)
    return paddle.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Layer):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, channels, 2) / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistable=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = paddle.arange(x, dtype=self.inv_freq.dtype)
        pos_y = paddle.arange(y, dtype=self.inv_freq.dtype)
        pos_z = paddle.arange(z, dtype=self.inv_freq.dtype)
        sin_inp_x = paddle.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = paddle.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = paddle.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = paddle.zeros(
            (x, y, z, self.channels * 3),
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels: 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].tile([batch_size, 1, 1, 1, 1])
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Layer):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.transpose([0, 2, 3, 4, 1])
        enc = self.penc(tensor)
        return enc.transpose([0, 4, 1, 2, 3])

    @property
    def org_channels(self):
        return self.penc.org_channels


GlobalParams = namedtuple('GlobalParams',
                          ['batch_norm_momentum', 'batch_norm_epsilon',
                           'dropout_rate', 'num_classes',
                           'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth',
                           'drop_connect_rate'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple('BlockArgs', ['kernel_size',
                                     'num_repeat', 'input_filters',
                                     'output_filters', 'expand_ratio',
                                     'id_skip', 'strides', 'se_ratio'])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def get_efficientnet_params(model_name, override_params=None):
    """Get efficientnet params based on model name
    """
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        # Note: the resolution here is just for reference, its values won't be used.
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    if model_name not in params_dict.keys():
        raise KeyError('There is no model named {}.'.format(model_name))

    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None)

    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=int(options['s'][0]),
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def round_filters(filters, global_params):
    """Round number of filters
    """
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of repeats
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += paddle.rand([batch_size, 1, 1, 1, 1], dtype=inputs.dtype, )
    binary_tensor = paddle.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3D(out_channels),
        nn.ReLU(),
        nn.Conv3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3D(out_channels),
        nn.ReLU()
    )


def up_conv(in_channels, out_channels):
    return nn.Conv3DTranspose(
        in_channels, out_channels, kernel_size=2, stride=2
    )


def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, out_channels)
    )


class Conv3dSamePadding(nn.Conv3D):
    """3D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding='same', dilation=dilation,
                         groups=groups,
                         bias_attr=bias, )
        self._name = name


class MBConvBlock(nn.Layer):
    """Mobile Inverted Residual Bottleneck Block
    """

    def __init__(self, block_args, global_params, idx):
        super().__init__()

        block_name = 'blocks_' + str(idx) + '_'

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = nn.Swish(block_name + '_swish')

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv3dSamePadding(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  bias=False,
                                                  name=block_name + 'expansion_conv')
            self._bn0 = nn.BatchNorm3D(num_features=out_channels,
                                       momentum=self.batch_norm_momentum,
                                       epsilon=self.batch_norm_epsilon,
                                       use_global_stats=True,
                                       name=block_name + 'expansion_batch_norm')

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides

        self._depthwise_conv = Conv3dSamePadding(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 groups=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=strides,
                                                 bias=False,
                                                 name=block_name + 'depthwise_conv')
        self._bn1 = nn.BatchNorm3D(num_features=out_channels,
                                   momentum=self.batch_norm_momentum,
                                   epsilon=self.batch_norm_epsilon,
                                   use_global_stats=True,
                                   name=block_name + 'depthwise_batch_norm')

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv3dSamePadding(in_channels=out_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv3dSamePadding(in_channels=num_squeezed_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')

        # Output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv3dSamePadding(in_channels=out_channels,
                                               out_channels=final_output_channels,
                                               kernel_size=1,
                                               bias=False,
                                               name=block_name + 'output_conv')
        self._bn2 = nn.BatchNorm3D(num_features=final_output_channels,
                                   momentum=self.batch_norm_momentum,
                                   epsilon=self.batch_norm_epsilon,
                                   use_global_stats=True,
                                   name=block_name + 'output_batch_norm')

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = nn.functional.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = nn.functional.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x


def create_conv(in_channels, out_channels, kernel_size, order, num_groups,
                padding, is3d, stride=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, 'Conv layer MUST be present'
    assert order[0
           ] not in 'rle', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', paddle.nn.ReLU()))
        elif char == 'l':
            modules.append(('LeakyReLU', paddle.nn.LeakyReLU()))
        elif char == 'e':
            modules.append(('ELU', paddle.nn.ELU()))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            if is3d:
                conv = paddle.nn.Conv3D(in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding, bias_attr=bias)
            else:
                conv = paddle.nn.Conv2D(in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding, bias_attr=bias)
            modules.append(('conv', conv))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', paddle.nn.GroupNorm(num_groups=
                                                             num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = paddle.nn.BatchNorm3D
            else:
                bn = paddle.nn.BatchNorm2D
            if is_before_conv:
                modules.append(('batchnorm', bn(in_channels)))
            else:
                modules.append(('batchnorm', bn(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )
    return modules


class SingleConv(paddle.nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order=
    'gcr', num_groups=8, padding=1, is3d=True, stride=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels,
                                        kernel_size, order, num_groups, padding, is3d, stride=stride):
            self.add_sublayer(name=name, sublayer=module)


class Encoder(nn.Layer):
    def __init__(self, in_channels, model_name, block_args_list, global_params):
        super().__init__()

        self.name = model_name
        self.block_args_list = block_args_list
        self.global_params = global_params
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon
        out_channels = round_filters(32, self.global_params)

        self.stem_conv = Conv3dSamePadding(in_channels,
                                           out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           bias=False,
                                           name='stem_conv')

        self.stem_batch_norm = nn.BatchNorm3D(num_features=out_channels,
                                              momentum=batch_norm_momentum,
                                              epsilon=batch_norm_epsilon,
                                              use_global_stats=True,
                                              name='stem_batch_norm')
        self.stem_swish = nn.Swish(name='stem_swish')
        self.blocks = nn.LayerList([])
        idx = 0
        for block_args in self.block_args_list:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self.blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            # The rest of the blocks
            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
                idx += 1
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self.head_conv = Conv3dSamePadding(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=False,
                                           name='head_conv')
        self.head_batch_norm = nn.BatchNorm3D(num_features=out_channels,
                                              momentum=batch_norm_momentum,
                                              epsilon=batch_norm_epsilon,
                                              use_global_stats=True,
                                              name='head_batch_norm')
        self.head_swish = nn.Swish(name='head_swish')

    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_batch_norm(x)
        x = self.stem_swish(x)
        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self.blocks)
            x = block(x, drop_connect_rate)
        # Head
        x = self.head_conv(x)
        x = self.head_batch_norm(x)
        x = self.head_swish(x)
        return x


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module._name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = tuple(output.shape[-3:])
                    # if shape not in shapes:
                    #     shapes.add(shape)
                    # else:
                    blocks[shape] = output

                elif module._name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module._name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.LayerList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_post_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Layer):
    def __init__(self, encoder, out_channels=2, concat_input=False, use_attn=False):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input
        self.use_attn = use_attn
        self.up_conv1 = up_conv(self.n_channels, self.n_channels // 2)
        self.double_conv1 = double_conv(self.size[0], self.n_channels // 2)
        self.up_conv2 = up_conv(self.n_channels // 2, self.n_channels // 4)
        self.double_conv2 = double_conv(self.size[1], self.n_channels // 4)
        self.up_conv3 = up_conv(self.n_channels // 4, self.n_channels // 8)
        self.double_conv3 = double_conv(self.size[2], self.n_channels // 8)
        self.up_conv4 = up_conv(self.n_channels // 8, self.n_channels // 16)
        self.double_conv4 = double_conv(self.size[3], self.n_channels // 16)

        if self.concat_input:
            self.up_conv_input = up_conv(self.n_channels // 16, self.n_channels // 32)
            self.double_conv_input = double_conv(self.size[4], self.n_channels // 32)
        else:
            self.up_conv5 = up_conv(self.n_channels // 16, self.n_channels // 32)

        if self.use_attn:
            self.mhca_list = nn.LayerList([
                nn.MultiHeadAttention(2 ** 3, 2),
                nn.MultiHeadAttention(4 ** 3, 4),
                nn.MultiHeadAttention(8 ** 3, 8),
                nn.MultiHeadAttention(16 ** 3, 8),
                nn.MultiHeadAttention(16 ** 3, 8),
            ])
            self.mhsa = nn.MultiHeadAttention(2 ** 3, 1)
            self.conv_list1 = nn.LayerList([
                SingleConv(in_channels=112, out_channels=112, kernel_size=2,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=2),
                SingleConv(in_channels=40, out_channels=40, kernel_size=2,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=2),
                SingleConv(in_channels=24, out_channels=24, kernel_size=2,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=2),
                SingleConv(in_channels=16, out_channels=16, kernel_size=2,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=2),
                SingleConv(in_channels=4, out_channels=4, kernel_size=4,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=4)
            ])

            self.upconv_list = nn.LayerList([
                nn.Conv3DTranspose(112, 112, kernel_size=2, stride=2, ),
                nn.Conv3DTranspose(40, 40, kernel_size=2, stride=2, ),
                nn.Conv3DTranspose(24, 24, kernel_size=2, stride=2, ),
                nn.Conv3DTranspose(16, 16, kernel_size=2, stride=2, ),
                nn.Conv3DTranspose(4, 4, kernel_size=4, stride=4, ),
            ])
            self.conv_list2 = nn.LayerList([
                SingleConv(in_channels=self.n_channels,
                           out_channels=self.n_channels, kernel_size=1,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=1),
                SingleConv(in_channels=self.n_channels // 2, out_channels=self.n_channels // 2, kernel_size=1,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=1),
                SingleConv(in_channels=self.n_channels // 4, out_channels=self.n_channels // 4, kernel_size=1,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=1),
                SingleConv(in_channels=self.n_channels // 8, out_channels=self.n_channels // 8, kernel_size=1,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=1),
                SingleConv(in_channels=self.n_channels // 16, out_channels=self.n_channels // 16, kernel_size=2,
                           order='cbr', num_groups=8, padding=0, is3d=True, stride=2)
            ])

        self.final_conv = nn.Conv3D(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560, 'efficientnet-l2': 1280, }
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {
            'efficientnet-b0': [self.n_channels // 2 + 112, self.n_channels // 4 + 40, self.n_channels // 8 + 24,
                                self.n_channels // 16 + 16, self.n_channels // 32 + 4, self.n_channels // 32],
            'efficientnet-b1': [592, 296, 152, 80, 36, 32],
            'efficientnet-b2': [600, 304, 152, 80, 36, 32], 'efficientnet-b3': [608, 304, 160, 88, 36, 32],
            'efficientnet-b4': [624, 312, 160, 88, 36, 32], 'efficientnet-b5': [640, 320, 168, 88, 36, 32],
            'efficientnet-b6': [656, 328, 168, 96, 36, 32], 'efficientnet-b7': [672, 336, 176, 96, 36, 32],
            'efficientnet-l2': [672, 336, 176, 96, 36, 32]}
        return size_dict[self.encoder.name]

    def func(self, x1, x, i):
        if self.use_attn:
            x1 += PositionalEncodingPermute3D(x1.shape[1])(x1)
            x += PositionalEncodingPermute3D(x.shape[1])(x)
            conv_x = self.conv_list2[i](x).reshape((x.shape[0], x.shape[1], -1))
            conv_encoder_features = self.conv_list1[i](x1)
            conv_encoder_shape = conv_encoder_features.shape
            x1 *= self.upconv_list[i](
                nn.functional.sigmoid(
                    self.mhca_list[i](
                        conv_encoder_features.reshape((x1.shape[0], x1.shape[1], -1)),
                        conv_x,
                        conv_x
                    ).reshape(conv_encoder_shape)
                )
            )
        return x1, x

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        x = blocks.popitem()[1]

        if self.use_attn:
            original_x_shape = x.shape
            x += PositionalEncodingPermute3D(x.shape[1])(x)
            x = x.reshape((x.shape[0], x.shape[1], -1))
            x = self.mhsa(x, x)
            x = x.reshape(original_x_shape)

        x1 = blocks.popitem()[1]
        x1, x = self.func(x1, x, 0)
        x = self.up_conv1(x)
        x = paddle.concat([x, x1], axis=1)
        x = self.double_conv1(x)

        x2 = blocks.popitem()[1]
        x2, x = self.func(x2, x, 1)
        x = self.up_conv2(x)
        x = paddle.concat([x, x2], axis=1)
        x = self.double_conv2(x)

        x3 = blocks.popitem()[1]
        x3, x = self.func(x3, x, 2)
        x = self.up_conv3(x)
        x = paddle.concat([x, x3], axis=1)
        x = self.double_conv3(x)

        x4 = blocks.popitem()[1]
        x4, x = self.func(x4, x, 3)
        x = self.up_conv4(x)
        x = paddle.concat([x, x4], axis=1)
        x = self.double_conv4(x)

        if self.concat_input:
            input_, x = self.func(input_, x, 4)
            x = self.up_conv_input(x)
            x = paddle.concat([x, input_], axis=1)
            x = self.double_conv_input(x)
        else:
            x = self.up_conv5(x)

        x = self.final_conv(x)

        return x


class EfficientUNet3DWithSamplePoints(EfficientUnet):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels=64,
                 model_name='efficientnet-b7',
                 use_position_input: bool = True,
                 use_attn=True):
        super().__init__(Encoder(in_channels, model_name,
                                 *get_efficientnet_params(model_name, )),
                         out_channels=hidden_channels,
                         use_attn=use_attn)
        self.final_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_channels, out_features=out_channels)
        )
        self.use_position_input = use_position_input

    def forward(self, x, output_points):
        x = super(EfficientUNet3DWithSamplePoints, self).forward(x)
        output_points = output_points.unsqueeze(axis=2).unsqueeze(axis=2)
        x = paddle.nn.functional.grid_sample(x=x, grid=output_points,
                                             align_corners=False)
        x = x.squeeze(axis=3).squeeze(axis=3)
        x = x.transpose(perm=[0, 2, 1])
        x = self.final_mlp(x)
        return x

    def data_dict_to_input(self, data_dict):
        input_grid_features = data_dict['sdf'].unsqueeze(axis=1)

        if self.use_position_input:
            grid_points = data_dict['sdf_query_points']
            input_grid_features = paddle.concat(x=(input_grid_features,
                                                   grid_points), axis=1)
        output_points = data_dict['vertices']
        return input_grid_features, output_points

    @paddle.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_var = self(input_grid_features, output_points)
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
        input_grid_features, output_points = self.data_dict_to_input(data_dict)
        pred_var = self(input_grid_features, output_points)

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
    def count_params(model):
        c = 0
        for p in list(model.parameters()):
            c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
        return c


    import operator
    from functools import reduce

    model_name = 'efficientnet-b0'
    out_channels = 64
    in_channels = 4
    x = paddle.rand(shape=(8, in_channels, 64, 64, 64))
    encoder = Encoder(in_channels, model_name,
                      *get_efficientnet_params(model_name, ))
    model = EfficientUnet(
        encoder,
        out_channels=out_channels,
        concat_input=True,
        use_attn=True
    )
    # model = encoder
    print(count_params(model))
    print(model(x).shape)

    pass
