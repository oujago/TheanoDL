# -*- coding: utf-8 -*-


from theano import tensor

from .base import Layer
from .convolution import Convolution
from .pooling import Pooling
from ..activation import Tanh


class NLPConvPooling(Layer):
    """
    Related Paper:
        Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform',
                 padding=(0, 0), ignore_border=True, mode='max'):
        super(NLPConvPooling, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.init = init
        self.padding = padding
        self.ignore_border = ignore_border
        self.mode = mode

        self.all_conv = []
        self.all_pooling = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], filter_heights[i], filter_width)
            self.all_conv.append(Convolution(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_conv[i].train_params)
            self.reg_params.extend(self.all_conv[i].reg_params)
            self.updates.update(self.all_conv[i].updates)

            pool_size = (image_shape[2] - filter_heights[i] + 1, 1)
            self.all_pooling.append(Pool2D(pool_size, padding, ignore_border, mode))
            self.train_params.extend(self.all_pooling[i].train_params)
            self.reg_params.extend(self.all_pooling[i].reg_params)
            self.updates.update(self.all_pooling[i].updates)


    def forward(self, input, **kwargs):
        outputs = []
        for i in range(len(self.filter_heights)):
            conv_out = self.all_conv[i](input)
            pool_out = self.all_pooling[i](conv_out)
            outputs.append(pool_out.flatten(2))

        return tensor.concatenate(outputs, axis=1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
            'padding': self.padding,
            'ignore_border': self.ignore_border,
            'mode': self.mode,
        }
        return config


class NLPConv(Layer):
    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform'):
        super(NLPConv, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.init = init

        self.truncat_len = image_shape[2] - max(filter_heights) + 1

        self.all_conv = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], filter_heights[i], filter_width)
            self.all_conv.append(Convolution(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_conv[i].train_params)
            self.reg_params.extend(self.all_conv[i].reg_params)
            self.updates.update(self.all_conv[i].updates)

    def __str__(self):
        return "NLPConv"

    def __call__(self, input):
        # if input.ndim == 3:
        #     input = input.dimshuffle(0, 'x', 1, 2)

        outputs = []
        for i in range(len(self.filter_heights)):
            # if self.filter_heights[i] > 1:
            #     zero = tensor.zeros(input.shape[:2] + (self.filter_heights[i]-1,) + input.shape[3:])
            #     conv_out = self.all_conv[i](tensor.concatenate([zero, input], axis=2))
            # else:
            #     conv_out = self.all_conv[i](input)
            conv_out = self.all_conv[i](input)
            output = conv_out.flatten(3)[:, :, -self.truncat_len:]
            output = output.dimshuffle(0, 2, 1)
            outputs.append(output)

        return tensor.concatenate(outputs, axis=-1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
        }
        return config


class NLPAsymConv(Layer):
    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform',
                 concat='truncate'):
        super(NLPAsymConv, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.init = init
        self.concat = concat

        self.all_row_conv = []
        self.all_col_conv = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], 1, filter_width)
            self.all_row_conv.append(Convolution(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_row_conv[i].train_params)
            self.reg_params.extend(self.all_row_conv[i].reg_params)
            self.updates.update(self.all_row_conv[i].updates)

            col_image_shape = (image_shape[0], self.filter_nums[i], image_shape[2], 1)
            filter_shape = (self.filter_nums[i], self.filter_nums[i], filter_heights[i], 1)
            self.all_col_conv.append(Convolution(rng, col_image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_col_conv[i].train_params)
            self.reg_params.extend(self.all_col_conv[i].reg_params)
            self.updates.update(self.all_col_conv[i].updates)

        if concat == 'truncate':
            self.truncat_len = image_shape[2] - max(filter_heights) + 1
        elif concat == '':
            pass
        else:
            raise ValueError("Unknown concatenate method: %s" % concat)

    def __call__(self, input):
        outputs = []
        for i in range(len(self.filter_heights)):
            row_conv_out = self.all_row_conv[i](input)
            col_conv_out = self.all_col_conv[i](row_conv_out)

            output = col_conv_out.flatten(3)[:, :, :self.truncat_len]
            output = output.dimshuffle(0, 2, 1)
            outputs.append(output)

        return tensor.concatenate(outputs, axis=-1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
            'concat': self.concat,
        }
        return config
