# -*- coding: utf-8 -*-

from theano import tensor


def get_loss(loss_type, predict_ys, actual_ys):
    if loss_type in ['bce', 'binary_crossentropy']:
        assert actual_ys.ndim == predict_ys.ndim
        return tensor.mean(tensor.nnet.binary_crossentropy(predict_ys, actual_ys))

    if loss_type in ['nll', 'negative_log_likelihood']:
        if actual_ys.ndim == predict_ys.ndim:
            return -tensor.mean(actual_ys * tensor.log(predict_ys) + (1 - actual_ys) * tensor.log(1 - predict_ys))
        else:
            return -tensor.mean(tensor.log(predict_ys)[tensor.arange(actual_ys.shape[0]), actual_ys])

    if loss_type in ['mse', 'mean_squared_error']:
        assert actual_ys.ndim == predict_ys.ndim
        return tensor.mean((predict_ys - actual_ys) ** 2)

    if loss_type in ['mae', 'mean_absolute_error']:
        assert actual_ys.ndim == predict_ys.ndim
        return tensor.mean(tensor.abs_(predict_ys - actual_ys), axis=-1)

    if loss_type in ['cce', 'categorical_crossentropy']:
        assert actual_ys.ndim == predict_ys.ndim
        return tensor.mean(tensor.nnet.categorical_crossentropy(predict_ys, actual_ys))

    raise ValueError('Unknown loss type: %s' % loss_type)
