# -*- coding: utf-8 -*-


from theano import tensor

from .base import Layer


class Flatten(Layer):
    """Flatten a tensor.

    Flattens a tensor to `outdim` dimensions by preserving the leading
    outdim - 1 shape components.

    .. note:: The interface Flatten(Op) is deprecated, you should use flatten.
    """
    def __init__(self, ndim=2):
        self.ndim = ndim

    def forward(self, input, **kwargs):
        return input.flatten(self.ndim)

    def to_json(self):
        config = {
            'ndim': self.ndim
        }
        return config


class Reshape(Layer):
    """Reshape a tensor.

    """
    def __init__(self, newshape):
        self.newshape = newshape

    def forward(self, input, **kwargs):
        return tensor.reshape(input, newshape=self.newshape)

    def to_json(self):
        config = {
            "newshape": self.newshape
        }
        return config


class Mean(Layer):
    """
    Computes the mean value along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis : None or int or (list of int) (see `Sum`)
        Compute the mean along this axis of the tensor.
        None means all axes (like numpy).
    dtype: None or string
        Dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    keepdims: bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    acc_dtype: None or string
        Dtype to use for the inner summation. This will not
        necessarily be the dtype of the output (in particular
        if it is a discrete (int/uint) dtype, the output will
        be in a float type). If None, then we use the same rules as `sum()`.

    Notes
    -----
    For gpu, if you specify dtype=float32, everything will be done on the gpu.
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, input, **kwargs):
        return tensor.mean(input, axis=self.axis)

    def to_json(self):
        config = {
            "axis": self.axis
        }
        return config


class Dimshuffle(Layer):
    """Reorder the dimensions of this variable, optionally inserting
    broadcasted dimensions.

    Parameters
    ----------
    pattern
        List/tuple of int mixed with 'x' for broadcastable dimensions.

    Examples
    --------
    For example, to create a 3D view of a [2D] matrix, call
    ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
    middle dimension is an implicit broadcasted dimension.  To do the same
    thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

    Notes
    -----
    This function supports the pattern passed as a tuple, or as a
    variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
    to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
    mixed with 'x' characters).

    See Also
    --------
    DimShuffle
    """
    def __init__(self, pattern):
        self.pattern = pattern

    def forward(self, input, **kwargs):
        return input.dimshuffle(self.pattern)

    def to_json(self):
        config = {
            'pattern': self.pattern
        }
        return config


class Concatenate(Layer):
    """Use :func:`tensor.concatenate()` to join multiple inputs.
    
    Parameters
    ----------
    axis: int
        The axis to concatenate.
    
    Raises
    ------
    TypeError
        The tensor_list must be a tuple or list.

    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    
    """
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, inputs, **kwargs):
        return tensor.concatenate(inputs, axis=self.axis)

    def to_json(self):
        cconfig = {
            'axis': self.axis
        }
        return cconfig

