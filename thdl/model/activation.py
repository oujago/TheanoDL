# -*- coding: utf-8 -*-

from theano import tensor


class Activation(object):
    def __call__(self, input):
        return self.call(input)

    def call(self, input):
        raise NotImplementedError()






def get_activation(name, **kwargs):
    """
    activation getting method

    :param name:
    :param kwargs:
    """
    if name == 'sigmoid':
        ''' Sigmoid activation function '''
        return tensor.nnet.sigmoid

    elif name == 'hard_sigmoid':
        ''' Hard_sigmoid activation function '''
        return tensor.nnet.hard_sigmoid

    elif name == 'tanh':
        ''' Tanh activation function '''
        return tensor.tanh

    elif name == 'scaled_tanh':
        ''' Scaled tanh :
        math:`\\varphi(x) = \\tanh(\\alpha \\cdot x) \\cdot \\beta`

        This is a modified tanh function which allows to rescale both the input and
        the output of the activation.

        Scaling the input down will result in decreasing the maximum slope of the
        tanh and as a result it will be in the linear regime in a larger interval
        of the input space. Scaling the input up will increase the maximum slope
        of the tanh and thus bring it closer to a step function.

        Scaling the output changes the output interval to :math:`[-\\beta,\\beta]`.

        Notes
        -----
        LeCun et al. (in [1]_, Section 4.4) suggest ``scale_in=2./3`` and
        ``scale_out=1.7159``, which has :math:`\\varphi(\\pm 1) = \\pm 1`,
        maximum second derivative at 1, and an effective gain close to 1.

        By carefully matching :math:`\\alpha` and :math:`\\beta`, the nonlinearity
        can also be tuned to preserve the mean and variance of its input:

          * ``scale_in=0.5``, ``scale_out=2.4``: If the input is a random normal
            variable, the output will have zero mean and unit variance.
          * ``scale_in=1``, ``scale_out=1.6``: Same property, but with a smaller
            linear regime in input space.
          * ``scale_in=0.5``, ``scale_out=2.27``: If the input is a uniform normal
            variable, the output will have zero mean and unit variance.
          * ``scale_in=1``, ``scale_out=1.48``: Same property, but with a smaller
            linear regime in input space.

        References
        ----------
        .. [1] LeCun, Yann A., et al. (1998):
           Efficient BackProp,
           http://link.springer.com/chapter/10.1007/3-540-49430-8_2,
           http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        .. [2] Masci, Jonathan, et al. (2011):
           Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction,
           http://link.springer.com/chapter/10.1007/978-3-642-21735-7_7,
           http://people.idsia.ch/~ciresan/data/icann2011.pdf
        '''
        scale_in = kwargs.get('scale_in', 0.5)
        scale_out = kwargs.get('scale_out', 2.4)
        return lambda x: tensor.tanh(x * scale_in) * scale_out

    elif name in ['relu', 'rectify']:
        ''' Rectify activation function '''
        return tensor.nnet.relu

    elif name in ['leaky_relu', 'leaky_rectify']:
        '''Leaky rectifier :math:`\\varphi(x) = \\max(\\alpha \\cdot x, x)`

        The leaky rectifier was introduced in [1]_. Compared to the standard
        rectifier :func:`rectify`, it has a nonzero gradient for negative input,
        which often helps convergence.

        Parameters
        ----------
        leakiness : float
            Slope for negative input, usually between 0 and 1.
            A leakiness of 0 will lead to the standard rectifier,
            a leakiness of 1 will lead to a linear activation function,
            and any value in between will give a leaky rectifier.

        See Also
        --------
        leaky_rectify: Instance with default leakiness of 0.01, as in [1]_.
        very_leaky_rectify: Instance with high leakiness of 1/3, as in [2]_.

        References
        ----------
        .. [1] Maas et al. (2013):
           Rectifier Nonlinearities Improve Neural Network Acoustic Models,
           http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
        .. [2] Graham, Benjamin (2014):
           Spatially-sparse convolutional neural networks,
           http://arxiv.org/abs/1409.6070

        '''
        leakiness = kwargs.get('leakiness', 0.01)
        return lambda x: tensor.nnet.relu(x, leakiness)

    elif name in ['very_leaky_rectify', 'very_leaky_relu']:
        return lambda x: tensor.nnet.relu(x, 1. / 3)

    elif name == 'elu':
        '''Exponential Linear Unit :math:`\\varphi(x) = (x > 0) ? x : e^x - 1`

        The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the
        linear rectifier :func:`rectify`, it has a mean activation closer to zero
        and nonzero gradient for negative input, which can help convergence.
        Compared to the leaky rectifier :class:`LeakyRectify`, it saturates for
        highly negative inputs.

        Notes
        -----
        In [1]_, an additional parameter :math:`\\alpha` controls the (negative)
        saturation value for negative inputs, but is set to 1 for all experiments.
        It is omitted here.

        References
        ----------
        .. [1] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
           Fast and Accurate Deep Network Learning by Exponential Linear Units
           (ELUs), http://arxiv.org/abs/1511.07289
        '''
        return lambda x: tensor.switch(x > 0, x, tensor.exp(x) - 1)

    elif name == 'softplus':
        ''' Softplus activation function '''
        return tensor.nnet.softplus

    elif name == 'softsign':
        ''' Softsign activation function '''
        from theano.sandbox import softsign
        return softsign

    elif name == 'softmax':
        ''' Softmax activation function '''
        return tensor.nnet.softmax

    elif name in ['linear', 'identity']:
        return lambda x: x

    else:
        raise ValueError("Unknown activation: ", name)
