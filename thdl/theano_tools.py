# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/11/5

@notes:
    
"""

from collections import OrderedDict

import numpy as np
from theano import shared
from theano import tensor

########################################
# variables and functions
########################################

dtype = tensor.config.floatX


########################################
# updates
########################################


def get_updates(optimizer, cost, params, learning_rate, max_norm=False, **kwargs):
    """ Functions to generate Theano update dictionaries for training.

    The update functions implement different methods to control the learning
    rate for use with stochastic gradient descent.

    Update functions take a loss expression or a list of gradient expressions and
    a list of parameters as input and return an ordered dictionary of updates:

    :param optimizer:
    :param cost:
    :param params:
    :param learning_rate:
    :param max_norm:
    :param kwargs:
    """

    grads = tensor.grad(cost=cost, wrt=params)
    if max_norm is not False:
        norm = tensor.sqrt(tensor.sum([tensor.sum(g ** 2) for g in grads]))
        if tensor.ge(norm, max_norm):
            grads = [g * max_norm / norm for g in grads]

    updates = OrderedDict()

    if optimizer in ['sgd', 'sgd_vanilla']:
        '''Stochastic Gradient Descent (SGD) updates

        Generates update expressions of the form:
        * ``param := param - learning_rate * gradient``
        '''

        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

    elif optimizer == 'momentum':
        '''Stochastic Gradient Descent (SGD) updates with momentum

        Generates update expressions of the form:

        * ``velocity := momentum * velocity - learning_rate * gradient``
        * ``param := param + velocity``

        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

        See Also
        --------
        nesterov_momentum : Nesterov's variant of SGD with momentum
        '''

        momentum = kwargs.get('momentum', 0.9)

        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

        for param in params:
            value = param.get_value(borrow=True)
            velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param]
            updates[velocity] = x - param
            updates[param] = x

    elif optimizer == 'nesterov_momentum':
        '''Stochastic Gradient Descent (SGD) updates with Nesterov momentum

        Generates update expressions of the form:

        * ``velocity := momentum * velocity - learning_rate * gradient``
        * ``param := param + momentum * velocity - learning_rate * gradient``

        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

        The classic formulation of Nesterov momentum (or Nesterov accelerated
        gradient) requires the gradient to be evaluated at the predicted next
        position in parameter space. Here, we use the formulation described at
        https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
        which allows the gradient to be evaluated at the current parameters.
        '''

        momentum = kwargs.get('momentum', 0.9)

        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

        for param in params:
            value = param.get_value(borrow=True)
            velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]

    elif optimizer == 'adagrad':
        '''Adagrad updates

        Scale learning rates by dividing with the square root of accumulated
        squared gradients. See [1]_ for further description.

        Notes
        -----
        Using step size eta Adagrad calculates the learning rate for feature i at
        time step t as:

        .. math:: \\eta_{t,i} = \\frac{\\eta}
           {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

        as such the learning rate is monotonically decreasing.

        Epsilon is not included in the typical formula, see [2]_.

        References
        ----------
        .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
               Adaptive subgradient methods for online learning and stochastic
               optimization. JMLR, 12:2121-2159.

        .. [2] Chris Dyer:
               Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
        '''

        epsilon = kwargs.get('epsilon', 1e-6)
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (learning_rate * grad / tensor.sqrt(accu_new + epsilon))

    elif optimizer == 'rmsprop':
        '''RMSProp updates

        Scale learning rates by dividing with the moving average of the root mean
        squared (RMS) gradients. See [1]_ for further description.

        Notes
        -----
        `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving average
        fast.

        Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
        learning rate :math:`\\eta_t` is calculated as:

        .. math::
           r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
           \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

        References
        ----------
        .. [1] Tieleman, T. and Hinton, G. (2012):
               Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
               Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
        '''

        rho = kwargs.get('rho', 0.9)
        epsilon = kwargs.get('epsilon', 1e-6)
        one = tensor.constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (learning_rate * grad / tensor.sqrt(accu_new + epsilon))

    elif optimizer == 'adadelta':
        ''' Adadelta updates

        Scale learning rates by the ratio of accumulated gradients to accumulated
        updates, see [1]_ and notes for further description.

        Notes
        -----
        rho should be between 0 and 1. A value of rho close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving average
        fast.

        rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
        work for multiple datasets (MNIST, speech).

        In the paper, no learning rate is considered (so learning_rate=1.0).
        Probably best to keep it at this value.
        epsilon is important for the very first update (so the numerator does
        not become 0).

        Using the step size eta and a decay factor rho the learning rate is
        calculated as:

        .. math::
           r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
           \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                                 {\sqrt{r_t + \epsilon}}\\\\
           s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2

        References
        ----------
        .. [1] Zeiler, M. D. (2012):
               ADADELTA: An Adaptive Learning Rate Method.
               arXiv Preprint arXiv:1212.5701.
        '''

        rho = kwargs.get('rho', 0.9)
        epsilon = kwargs.get('epsilon', 1e-6)
        # Using theano constant to prevent upcasting of float32
        one = tensor.constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            update = (grad * tensor.sqrt(delta_accu + epsilon) / tensor.sqrt(accu_new + epsilon))
            updates[param] = param - learning_rate * update

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates[delta_accu] = delta_accu_new

    elif optimizer == 'adam':
        ''' Adam updates

        Adam updates implemented as in [1]_.

        Notes
        -----
        The paper [1]_ includes an additional hyperparameter lambda. This is only
        needed to prove convergence of the algorithm and has no practical use
        (personal communication with the authors), it is therefore omitted here.

        References
        ----------
        .. [1] Kingma, Diederik, and Jimmy Ba (2014):
               Adam: A Method for Stochastic Optimization.
               arXiv preprint arXiv:1412.6980.
        '''

        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)

        t_prev = shared(np.asarray(0., dtype=dtype))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = tensor.constant(1)

        t = t_prev + 1
        a_t = learning_rate * tensor.sqrt(one - beta2 ** t) / (one - beta1 ** t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = beta1 * m_prev + (one - beta1) * g_t
            v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
            step = a_t * m_t / (tensor.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t

    elif optimizer == 'adamax':
        ''' Adamax updates

        Adamax updates implemented as in [1]_. This is a variant of of the Adam
        algorithm based on the infinity norm.

        References
        ----------
        .. [1] Kingma, Diederik, and Jimmy Ba (2014):
               Adam: A Method for Stochastic Optimization.
               arXiv preprint arXiv:1412.6980.
        '''

        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)

        t_prev = shared(np.asarray(0., dtype=dtype))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = tensor.constant(1)

        t = t_prev + 1
        a_t = learning_rate / (one - beta1 ** t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            u_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = beta1 * m_prev + (one - beta1) * g_t
            u_t = tensor.maximum(beta2 * u_prev, abs(g_t))
            step = a_t * m_t / (u_t + epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = param - step

        updates[t_prev] = t

    else:
        return ValueError('Unknown optimizer: ', optimizer)

    return updates


########################################
# initializations
########################################


def get_shared(rng, size, init, dim_ordering='th', borrow=True, name=None, **kwargs):
    """
    Initialization getting method

    :param rng:
    :param size:
    :param init:
    :param dim_ordering:
    :param borrow:
    :param name:
    :param kwargs:
    """
    if init == 'zero':
        return shared(value=np.asarray(np.zeros(size), dtype=dtype), name=name, borrow=borrow)

    if init == 'one':
        return shared(value=np.asarray(np.ones(size), dtype=dtype), name=name, borrow=borrow)

    ##########################################
    #               Get Fans                 #
    # adapted from keras/initializations.py  #
    ##########################################
    if len(size) == 2:
        fan_in = size[0]
        fan_out = size[1]

    elif len(size) == 4 or len(size) == 5:
        # assuming convolution kernels (2D or 3D)
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering.lower() == 'th':
            respective_field_size = np.prod(size[2:])
            fan_in = size[1] * respective_field_size
            fan_out = size[0] * respective_field_size
        elif dim_ordering.lower() == 'tf':
            respective_field_size = np.prod(size[:-2])
            fan_in = size[-2] * respective_field_size
            fan_out = size[-1] * respective_field_size
        else:
            raise ValueError("Invalid dim_ordering: ", dim_ordering)

    else:
        fan_in = np.sqrt(np.prod(size))
        fan_out = np.sqrt(np.prod(size))

    ##########################################
    #               Get Values               #
    # adapted from keras/initializations.py  #
    ##########################################
    if init == 'uniform':
        scale = kwargs.get('scale', 0.05)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'normal':
        scale = kwargs.get('scale', 0.05)
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'lecun_uniform':
        ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf '''
        scale = np.sqrt(3. / fan_in)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'glorot_normal':
        ''' Reference: Glorot & Bengio, AISTATS 2010 '''
        scale = np.sqrt(2. / (fan_in + fan_out))
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'glorot_uniform':
        scale = np.sqrt(6. / (fan_in + fan_out))
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'he_normal':
        ''' Reference:  He et al., http://arxiv.org/abs/1502.01852 '''
        scale = np.sqrt(2. / fan_in)
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'he_uniform':
        scale = np.sqrt(6. / fan_in)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'orthogonal':
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120 '''
        flat_shape = (size[0], np.prod(size[1:]))
        a = rng.normal(loc=0.0, scale=1.0, size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        value = q.reshape(size)

    elif init == 'rand':
        value = rng.rand(*size) * 0.2 + 0.1

    else:
        raise ValueError("unknown init type: %s" % init)

    value = np.asarray(value, dtype=dtype)
    return shared(value=value, name=name, borrow=borrow)


########################################
# activations
########################################


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


########################################
# losses
########################################


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


########################################
# common variables
########################################

def get_clstm_variables(rng, n_in, n_out, init, inner_init):
    """
    Coupled LSTM
    """
    i_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    i_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    i_h_b = get_shared(rng, size=(n_out,), init='zero')

    g_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    g_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    g_h_b = get_shared(rng, size=(n_out,), init='zero')

    o_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    o_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    o_h_b = get_shared(rng, size=(n_out,), init='zero')

    return i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_lstm_variables(rng, n_in, n_out, init, inner_init):
    """
    Standard LSTM
    """
    f_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='one')

    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_clstm_variables(rng, n_in, n_out, init, inner_init)

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_plstm_variables(rng, n_in, n_out, init, inner_init, peephole_init):
    """
    Peephole LSTM
    """
    f_x2h_W, f_h2h_W, f_h_b, \
    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_lstm_variables(rng, n_in, n_out, init, inner_init)

    p_f = get_shared(rng, (n_out, ), init=peephole_init)
    p_i = get_shared(rng, (n_out,), init=peephole_init)
    p_o = get_shared(rng, (n_out,), init=peephole_init)

    return f_x2h_W, f_h2h_W, p_f, f_h_b, \
           i_x2h_W, i_h2h_W, p_i, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, p_o, o_h_b


def get_gru_variables(rng, n_in, n_out, init, inner_init):
    r_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    r_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    r_h_b = get_shared(rng, size=(n_out,), init='zero')

    z_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    z_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    z_h_b = get_shared(rng, size=(n_out,), init='zero')

    f_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='zero')

    return r_x2h_W, r_h2h_W, r_h_b, \
           z_x2h_W, z_h2h_W, z_h_b, \
           f_x2h_W, f_h2h_W, f_h_b


def get_mgu_variables(rng, n_in, n_out, init, inner_init):
    f_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='zero')

    i_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    i_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    i_h_b = get_shared(rng, size=(n_out,), init='zero')

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b


def get_rnn_variables(rng, n_in, n_out, init, inner_init):
    x2o = get_shared(rng, size=(n_in, n_out), init=init)
    o2o = get_shared(rng, size=(n_out, n_out), init=inner_init)
    b = get_shared(rng, size=(n_out,), init='zero')

    return x2o, o2o, b


########################################
# regularization
########################################


def get_regularization(regularizer, params, scale):
    if scale == 0.:
        return tensor.sum(shared(value=np.array(0., dtype=dtype)))

    regularizer = regularizer.lower()

    if regularizer == 'l1':
        return tensor.sum([tensor.sum(tensor.abs_(ps)) for ps in params]) * scale

    elif regularizer == 'l2':
        return tensor.sum([tensor.sum(tensor.sqr(ps)) for ps in params]) * scale

    else:
        raise ValueError("Unknown regularizer: %s" % regularizer)