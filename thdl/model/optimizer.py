# -*- coding: utf-8 -*-


from collections import OrderedDict

import numpy as np
from theano import shared
from theano import tensor

from .utils.random import get_dtype


class Optimizer(object):
    """
    Object to generate Theano update dictionaries for training.

    The update functions implement different methods to control the learning
    rate for use with stochastic gradient descent.

    Update functions take a loss expression or a list of gradient expressions and
    a list of parameters as input and return an ordered dictionary of updates:
    """

    def __init__(self, learning_rate, clip_norm=0., max_norm=0.):
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.max_norm = max_norm

    def __call__(self, params, cost):
        return self.get_updates(params, cost)

    def get_updates(self, params, cost):
        grads = tensor.grad(cost=cost, wrt=params)

        if self.max_norm > 0.:
            norm = tensor.sqrt(tensor.sum([tensor.sum(g ** 2) for g in grads]))
            if tensor.ge(norm, self.max_norm):
                grads = [g * self.max_norm / norm for g in grads]

        if self.clip_norm > 0.:
            grads = [tensor.clip(g, 0, self.clip_norm) for g in grads]

        return grads


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    """

    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)

    def get_updates(self, params, cost):
        updates = OrderedDict()
        grads = super(SGD, self).get_updates(params, cost)
        for param, grad in zip(params, grads):
            updates[param] = param - self.learning_rate * grad
        return updates


class Momentum(Optimizer):
    """
    Stochastic Gradient Descent (SGD) updates with momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    """

    def __init__(self, momentum=0.9, **kwargs):
        self.momentum = momentum
        super(Momentum, self).__init__(**kwargs)

    def get_updates(self, params, cost):
        grads = super(Momentum, self).get_updates(params, cost)

        updates = OrderedDict()

        for param, grad in zip(params, grads):
            updates[param] = param - self.learning_rate * grad

        for param in params:
            value = param.get_value(borrow=True)
            velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            x = self.momentum * velocity + updates[param]
            updates[velocity] = x - param
            updates[param] = x

        return updates


class NesterovMomentum(Optimizer):
    """
    Stochastic Gradient Descent (SGD) updates with Nesterov momentum

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
    """

    def __init__(self, momentum=0.9, **kwargs):
        self.momentum = momentum
        super(NesterovMomentum, self).__init__(**kwargs)

    def get_updates(self, params, cost):
        grads = super(NesterovMomentum, self).get_updates(params, cost)

        updates = OrderedDict()

        for param, grad in zip(params, grads):
            updates[param] = param - self.learning_rate * grad

        for param in params:
            value = param.get_value(borrow=True)
            velocity = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            x = self.momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = self.momentum * x + updates[param]

        return updates


class Adagrad(Optimizer):
    """
    Adagrad updates

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
           Adaptive subgradient methods for online learning and stochastic optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    def __init__(self, epsilon, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        grads = super(Adagrad, self).get_updates(params, cost)

        updates = OrderedDict()

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self.learning_rate * grad / tensor.sqrt(accu_new + self.epsilon))

        return updates


class RMSprop(Optimizer):
    """
    RMSProp updates

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
    """

    def __init__(self, rho=0.9, epsilon=1e-6, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        grads = super(RMSprop, self).get_updates(params, cost)

        updates = OrderedDict()
        one = tensor.constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            accu_new = self.rho * accu + (one - self.rho) * grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self.learning_rate * grad / tensor.sqrt(accu_new + self.epsilon))

        return updates


class Adadelta(Optimizer):
    """
    Adadelta updates

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
           ADADELTA: An Adaptive Learning Rate Method. arXiv Preprint arXiv:1212.5701.
    """

    def __init__(self, rho=0.9, epsilon=1e-6, **kwargs):
        super(Adadelta, self).__init__(**kwargs)

        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        grads = super(Adadelta, self).get_updates(params, cost)

        updates = OrderedDict()
        one = tensor.constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            # update accu (as in rmsprop)
            accu_new = self.rho * accu + (one - self.rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            update = (grad * tensor.sqrt(delta_accu + self.epsilon) / tensor.sqrt(accu_new + self.epsilon))
            updates[param] = param - self.learning_rate * update

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = self.rho * delta_accu + (one - self.rho) * update ** 2
            updates[delta_accu] = delta_accu_new

        return updates


class Adam(Optimizer):
    """
    Adam updates

    Adam updates implemented as in [1]_.

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.
    """

    def __init__(self, beta1, beta2, epsilon, **kwargs):
        super(Adam, self).__init__(**kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        grads = super(Adam, self).get_updates(params, cost)

        updates = OrderedDict()
        t_prev = shared(np.asarray(0., dtype=get_dtype()))
        one = tensor.constant(1)
        t = t_prev + 1
        a_t = self.learning_rate * tensor.sqrt(one - self.beta2 ** t) / (one - self.beta1 ** t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = self.beta1 * m_prev + (one - self.beta1) * g_t
            v_t = self.beta2 * v_prev + (one - self.beta2) * g_t ** 2
            step = a_t * m_t / (tensor.sqrt(v_t) + self.epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t

        return updates


class Adamax(Optimizer):
    """
    Adamax updates

    Adamax updates implemented as in [1]_. This is a variant of of the Adam
    algorithm based on the infinity norm.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.
    """

    def __init__(self, beta1, beta2, epsilon, **kwargs):
        super(Adamax, self).__init__(**kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        grads = super(Adamax, self).get_updates(params, cost)

        t_prev = shared(np.asarray(0., dtype=get_dtype()))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = tensor.constant(1)

        t = t_prev + 1
        a_t = self.learning_rate / (one - self.beta1 ** t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            u_prev = shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

            m_t = self.beta1 * m_prev + (one - self.beta1) * g_t
            u_t = tensor.maximum(self.beta2 * u_prev, abs(g_t))
            step = a_t * m_t / (u_t + self.epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = param - step

        updates[t_prev] = t

        return updates
