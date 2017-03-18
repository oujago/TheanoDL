# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

from collections import OrderedDict

import numpy as np
from theano import shared
from theano import tensor

from .variables import dtype


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

