# -*- coding: utf-8 -*-


from collections import OrderedDict

from theano import function
from theano import tensor

from thdl.model.objective import get_loss
from thdl.model.optimizer import get_updates
from thdl.model.regularization import get_regularization


class Model(object):
    def __init__(self, l1=0., l2=0., loss='nll', optimizer='sgd', seed=23455, max_norm=False):
        self.l1 = l1
        self.l2 = l2
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed
        self.max_norm = max_norm

        self.lr = tensor.scalar()

        self.layers = []
        self.reg_params = []
        self.train_params = []
        self.masks = []
        self.updates = OrderedDict()
        self.train_test_split = False

        self.train = None
        self.predict = None

    def __check_train_test_split(self, layer):
        if type(layer).__name__ in ['Dropout', 'BatchNormal']:
            self.train_test_split = True

    def add(self, layer):
        self.layers.append(layer)
        self.__check_train_test_split(layer)

        self.reg_params += layer.reg_params
        self.train_params += layer.train_params
        self.updates.update(layer.updates)
        self.masks += layer.masks

    def __calc(self, train=True):
        assert type(self.layers[0]).__name__ == 'XY'

        input = self.layers[0].X
        output = self.layers[0].Y

        for layer in self.layers[1:]:
            if type(layer).__name__ in ['Dropout', 'BatchNormal']:
                input = layer(input, train=train)
            else:
                input = layer(input)

        prob_ys = input
        ys = tensor.argmax(prob_ys, axis=1)
        loss = get_loss(self.loss, prob_ys, output)
        return prob_ys, ys, loss

    def compile(self):

        # calc
        train_prob_ys, train_ys, train_loss = self.__calc(True)
        if self.train_test_split:
            predict_prob_ys, predict_ys, predict_loss = self.__calc(False)
        else:
            predict_prob_ys, predict_ys, predict_loss = train_prob_ys, train_ys, train_loss

        # l1, l2 loss
        l1_loss = get_regularization('l1', self.reg_params, self.l1)
        l2_loss = get_regularization('l2', self.reg_params, self.l2)
        # l2_loss = get_regularization('l2', self.layers[-1].train_params, self.l2)
        # l2_loss = get_regularization('l2', self.layers[-1].reg_params, self.l2)
        loss = [train_loss]
        if self.l1 > 0.:
            loss.append(l1_loss)
        if self.l2 > 0.:
            loss.append(l2_loss)

        # get updates
        updates = get_updates(self.optimizer, sum(loss), self.train_params, self.lr, self.max_norm)
        self.updates.update(updates)

        # train functions
        self.train = function(inputs=[self.layers[0].X, self.layers[0].Y, self.lr] + self.masks,
                              outputs=[train_loss, l1_loss, l2_loss, train_ys],
                              updates=updates, )

        # test functions
        self.predict = function(inputs=[self.layers[0].X, self.layers[0].Y] + self.masks,
                                outputs=[predict_loss, predict_ys])

    def to_json(self):
        layers_json = OrderedDict()
        for layer in self.layers:
            layers_json[str(layer)] = layer.to_json()

        config = {
            'model_json': {
                'l1': self.l1,
                'l2': self.l2,
                'loss': self.loss,
                'optimizer': self.optimizer,
                'seed': self.seed,
                'max_norm': self.max_norm,
            },
            'layers_json': layers_json
        }

        return config
