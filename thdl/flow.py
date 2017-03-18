# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""


import sys
import time
from collections import Counter

import numpy as np

from thdl.data.nlp_data import yield_item
from thdl.tool.other import time_format
from .data import DataCls
from .evaluation import EvalCls
from .execution import ExeCls


class Task:
    @staticmethod
    def _epoch_exe(func, y_num, **kwargs):
        """
        This is one epoch's steps:
            First train or predict, Then evaluate.

        :param func: model train function or model predict function.
        :param train: If this is epoch train execution, then return one more thing - loss.
        :param kwargs: function 'func' parameters
        """
        # train or predict  function
        predictions, origins, loss = func(**kwargs)
        loss = np.mean(np.asarray(loss[:-1]), axis=0)  # if train, loss is list, if predict, loss is number

        # evaluate
        confusion_matrix = EvalCls.get_confusion_matrix(predictions, origins, y_num)
        evaluation_matrix = EvalCls.get_evaluation_matrix(confusion_matrix)

        # return
        return confusion_matrix, evaluation_matrix, loss

    @staticmethod
    def hold_out_validation(model_cls, data_cls, execute_cls, evaluate_cls, file=sys.stdout, name=None):
        """
        losses: may include train loss, L1 loss, L2 loss
        *_acc: accuracy
        *_maf1: macro F1
        *_val_mat: confusion matrix
        *_f1_mat: F1 matrix


        :param model_cls: the model class, to train or predict
        :param data_cls: the data class, to provide data
        :param execute_cls: the execution class, to provide whole data training and predicting
        :param evaluate_cls: the evaluation class, to provide all kinds of evaluation methods
        :param file: The output file
        :param name: Hold out validation name, if None, name = 'default'
        """

        ##############################
        # Preparation
        ##############################
        # checking
        assert isinstance(data_cls, DataCls)
        assert isinstance(execute_cls, ExeCls)
        assert isinstance(evaluate_cls, EvalCls)
        assert hasattr(model_cls, 'compile') and hasattr(model_cls, 'to_json')
        assert hasattr(model_cls, 'predict') and hasattr(model_cls, 'train')

        # variables
        t0 = time.time()
        stdout = sys.stdout
        sys.stdout = file
        history_name = name or 'default'
        y_num = len(evaluate_cls.index2tag)
        shuffle_rng = np.random.RandomState(execute_cls.shuffle_seed)

        ##############################
        # Parameter logging
        ##############################
        print("", file=file)
        for name, cls in (('Model Class', model_cls), ("Data Class", data_cls),
                          ("Evaluation Class", evaluate_cls), ("Execute Class", execute_cls)):
            print('%s Parameter:' % name, file=file)
            for key, value in sorted(cls.to_json().items()):
                if isinstance(value, dict):
                    print("\t%s: " % key, file=file)
                    for key2, value2 in value.items():
                        print("\t\t%s = %s" % (key2, value2), file=file)
                else:
                    print("\t%s = %s" % (key, value), file=file)
            print("", file=file)
        file.flush()

        ##############################
        # Data getting
        ##############################
        # X: input
        all_xs = data_cls.get_xs()
        train_xs = all_xs['train']
        valid_xs = all_xs['valid']
        test_xs = all_xs['test']

        # Y: output
        all_ys = data_cls.get_ys()
        train_ys = all_ys['train']
        valid_ys = all_ys['valid']
        test_ys = all_ys['test']

        # count
        index2tag = evaluate_cls.index2tag
        for aspect in ['train', 'valid', 'test']:
            print("In %s data:" % aspect, file=file)
            print("\tTotal data number - %d" % len(all_ys[aspect]), file=file)
            counter = Counter(yield_item(all_ys[aspect]))
            for key, value in sorted(counter.items()):
                print("\t%s data number - %d" % (index2tag[key], value), file=file)
            print("", file=file)

        ##############################
        # Model
        ##############################
        t1 = time.time()
        print("building model ...", file=file)
        model_cls.compile()
        print("building done, used %s.\n" % time_format(time.time() - t1), file=file)

        ##############################
        # Execution
        ##############################
        for epoch in range(execute_cls.epochs):
            # epoch variables
            t1 = time.time()
            lr = execute_cls.lr

            # start work
            if execute_cls.shuffle:
                s = shuffle_rng.randint(0, 99999)
                np.random.seed(s)  # definitely important
                # train_xs = np.random.permutation(train_xs)
                np.random.shuffle(train_xs)
                np.random.seed(s)  # definitely important
                # train_ys = np.random.permutation(train_ys)
                np.random.shuffle(train_ys)

            # training
            confusion_mat, evaluation_mat, loss = Task._epoch_exe(
                execute_cls.exe_train, y_num, model=model_cls, all_xs=train_xs, all_ys=train_ys, lr=lr)
            if 'training' in evaluate_cls.aspects:
                evaluate_cls.add_history(history_name, 'training', confusion_mat, evaluation_mat, loss)

            # trained
            if 'trained' in evaluate_cls.aspects:
                confusion_mat, evaluation_mat, loss = Task._epoch_exe(
                    execute_cls.exe_predict, y_num, model=model_cls, all_xs=train_xs, all_ys=train_ys)
                evaluate_cls.add_history(history_name, 'trained', confusion_mat, evaluation_mat, loss)

            # valid
            if 'valid' in evaluate_cls.aspects:
                confusion_mat, evaluation_mat, loss = Task._epoch_exe(
                    execute_cls.exe_predict, y_num, model=model_cls, all_xs=valid_xs, all_ys=valid_ys)
                evaluate_cls.add_history(history_name, 'valid', confusion_mat, evaluation_mat, loss)
            else:
                raise OSError("Model must valid.")

            # test
            if 'test' in evaluate_cls.aspects:
                confusion_mat, evaluation_mat, loss = Task._epoch_exe(
                    execute_cls.exe_predict, y_num, model=model_cls, all_xs=test_xs, all_ys=test_ys)
                evaluate_cls.add_history(history_name, 'test', confusion_mat, evaluation_mat, loss)
            else:
                raise OSError("Model must test.")

            # end work
            execute_cls.lr = lr * execute_cls.decay
            evaluate_cls.output_epoch(history_name, epoch, file=file)
            print("Used time %s" % time_format(time.time() - t1))
            file.flush()
        else:
            evaluate_cls.output_bests(history_name, file=file)

        # end end work
        print("Used time %s" % time_format(time.time() - t0), file=file)
        file.flush()
        sys.stdout = stdout

    @staticmethod
    def cv_plot(evaluate, file_path, tags=None, file=sys.stdout):
        evaluate.output_total_bests(file=file)

        t0 = time.time()
        evaluate.plot_history_losses("%s-losses.png" % file_path)
        evaluate.plot_history_evaluations("%s-evals.png" % file_path, mark_best=True)
        if tags is not None:
            for tag in tags:
                evaluate.plot_history_evaluations('%s-%s.png' % (file_path, tag), tag)
        evaluate.plot_bests('%s-bests.png' % file_path)
        print("Plotting used time %s" % time_format(time.time() - t0), file=file)

