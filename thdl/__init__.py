# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 2016/11/8

@notes:
    
"""


backend = None


def use(framework):
    global backend

    if framework.lower() == 'theano':
        backend = 'theano'

    if framework.lower() in ['tf', 'tensorflow']:
        backend = 'tensorflow'

    if framework.lower() == 'neon':
        backend = 'neon'


def get():
    return backend



