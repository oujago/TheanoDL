# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

import json
import os


def set_config(configuration):
    with open(_config_path, 'w') as f:
        f.write(json.dumps(configuration, indent=4))


_thdl_base_dir = os.path.expanduser('~')
if not os.access(_thdl_base_dir, os.W_OK):
    _thdl_base_dir = '/tmp'

_thdl_dir = os.path.join(_thdl_base_dir, '.thdl')
if not os.path.exists(_thdl_dir):
    os.makedirs(_thdl_dir)

_config_path = os.path.expanduser(os.path.join(_thdl_dir, 'thdl.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))

else:
    _config = {
        'data': {
            'w2v': {

            }
        },
        'evaluation': {},
        'execution': {},
        'model': {}
    }
    set_config(_config)


def get_config():
    """
    Publicly accessible method for configuration.
    """
    return _config


def set_w2v(w2v_name, w2v_index, w2v_data):
    if w2v_name in _config['data']['w2v']:
        print("Overwrite existing word2vec configuration: %s" % w2v_name)
    _config['data']['w2v'][w2v_name] = (w2v_index, w2v_data)
    set_config(_config)


