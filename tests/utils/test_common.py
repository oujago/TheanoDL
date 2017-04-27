# -*- coding: utf-8 -*-

import pytest


def test_now():
    from thdl.utils.common import now

    assert now().count('-') == 5


def test_today():
    from thdl.utils.common import today

    assert today().count('-') == 2


def test_time_format():
    from thdl.utils.common import time_format

    assert time_format(36) == '36 s'
    assert time_format(90) == "1 min 30 s"
    assert time_format(5420) == "1 h 30 min 20 s"
    assert time_format(20.5) == "20 s 500 ms"


def test_dict_to_str():
    from thdl.utils.common import dict_to_str

    test_dict0 = {'a': 1, "1": 'a'}
    assert dict_to_str(test_dict0) == '1-a-a-1'

    test_dict1 = {'a': 1, 1: 'a'}
    assert dict_to_str(test_dict1) == '1-a-a-1'

    test_dict2 = {'a': 1, "b": {'c': 3, 'd': {"e": 5}}}
    assert dict_to_str(test_dict2) == 'a-1-b-c-3-d-e-5'
