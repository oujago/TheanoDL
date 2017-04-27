# -*- coding: utf-8 -*-

import pytest


def test_check_duplicate_path():
    from thdl.utils.file import check_duplicate_path

    not_exist_filepath = "./tests/utils/test_file2.py"
    assert check_duplicate_path(not_exist_filepath) == not_exist_filepath

    exist_filepath = './tests/utils/test_file.py'
    assert check_duplicate_path(exist_filepath) == './tests/utils/test_file(1).py'


