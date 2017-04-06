# -*- coding: utf-8 -*-


import math
import os

from datetime import datetime


def now():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def today():
    return datetime.now().strftime('%Y-%m-%d')


def time_format(total_time):
    if total_time > 3600:
        return "%f h" % (total_time / 3600)
    elif total_time > 60:
        return "%f min" % (total_time / 60)
    else:
        return "%f s" % total_time
