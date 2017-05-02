# -*- coding: utf-8 -*-


from datetime import datetime


def now():
    """Get the format time of NOW.
    
    For example:
    
    >>> now()
    >>> "2017-04-26-16-44-56"
    
    :return: :class:`str` instance.
    """
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def today():
    """Get the format time of TODAY.
    
    For example:
    
    >>> today()
    >>> "2017-04-26"
    
    :return: :class:`str` instance.
    """
    return datetime.now().strftime('%Y-%m-%d')


def time_format(total_time):
    """Change the total time into the normal time format.
    
    For examples:
    
    >>> time_format(36)
    >>> "36 s"
    >>> time_format(90)
    >>> "1 min 30 s "
    >>> time_format(5420)
    >>> "1 h 30 min 20 s"
    >>> time_format(20.5)
    >>> "20 s 500 ms"
    
    :param total_time: :class:`float` or :class:`str` instance.
        The total seconds of the time. 
    :return: :class:`str` instance.
        The format string about time.
    """
    if total_time < 0:
        raise ValueError
    if total_time == 0:
        return ""
    if total_time < 1:
        return "{} ms".format(int(total_time * 1000))
    if total_time < 60:
        sec_integer = int(total_time)
        sec_decimal = total_time - sec_integer
        return ("{} s {}".format(sec_integer, time_format(sec_decimal))).strip()
    if total_time < 3600:
        min_integer = int(total_time / 60)
        sec_decimal = total_time - min_integer * 60
        return ("{} min {}".format(min_integer, time_format(sec_decimal))).strip()
    if total_time >= 3600:
        hour_integer = int(total_time / 60 / 60)
        sec_decimal = total_time - hour_integer * 60 * 60
        return ("{} h {}".format(hour_integer, time_format(sec_decimal))).strip()
    raise ValueError


def dict_to_str(dict_obj, js='-'):
    """Change dict object to string.
    
    :param dict_obj: :class:`dict` instance.
        The dict object to string.
    :param js: :class:`str` instance.
        The join symbol of keys and values.
    :return: :class:`str` instance.
        Return the string object.
    """
    if isinstance(dict_obj, dict):
        try:
            sorted_items = sorted(dict_obj.items())
        except TypeError:
            sorted_items = sorted([(str(key), value) for key, value in dict_obj.items()])
        res = []
        for key, value in sorted_items:
            res.append(str(key))
            if isinstance(value, dict):
                res.append(dict_to_str(value, js))
            else:
                res.append(str(value))
        return js.join(res)


def is_iterable(obj):
    """Check weather the input is an iterable object.
    
    :param obj: Any object.
    :return: :class:`boolean` instance. True or False.
    """

    if obj.__class__.__name__ in ['tuple', 'list']:
        return True
    else:
        return False



if __name__ == '__main__':
    a = {'a': 1, "b": {'c': 3, 'd': {"e": 5}}}
    print(dict_to_str(a))
