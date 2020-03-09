def get_lr_func(type, **kwargs):
    if type == 'linear':
        return lr_linear(**kwargs)
    elif type == 'const':
        return lr_const(**kwargs)

    return None


def lr_const(max=3e-4, min=3e-4):
    assert max == min

    def f(x):
        return max

    return f


def lr_linear(max=1e-3, min=3e-4):
    """
    x in [0, 1]: 1 is beginning; 0 is the end
    """
    assert max >= min

    def f(x):
        return (x - 1) * (max - min) + max

    return f

