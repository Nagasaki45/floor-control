import itertools


def dedup(iterable, key=lambda x: x):
    '''
    Generates same iterable without duplications.

        list(dedup([1, 1, 2, 1, 2, 2, 3])) == [1, 2, 1, 2, 3]
    '''
    it = iter(iterable)
    current = next(it)
    yield current
    for item in it:
        if key(current) != key(item):
            current = item
            yield current


def pairwise(iterable):
    '''
    Generates consecutive pairs from iterable.

        list(pairwise([1, 2, 3, 4]) == [(1, 2), (2, 3), (3, 4)]
    '''
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
