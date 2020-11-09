import itertools


def dedup(iterable, key=lambda x: x):
    '''
    Generates same iterable without duplications.

        list(dedup([1, 1, 2, 1, 2, 2, 3])) == [1, 2, 1, 2, 3]
    '''
    it = iter(iterable)
    try:
        current = next(it)
    except StopIteration:
        return
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


def intervals_to_values_gen(intervals, sample_rate, key):
    '''
    Takes (sorted and non-overlapping) intervals and yields
    an array with values from key (or NaN).

    Example:

    >>> intervals = [
    ...     {
    ...         'start_time': 0,
    ...         'end_time': 0.5,
    ...         'result': 'X',
    ...     },
    ...     {
    ...         'start_time': 1,
    ...         'end_time': 1.2,
    ...         'result': 'Y',
    ...     },
    ... ]
    >>> gen = intervals_to_values_gen(
    ...     intervals,
    ...     sample_rate=4,
    ...     key='result',
    ... )
    >>> list(gen)
    ['X', 'X', 'X', nan, 'Y']
    '''
    intervals = iter(intervals)
    interval = next(intervals)
    for i in itertools.count(step=1 / sample_rate):
        if i > interval['end_time']:
            try:
                interval = next(intervals)
            except StopIteration:
                break
        if i < interval['start_time']:
            yield float('nan')
        else:
            yield interval[key]
