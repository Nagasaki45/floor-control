import itertools

import utils.iteration


def utterances_to_floor_intervals_gen(utterances_df):
    '''
    A state machine that yields floor intervals based on
    utterances intervals.
    '''
    intervals = utterances_df.sort_values('start_time').iterrows()
    _, cur = next(intervals)
    cur = cur.to_dict()
    while True:
        try:
            _, nex = next(intervals)
        except StopIteration:
            yield cur
            break
        nex = nex.to_dict()
        # Current and next one are same speaker -> merge
        if cur['participant'] == nex['participant']:
            cur = {
                'start_time': cur['start_time'],
                'end_time': nex['end_time'],
                'participant': cur['participant'],
            }
        # Current ends before next one starts -> output current
        elif cur['end_time'] <= nex['start_time']:
            yield cur
            cur = nex
        # Next is completely within current -> ignore it
        elif (
            nex['start_time'] >= cur['start_time']
            and nex['end_time'] <= cur['end_time']
        ):
            pass
        # Otherwise it's a partial overlap
        else:
            yield {
                'start_time': cur['start_time'],
                'end_time': nex['start_time'],
                'participant': cur['participant'],
            }
            cur = {
                'start_time': cur['end_time'],
                'end_time': nex['end_time'],
                'participant': nex['participant']
            }


def gen(utterances_df, sample_rate):
    '''
    Generate the annotated floor values (per timestamp)
    as described in figure 2 in the paper.
    '''
    yield from utils.iteration.intervals_to_values_gen(
        utterances_to_floor_intervals_gen(utterances_df),
        sample_rate=sample_rate,
        key='participant',
    )