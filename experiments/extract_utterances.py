import pathlib

import numpy as np
import pandas as pd

import utils.duel
import utils.path

ANNOTATIONS_TIERS = ['A-utts', 'B-utts']
BACKCHANNEL_WORDS = {'ja', 'okay', 'ohm', 'mhm', 'genau'}
MAX_BACKCHANNEL_DURATION = 0.5
OUT_DIR = pathlib.Path('features') / 'utterances'


def is_backchannel(interval):
    words = [w.lower() for w in interval.text.split()]
    duration = interval.end_time - interval.start_time
    short_enough = duration < MAX_BACKCHANNEL_DURATION
    return all(w in BACKCHANNEL_WORDS for w in words) and short_enough


def main():
    utils.path.empty_dir(OUT_DIR)

    data_gen = utils.duel.load_sessions_gen()

    for session in data_gen:
        tg = session['textgrid']

        for part in session['parts']:
            out_filepath = OUT_DIR / f'{session["name"]}-{part["name"]}.csv'
            print(f'Generating {out_filepath}')

            utterances = []
            for participant, tier_name in enumerate(ANNOTATIONS_TIERS):
                tier = tg.get_tier_by_name(tier_name)
                intervals = tier.get_annotations_between_timepoints(
                    part['start_time'],
                    part['end_time']
                )
                for interval in intervals:
                    utterances.append(
                        {
                            'start_time': interval.start_time - part['start_time'],
                            'end_time': interval.end_time - part['start_time'],
                            'participant': participant,
                            'backchannel': is_backchannel(interval),
                        }
                    )
            df = pd.DataFrame(utterances)
            df.to_csv(out_filepath, index=False)


if __name__ == '__main__':
    main()
