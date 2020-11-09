import pathlib

import floor_control.core
import numpy as np

import utils.path

IN_DIR = pathlib.Path('features') / 'FCD'
OUT_DIR = pathlib.Path('predictions') / 'FCD'
BUFFER_DURATION = 0.02


def gen_from_rms(rms, cutoff_freq, hysteresis):
    filters = [
        floor_control.core.Filter(
            cutoff_freq=cutoff_freq, sample_rate=1 / BUFFER_DURATION
        )
        for _ in range(2)
    ]
    argmax = floor_control.core.StableArgmax(hysteresis=hysteresis)

    for row in rms:
        smooth = [f.process(x) for f, x in zip(filters, row)]
        yield argmax.process(smooth)


def main():
    utils.path.empty_dir(OUT_DIR)

    for in_filepath in IN_DIR.iterdir():
        out_filepath = OUT_DIR / in_filepath.name
        print(f'Generating {out_filepath}')
        rms = np.load(in_filepath)
        gen = gen_from_rms(rms, cutoff_freq=0.35, hysteresis=0.1)
        np.save(out_filepath, np.array(list(gen)).astype(float))


if __name__ == '__main__':
    main()
