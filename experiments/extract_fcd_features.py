import pathlib

import numpy as np

import utils.audio
import utils.duel
import utils.path

OUT_DIR = pathlib.Path('features') / 'FCD'
BUFFER_DURATION = 0.02


def main():
    utils.path.empty_dir(OUT_DIR)

    data_gen = utils.duel.load_sessions_gen()

    for session in data_gen:
        samples, sample_rate = utils.duel.load_samples(session)

        for part in session['parts']:
            start_pos = int(part['start_time'] * sample_rate)
            end_pos = int(part['end_time'] * sample_rate)
            buffer_size = int(sample_rate * BUFFER_DURATION)

            part_samples = samples[start_pos:end_pos]

            out_filepath = OUT_DIR / f'{session["name"]}-{part["name"]}.npy'
            print(f'Generating {out_filepath}')

            buffers = utils.audio.to_chunks(part_samples, buffer_size)
            rms = np.apply_along_axis(utils.audio.rms, axis=1, arr=buffers)

            np.save(out_filepath, rms)


if __name__ == '__main__':
    main()
