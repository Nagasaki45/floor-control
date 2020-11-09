import audioop
import pathlib

import numpy as np

import utils.audio
import utils.duel
import utils.path

OUT_DIR = pathlib.Path('features') / 'FCD'
BUFFER_DURATION = 0.02


class RmsExtractor():
    def __init__(self, sample_width, **_):
        self._sample_width = sample_width

    def process(self, fragments):
        return [audioop.rms(x, self._sample_width) for x in fragments]


def main():
    utils.path.empty_dir(OUT_DIR)

    data_gen = utils.duel.load_sessions_gen()

    for session in data_gen:
        for part in session['parts']:
            out_filepath = OUT_DIR / f'{session["name"]}-{part["name"]}.npy'
            print(f'Generating {out_filepath}')

            gen = utils.audio.wav_per_buffer_feature_extractor_gen(
                session['audio_filepath'],
                part['start_time'],
                part['end_time'],
                BUFFER_DURATION,
                session['swapped_stereo'],
                extractor_class=RmsExtractor,
            )
            rms = np.array(list(gen))

            np.save(out_filepath, rms)


if __name__ == '__main__':
    main()
