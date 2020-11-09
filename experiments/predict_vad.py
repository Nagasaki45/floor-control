import pathlib
import subprocess
import tempfile

import numpy as np
import webrtcvad

import utils.audio
import utils.duel
import utils.path

OUT_DIR = pathlib.Path('predictions') / 'VAD'
BUFFER_DURATION = 0.02


def upsample(source, target):
    subprocess.run([
        'ffmpeg',
        '-y',  # Overwrite, it will always exist because a temp is created
        '-i', source,
        '-ar', '48000',
        target,
    ])


class VadDetector:
    def __init__(
        self,
        sample_rate,
        vad_mode,
        **_,
    ):
        self._vad = webrtcvad.Vad(vad_mode)
        self._sample_rate = sample_rate
        self._current_floor_holder = None

    def process(self, buffer):
        vad_vals = [self._vad.is_speech(x, self._sample_rate) for x in buffer]
        # Change floor holder when only one is vocalising
        if sum(vad_vals) == 1:
            self._current_floor_holder = vad_vals.index(True)
        return self._current_floor_holder


def main():
    utils.path.empty_dir(OUT_DIR)

    data_gen = utils.duel.load_sessions_gen()

    for session in data_gen:

        with tempfile.NamedTemporaryFile(suffix='.wav') as tf:
            
            # Because webrtcvad doesn't work with 44.1
            upsample(str(session['audio_filepath'].resolve()), tf.name)

            for part in session['parts']:
                out_filepath = OUT_DIR / f'{session["name"]}-{part["name"]}.npy'
                print(f'Generating {out_filepath}')

                gen = utils.audio.wav_per_buffer_feature_extractor_gen(
                    tf.name,
                    part['start_time'],
                    part['end_time'],
                    BUFFER_DURATION,
                    session['swapped_stereo'],
                    extractor_class=VadDetector,
                    extractor_params={'vad_mode': 3},
                )
                np.save(out_filepath, np.array(list(gen)).astype(float))


if __name__ == '__main__':
    main()
