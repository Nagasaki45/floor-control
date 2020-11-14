import audioop
import pathlib
import subprocess
import tempfile
import wave

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

                with wave.open(tf.name) as f:
                    sample_rate = f.getframerate()
                    sample_width = f.getsampwidth()
                    buffer_size = int(sample_rate * BUFFER_DURATION)
                    channels = f.getnchannels()

                    vad = webrtcvad.Vad(3)  # aggressive vad mode

                    pos = 0  # Counting samples, not bytes
                    start_pos = int(sample_rate * part['start_time'])
                    end_pos = int(sample_rate * part['end_time'])

                    # Seek to start_time
                    f.readframes(start_pos)
                    pos += start_pos

                    results = []
                    while pos < end_pos:
                        buffer = f.readframes(buffer_size)
                        if len(buffer) != buffer_size * sample_width * channels:
                            break
                        l = audioop.tomono(buffer, sample_width, 1, 0)
                        r = audioop.tomono(buffer, sample_width, 0, 1)
                        if session['swapped_stereo']:
                            l, r = r, l
                        vad_vals = [vad.is_speech(x, sample_rate) for x in [l, r]]
                        # Change floor holder when only one is vocalising
                        if sum(vad_vals) == 1:
                            current_floor_holder = vad_vals.index(True)
                        else:
                            current_floor_holder = results[-1]
                        results.append(current_floor_holder)
                        pos += buffer_size

                np.save(out_filepath, np.array(results).astype(float))


if __name__ == '__main__':
    main()
