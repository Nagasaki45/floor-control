import pathlib

import numpy as np
import scipy.stats as sstats

import utils.audio
import utils.duel
import utils.path

OUT_DIR = pathlib.Path('features') / 'LSTM'
BUFFER_DURATION = 0.05


def calculate_voice_activity(textgrid, start_time, end_time, buffer_duration):
    a_utts = textgrid.get_tier_by_name('A-utts')
    b_utts = textgrid.get_tier_by_name('B-utts')
    return np.array([
        [bool(utts.get_annotations_by_time(t)) for utts in [a_utts, b_utts]]
        for t in np.arange(start_time, end_time, buffer_duration)
    ])


def calculate_X_and_y(
        start_time,
        end_time,
        textgrid,
        samples,
        sample_rate,
        buffer_duration,
    ):

    samples = samples[int(start_time * sample_rate):int(end_time * sample_rate)]
    buffer_size = int(sample_rate * buffer_duration)
    buffers = utils.audio.to_chunks(samples, buffer_size)

    y = calculate_voice_activity(textgrid, start_time, end_time, buffer_duration)

    freq = utils.audio.calculate_freq(buffers, sample_rate)
    pitch = 69 + 12 * np.log2(freq / 440)  # MIDI note
    pitch[pitch < 0] = 0
    voiced = (freq != 0)
    power = np.clip(utils.audio.calculate_power(buffers), -96, 0)  # 96dB is 16bit dynamic range
    spectral_flux = np.nan_to_num(utils.audio.calculate_spectral_flux(buffers))

    X = np.hstack([
        y,
        np.apply_along_axis(sstats.zscore, axis=0, arr=pitch),
        pitch,
        voiced,
        np.apply_along_axis(sstats.zscore, axis=0, arr=power),
        np.apply_along_axis(sstats.zscore, axis=0, arr=spectral_flux),
    ])

    return X, y


def main():
    utils.path.empty_dir(OUT_DIR)

    data_gen = utils.duel.load_sessions_gen()

    for session in data_gen:
        samples, sample_rate = utils.duel.load_samples(session)

        for part in session['parts']:
            x_filepath = OUT_DIR / f'X-{session["name"]}-{part["name"]}.npy'
            y_filepath = OUT_DIR / f'y-{session["name"]}-{part["name"]}.npy'
            print(f'Generating {x_filepath} & {y_filepath}')

            X, y = calculate_X_and_y(
                part['start_time'],
                part['end_time'],
                session['textgrid'],
                samples,
                sample_rate,
                BUFFER_DURATION,
            )

            np.save(x_filepath, X)
            np.save(y_filepath, y)


if __name__ == '__main__':
    main()
