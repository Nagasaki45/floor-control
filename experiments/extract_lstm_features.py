import pathlib

import numpy as np
import scipy.stats as sstats
import scipy.signal as ssignal

import utils.duel
import utils.path
import utils.yin

OUT_DIR = pathlib.Path('features') / 'LSTM'
BUFFER_DURATION = 0.05


def to_chunks(samples, chunk_size):
    samples = np.concatenate(
        (
            samples,
            np.zeros(shape=(chunk_size - len(samples) % chunk_size, samples.shape[-1])),
        )
    )
    return samples.reshape(-1, chunk_size, samples.shape[-1])


def calculate_voice_activity(textgrid, start_time, end_time, buffer_duration):
    a_utts = textgrid.get_tier_by_name('A-utts')
    b_utts = textgrid.get_tier_by_name('B-utts')
    return np.array([
        [bool(utts.get_annotations_by_time(t)) for utts in [a_utts, b_utts]]
        for t in np.arange(start_time, end_time, buffer_duration)
    ])


def calculate_freq(buffers, sample_rate):
    return np.apply_along_axis(
        utils.yin.compute_yin,
        axis=1,
        arr=buffers,
        sample_rate=sample_rate
    )


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def calculate_power(buffers):
    return 10 * np.log10(np.apply_along_axis(rms, axis=1, arr=buffers))


def calculate_spectral_flux(buffers):
    n_buffers, buffer_size, channels = buffers.shape
    flatten = buffers.reshape(n_buffers * buffer_size, -1)
    _, _, stft = ssignal.stft(
        flatten,
        nperseg=buffer_size,
        axis=0,
        window='hamming',
        noverlap=0,
    )
    abs_stft = np.abs(stft)
    diff = abs_stft[:, :, :-1] - abs_stft[:, :, 1:]
    rectified = (diff + np.abs(diff)) / 2
    flux = np.sum(rectified, axis=0) / np.sum(abs_stft[:, :, :-1], axis=0)
    return flux.T


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
    buffers = to_chunks(samples, buffer_size)

    y = calculate_voice_activity(textgrid, start_time, end_time, buffer_duration)

    freq = calculate_freq(buffers, sample_rate)
    pitch = 69 + 12 * np.log2(freq / 440)  # MIDI note
    pitch[pitch < 0] = 0
    voiced = (freq != 0)
    power = np.clip(calculate_power(buffers), -96, 0)  # 96dB is 16bit dynamic range
    spectral_flux = np.nan_to_num(calculate_spectral_flux(buffers))

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
