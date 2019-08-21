# Notes
# - No Truncated Back-propagation Through Time
# - Assume L2 regularization on the LSTM's kernal
# - Voice activity of participant A is not used to predict participant's B
# - Voiced (in pitch paragraph) is when pitch is not 0.

import pathlib

import numpy as np
import scipy.stats as sstats
import scipy.signal as ssignal

import data_loading
import yin

DUEL_DIR = pathlib.Path('~/DUEL').expanduser()
GERMAN_DIR = DUEL_DIR / 'de'
ANNOTATIONS_DIR =  GERMAN_DIR / 'transcriptions_annotations'
AUDIO_DIR = GERMAN_DIR / 'audio'
OUT_DIR = pathlib.Path('data')
BUFFER_DURATION = 0.05


# Data preparation utils

def to_frames(samples, buffer_size):
    samples = np.concatenate(
        (
            samples,
            np.zeros(shape=(buffer_size - len(samples) % buffer_size, samples.shape[-1])),
        )
    )
    return samples.reshape(-1, buffer_size, samples.shape[-1])


def calculate_voice_activity(textgrid, start_time, end_time, buffer_duration):
    a_utts = textgrid.get_tier_by_name('A-utts')
    b_utts = textgrid.get_tier_by_name('B-utts')
    return np.array([
        [bool(utts.get_annotations_by_time(t)) for utts in [a_utts, b_utts]]
        for t in np.arange(start_time, end_time, buffer_duration)
    ])


def calculate_pitch(frames, sample_rate):
    return np.apply_along_axis(
        yin.compute_yin,
        axis=1,
        arr=frames,
        sample_rate=sample_rate
    )


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def calculate_power(frames):
    return 10 * np.log10(np.apply_along_axis(rms, axis=1, arr=frames))


def calculate_spectral_flux(frames):
    n_frames, frame_size, channels = frames.shape
    flatten = frames.reshape(n_frames * frame_size, -1)
    _, _, stft = ssignal.stft(
        flatten,
        nperseg=frame_size,
        axis=0,
        window='hamming',
        noverlap=0,
    )
    abs_stft = np.abs(stft)
    diff = abs_stft[:, :, :-1] - abs_stft[:, :, 1:]
    rectified = (diff + np.abs(diff)) / 2
    flux = np.sum(rectified, axis=0) / np.sum(abs_stft[:, :, :-1], axis=0)
    return flux.T


def calculate_X_and_ys(
        start_time,
        end_time,
        textgrid,
        samples,
        sample_rate,
        buffer_duration,
    ):

    samples = samples[int(start_time * sample_rate):int(end_time * sample_rate)]
    buffer_size = int(sample_rate * buffer_duration)
    frames = to_frames(samples, buffer_size)

    ys = calculate_voice_activity(textgrid, start_time, end_time, buffer_duration)

    pitch = calculate_pitch(frames, sample_rate)
    voiced = (pitch != 0)
    power = calculate_power(frames)
    spectral_flux = calculate_spectral_flux(frames)

    X = np.hstack([
        ys,
        sstats.zscore(pitch),
        voiced,
        sstats.zscore(power),
        sstats.zscore(spectral_flux),
    ])

    return X, ys


def main():
    data_gen = data_loading.generator(ANNOTATIONS_DIR, AUDIO_DIR)

    for session in data_gen:
        session_name = session['name']
        data_loading.load_samples(session)

        for part in session['parts']:
            part_name = part['name']
            print(f'Processing {session_name}/{part_name}')

            X, ys = calculate_X_and_ys(
                part['start_time'],
                part['end_time'],
                session['textgrid'],
                session['samples'],
                session['sample_rate'],
                BUFFER_DURATION,
            )

            np.save(OUT_DIR / f'X-{session_name}-{part_name}.npy', X)
            np.save(OUT_DIR / f'ys-{session_name}-{part_name}.npy', ys)


if __name__ == '__main__':
    main()
