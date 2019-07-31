# Notes
# - No Truncated Back-propagation Through Time
# - Assume L2 regularization on the LSTM's kernal
# - Voice activity of participant A is not used to predict participant's B
# - Voiced (in pitch paragraph) is when pitch is not 0.

import pathlib

import numpy as np
import scipy.io.wavfile as swavfile
import scipy.stats as sstats
import scipy.signal as ssignal
import tgt

import yin

DUEL_DIR = pathlib.Path('~/DUEL').expanduser()
GERMAN_DIR = DUEL_DIR / 'de'
ANNOTATIONS_DIR =  GERMAN_DIR / 'transcriptions_annotations'
AUDIO_DIR = GERMAN_DIR / 'audio'
BUFFER_DURATION = 0.05
SEQUENCE_DURATION = 60
SWAPPED_STEREO = {'r12', 'r13', 'r16'}


# Data loading utils

def load_data_generator(annotations_dir):
    for session_dir in annotations_dir.glob('r*'):
        session = session_dir.name
        yield (
            session,
            {
                'textgrid': tgt.io.read_textgrid(next(session_dir.glob('r*.TextGrid'))),
                'audio_filepath': AUDIO_DIR / session / (session + '.wav'),
            },
        )


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


def calculate_X_and_ys(textgrid, audio_filepath, buffer_duration, swapped_stereo):
    sample_rate, samples = swavfile.read(audio_filepath)
    samples = (samples / np.iinfo(samples.dtype).max)
    if swapped_stereo:
        samples = samples[:, [1, 0]]

    for part in textgrid.get_tier_by_name('Part').intervals:
        start_time = part.start_time
        end_time = part.end_time

        buffer_size = int(sample_rate * buffer_duration)
        frames = to_frames(
            samples[int(start_time * sample_rate):int(end_time * sample_rate)],
            buffer_size,
        )

        ys = calculate_voice_activity(textgrid, start_time, end_time, buffer_duration)

        pitch = calculate_pitch(frames, sample_rate)
        voiced = (pitch != 0)
        power = calculate_power(frames)
        spectral_flux = calculate_spectral_flux(frames)

        X = np.hstack([
            sstats.zscore(pitch),
            voiced,
            sstats.zscore(power),
            sstats.zscore(spectral_flux),
        ])

        yield X, ys


def main():
    data = dict(load_data_generator(ANNOTATIONS_DIR))

    Xs, yss = [], []

    for session_name, session_data in sorted(data.items()):
        print('Processing', session_name)
        gen = calculate_X_and_ys(
            session_data['textgrid'],
            session_data['audio_filepath'],
            BUFFER_DURATION,
            session_name in SWAPPED_STEREO,
        )
        for X, ys in gen:
            Xs.append(X)
            yss.append(ys)

    np.save('X.npy', np.vstack(Xs))
    np.save('ys.npy', np.vstack(yss))


if __name__ == '__main__':
    main()
