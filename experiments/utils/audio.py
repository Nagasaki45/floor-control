import numpy as np
import scipy.signal as ssignal

from . import yin


def to_chunks(samples, chunk_size):
    samples = np.concatenate(
        (
            samples,
            np.zeros(shape=(chunk_size - len(samples) % chunk_size, samples.shape[-1])),
        )
    )
    return samples.reshape(-1, chunk_size, samples.shape[-1])


def calculate_freq(buffers, sample_rate):
    return np.apply_along_axis(
        yin.compute_yin,
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
