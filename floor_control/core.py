import audioop

import numpy as np
import scipy.signal as ss


class StableArgmax:
    def __init__(self, hysteresis):
        self._hysteresis = hysteresis
        self._previous = None

    def process(self, samples):
        argsort = np.argsort(samples)
        max_ = samples[argsort[-1]]
        next_ = samples[argsort[-2]]
        if max_ - next_ > self._hysteresis:
            self._previous = argsort[-1]
        return self._previous


class Filter:
    def __init__(self, cutoff_freq, sample_rate, order=2):
        self._b, self._a = ss.butter(N=order, Wn=cutoff_freq, fs=sample_rate)
        # Initial condition
        self._zi = ss.lfiltic(self._b, self._a, y=[])

    def process(self, sample):
        result, self._zi = ss.lfilter(self._b, self._a, [sample], zi=self._zi)
        return result[0]
