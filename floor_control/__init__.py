import audioop

import webrtcvad

from . import core


class VadFilterDetector:
    def __init__(
        self,
        sample_rate,
        sample_width,
        buffer_size,
        cutoff_freq,
        hysteresis,
        num_of_interactants=2,
        vad_mode=3
    ):
        self._sample_rate = sample_rate
        self._vad = webrtcvad.Vad(mode=vad_mode)
        self._filters = [
            core.Filter(cutoff_freq=cutoff_freq, sample_rate=buffer_size / sample_rate)
            for _ in range(num_of_interactants)
        ]
        self._argmax = core.StableArgmax(hysteresis=hysteresis)

    def process(self, fragments):
        vad = [self._vad.is_speech(f, sample_rate=self._sample_rate) for f in fragments]
        smooth = [f.process(v) for f, v in zip(self._filters, vad)]
        return self._argmax.process(smooth)


class RmsFilterDetector:
    def __init__(
        self,
        sample_rate,
        sample_width,
        buffer_size,
        cutoff_freq,
        hysteresis,
        num_of_interactants=2,
    ):
        self._sample_width = sample_width
        self._filters = [
            core.Filter(cutoff_freq=cutoff_freq, sample_rate=sample_rate / buffer_size)
            for _ in range(num_of_interactants)
        ]
        self._argmax = core.StableArgmax(hysteresis=hysteresis)

    def process(self, fragments):
        rms = [audioop.rms(f, self._sample_width) for f in fragments]
        smooth = [f.process(s) for f, s in zip(self._filters, rms)]
        return self._argmax.process(smooth)
