import audioop

from . import core


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
