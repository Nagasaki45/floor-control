import audioop

from . import core


class FloorControlDetector:
    def __init__(
        self,
        sample_rate,
        sample_width,
        buffer_duration=0.02,
        cutoff_freq=0.35,
        hysteresis=0.1,
        num_of_interactants=2,
    ):
        self._sample_width = sample_width
        self._filters = [
            core.Filter(cutoff_freq=cutoff_freq, sample_rate=1 / buffer_duration)
            for _ in range(num_of_interactants)
        ]
        self._argmax = core.StableArgmax(hysteresis=hysteresis)

    def process(self, fragments):
        rms = [audioop.rms(f, self._sample_width) for f in fragments]
        smooth = [f.process(s) for f, s in zip(self._filters, rms)]
        return self._argmax.process(smooth)
