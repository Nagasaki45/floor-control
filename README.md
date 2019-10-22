# floor-control

## A Simple, Real-Time Model of Turn-Taking in Dialogue

This repo is part of a research project that aims to detect the floor holder in dialogues based on audio data.
Here you can find the model and the code for evaluation it in the `experiments` folder.
The paper, describing the model in length, is not out yet.
More info will be added in time.

## Installation

```bash
$ pip install git+https://github.com/nagasaki45/floor-control.git
```

## Usage example

```python
import audioop
import wave

from floor_control import FloorControlDetector

BUFFER_DURATION = 0.02

with wave.open('/home/nagasaki45/DUEL/de/audio/r1/r1.wav') as wf:
    
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    buffer_size = int(BUFFER_DURATION * sample_rate)

    detector = FloorControlDetector(
        sample_rate=sample_rate,
        sample_width=sample_width,
    )

    while True:
        buffer = wf.readframes(buffer_size)
        if len(buffer) != buffer_size * sample_width * wf.getnchannels():
            break
        left = audioop.tomono(buffer, 2, 1, 0)
        right = audioop.tomono(buffer, 2, 0, 1)
        floor_holder = detector.process([left, right])
        # Do something with the floor holder
```

*Note* that we investigated the model only with a buffer duration of 20ms.
This is the default value for the `FloorControlDetector` although other values can be set.

## Running the tests

```bash
$ pip install -r requirements.txt
$ pytest
```
