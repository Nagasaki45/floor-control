import audioop
import wave


def wav_per_buffer_feature_extractor_gen(
    filepath,
    start_time,
    end_time,
    buffer_duration,
    swap_stereo,
    extractor_class,
    extractor_params=None,
):
    '''
    Stream a wav file through a feature extractor.
    '''
    if extractor_params is None:
        extractor_params = {}

    with wave.open(str(filepath)) as f:
        sample_rate = f.getframerate()
        sample_width = f.getsampwidth()
        buffer_size = int(sample_rate * buffer_duration)
        channels = f.getnchannels()

        extractor_params['sample_rate'] = sample_rate
        extractor_params['sample_width'] = sample_width

        extractor = extractor_class(**extractor_params)

        pos = 0  # Counting samples, not bytes
        start_pos = int(sample_rate * start_time)
        end_pos = int(sample_rate * end_time)

        # Seek to start_time
        f.readframes(start_pos)
        pos += start_pos

        while pos < end_pos:
            buffer = f.readframes(buffer_size)
            if len(buffer) != buffer_size * sample_width * channels:
                break
            l = audioop.tomono(buffer, sample_width, 1, 0)
            r = audioop.tomono(buffer, sample_width, 0, 1)
            if swap_stereo:
                l, r = r, l
            yield extractor.process([l, r])
            pos += buffer_size