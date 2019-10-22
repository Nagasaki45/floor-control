from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from floor_control import FloorControlDetector


@given(
    # 20ms of 16bit audio in 48Khz sample rate
    a_frame=st.binary(min_size=int(0.02 * 48000 * 2), max_size=int(0.02 * 48000 * 2)),
    b_frame=st.binary(min_size=int(0.02 * 48000 * 2), max_size=int(0.02 * 48000 * 2)),
)
def test_process_once(a_frame, b_frame):
    floor_detector = FloorControlDetector(sample_rate=48000, sample_width=2)
    floor_holder = floor_detector.process([a_frame, b_frame])
    assert floor_holder in [0, 1, None]


@given(
    # 20ms of 16bit audio in 48Khz sample rate
    a_frame=st.binary(min_size=int(0.02 * 48000 * 2), max_size=int(0.02 * 48000 * 2)),
    b_frame=st.binary(min_size=int(0.02 * 48000 * 2), max_size=int(0.02 * 48000 * 2)),
)
def test_process_1_sec(a_frame, b_frame):
    floor_detector = FloorControlDetector(sample_rate=48000, sample_width=2)
    floor_detected = False
    for i in range(50):
        floor_holder = floor_detector.process([a_frame, b_frame])
        if floor_holder is not None:
            floor_detected = True
        if floor_detected:
            assert floor_holder in [0, 1]
