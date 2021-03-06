import pathlib

import numpy as np
import scipy.io.wavfile as swavfile
import tgt

DUEL_DIR = pathlib.Path('~/DUEL').expanduser()
SUB_DUEL_DIR = DUEL_DIR / 'de'
ANNOTATIONS_DIR =  SUB_DUEL_DIR / 'transcriptions_annotations'
AUDIO_DIR = SUB_DUEL_DIR / 'audio'

# For some of the session the audio files are swapped
SWAPPED_STEREO = {'r12', 'r13', 'r16'}


def load_sessions_gen(annotations_dir=ANNOTATIONS_DIR, audio_dir=AUDIO_DIR):
    session_dirs = annotations_dir.glob('r*')
    session_nums = (int(session_dir.name[1:]) for session_dir in session_dirs)
    for session_num in sorted(session_nums):
        session_dir = annotations_dir / f'r{session_num}'
        session = session_dir.name
        textgrid = tgt.io.read_textgrid(next(session_dir.glob('r*.TextGrid')))

        parts = []
        for part in textgrid.get_tier_by_name('Part').intervals:
            parts.append({
                'name': part.text,
                'start_time': part.start_time,
                'end_time': part.end_time,
            })
        yield {
            'name': session,
            'textgrid': textgrid,
            'audio_filepath': audio_dir / session / (session + '.wav'),
            'parts': parts,
            'swapped_stereo': session in SWAPPED_STEREO,
        }


def load_samples(session):
    sr, samples = swavfile.read(session['audio_filepath'])
    samples = (samples / np.iinfo(samples.dtype).max)
    if session['swapped_stereo']:
        samples = samples[:, [1, 0]]
    return samples, sr
