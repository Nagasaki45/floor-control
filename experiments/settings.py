import pathlib

DUEL_DIR = pathlib.Path('~/DUEL').expanduser()
SUB_DUEL_DIR = DUEL_DIR / 'de'
ANNOTATIONS_DIR =  SUB_DUEL_DIR / 'transcriptions_annotations'
AUDIO_DIR = SUB_DUEL_DIR / 'audio'
BUFFER_DURATION = 0.02
BACKCHANNEL_WORDS = {'ja', 'okay', 'ohm', 'mhm', 'genau'}
MAX_BACKCHANNEL_DURATION = 0.5
ANNOTATIONS_TIERS = ['A-utts', 'B-utts']
COMPARABLE_TIERS = ['fcd', 'random', 'vad', 'lstm']
