import pathlib
import shutil

PARTS_DIR = pathlib.Path('features') / 'utterances'


def empty_dir(dir_ : pathlib.Path):
    '''
    Delete dir if exists and create a new one.
    '''
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)


def session_parts_gen(*, train_set, test_set):
    '''
    Yield part names according to train_set and
    test_set boolean flags.
    '''
    for i, filepath in enumerate(sorted(PARTS_DIR.iterdir())):
        part_name = filepath.name.replace('.csv', '')
        if train_set and (i % 4 != 0):  # Test-set only
            yield part_name
        elif test_set and (i % 4 == 0):
            yield part_name
