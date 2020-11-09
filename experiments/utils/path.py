import pathlib
import shutil


def empty_dir(dir_ : pathlib.Path):
    '''
    Delete dir if exists and create a new one.
    '''
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)