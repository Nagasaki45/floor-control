import itertools
import pathlib

import numpy as np

import utils.lstm
import utils.path

IN_DIR = pathlib.Path('features') / 'LSTM'
OUT_DIR = pathlib.Path('models') / 'LSTM'
EPOCHS = 100


def main():
    utils.path.empty_dir(OUT_DIR)

    Xs, ys = [], []

    filenames = zip(sorted(IN_DIR.glob('X*')), sorted(IN_DIR.glob('y*')))

    for part in utils.path.session_parts_gen(train_set=True, test_set=False):
        Xs.append(np.load(IN_DIR / f'X-{part}.npy'))
        ys.append(np.load(IN_DIR / f'y-{part}.npy'))

    X = np.vstack(Xs)
    y = np.vstack(ys)

    for interactant, full in itertools.product([0, 1], [False, True]):
        suffix = 'full' if full else 'partial'
        out_filepath = OUT_DIR / f'model_{interactant}_{suffix}.h5'
        print(f'Generating {out_filepath}')
        model = utils.lstm.prepare_model(full=full)
        batch_generator = utils.lstm.BatchGenerator(
            X if full else X[:, 2:],  # Droping the 1st feature: voice activity
            y[:, interactant]
        )
        model.fit(batch_generator, epochs=EPOCHS)
        model.save(out_filepath)


if __name__ == '__main__':
    main()
