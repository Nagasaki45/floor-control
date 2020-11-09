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

    for i, (x_filepath, y_filepath) in enumerate(filenames):
        if (i % 4 == 0):  # Train set only
            continue
        Xs.append(np.load(x_filepath))
        ys.append(np.load(y_filepath))

    X = np.vstack(Xs)
    y = np.vstack(ys)

    for interactant in [0, 1]:
        model = utils.lstm.prepare_model()
        batch_generator = utils.lstm.BatchGenerator(X, y[:, interactant])
        model.fit(batch_generator, epochs=EPOCHS)
        model.save(OUT_DIR / f'model_{interactant}.h5')


if __name__ == '__main__':
    main()
