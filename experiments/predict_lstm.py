import pathlib

import keras.models
import numpy as np

import utils.lstm
import utils.path

IN_DIR = pathlib.Path('features') / 'LSTM'
MODELS_DIR = pathlib.Path('models') / 'LSTM'
OUT_DIR = pathlib.Path('predictions') / 'LSTM'


def main():
    utils.path.empty_dir(OUT_DIR)

    for type_ in ['full', 'partial']:

        models = [
            keras.models.load_model(MODELS_DIR / f'model_{i}_{type_}.h5')
            for i in range(2)
        ]

        for part in utils.path.session_parts_gen(train_set=False, test_set=True):

            out_filepath = OUT_DIR / f'{type_}-{part}.npy'
            print(f'Generating {out_filepath}')

            X = np.load(IN_DIR / f'X-{part}.npy')
            if type_ == 'partial':
                X = X[:, 2:]
            batch_generator = utils.lstm.BatchGenerator(X, np.zeros(len(X)))
            predictions = [m.predict(batch_generator) for m in models]
            # The model predict the next 3 seconds
            # only the first value is of interest
            predictions = np.vstack([p[:, 0] for p in predictions]).T
            floor_holder = predictions.argmax(axis=1)
            # Also, at index 0 is the prediction for 10 seconds (plus 1 frame)
            floor_holder = utils.lstm.shift_predictions(floor_holder)
            np.save(out_filepath, floor_holder)


if __name__ == '__main__':
    main()
