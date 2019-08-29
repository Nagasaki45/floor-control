import pathlib

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence
from keras import regularizers
import numpy as np

import data_loading
import settings

DATA_DIR = pathlib.Path('data')
_FRAME_RATE = 20
_SEQUENCE_DURATION = 10
_FUTURE_DURATION = 3
SEQUENCE_LENGTH = _FRAME_RATE * _SEQUENCE_DURATION
PREDICTION_LENGTH = _FRAME_RATE * _FUTURE_DURATION
BATCH_SIZE = 512


def session_part_generator(data):
    for session in data:
        for part in session['parts']:
            yield session, part


def prepare_model():
    model = Sequential()

    model.add(
        LSTM(
            units=10,
            activation='tanh',
            kernel_regularizer=regularizers.l2(0.001),
            input_shape=(
                SEQUENCE_LENGTH,
                2 * 6,  # Voice activation, relative pitch, absolute pitch, voiced, power, and spectral flux per interactant
            ),
        )
    )
    model.add(Dense(units=PREDICTION_LENGTH, activation='sigmoid'))

    model.compile(
        loss='mean_squared_error',
        optimizer=RMSprop(lr=0.01),
    )

    return model


class BatchGenerator(Sequence):
    def __init__(self, X, y, sequence_length, prediction_length, batch_size):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def __getitem__(self, idx):
        inputs = np.zeros((self.batch_size, self.sequence_length, self.X.shape[-1]))
        targets = np.zeros((self.batch_size, self.prediction_length))
        for idx_in_batch in range(self.batch_size):
            start = idx * self.batch_size + idx_in_batch
            end = start + self.sequence_length
            prediction_end = end + self.prediction_length
            inputs[idx_in_batch] = self.X[start:end]
            targets[idx_in_batch] = self.y[end:prediction_end]

        return inputs, targets

    def __len__(self):
        return int((len(self.X) - self.sequence_length - self.prediction_length) / self.batch_size)


def main():
    Xs, yss = [], []

    data = data_loading.generator(settings.ANNOTATIONS_DIR, settings.AUDIO_DIR)
    for i, (session, part) in enumerate(session_part_generator(data)):
        if (i % 4 == 0):  # Skip test-set
            continue
        Xs.append(np.load(DATA_DIR / f'X-{session["name"]}-{part["name"]}.npy'))
        yss.append(np.load(DATA_DIR / f'ys-{session["name"]}-{part["name"]}.npy'))

    X = np.vstack(Xs)
    ys = np.vstack(yss)

    for interactant in [0, 1]:
        model = prepare_model()
        batch_generator = BatchGenerator(
            X,
            ys[:, interactant],
            SEQUENCE_LENGTH,
            PREDICTION_LENGTH,
            BATCH_SIZE,
        )
        model.fit_generator(batch_generator, epochs=100)
        model.save(f'model_{interactant}.h5')


if __name__ == '__main__':
    main()
