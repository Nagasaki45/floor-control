from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence
from keras import regularizers
import numpy as np

_FRAME_RATE = 20
_SEQUENCE_DURATION = 10
SEQUENCE_LENGTH = int(_FRAME_RATE * _SEQUENCE_DURATION)
BATCH_SIZE = 256


def prepare_model():
    model = Sequential()

    model.add(
        LSTM(
            units=10,
            activation='tanh',
            kernel_regularizer=regularizers.l2(0.001),
            input_shape=(
                SEQUENCE_LENGTH,
                2 * 4,  # Pitch, voiced, power, and spectral flux per interactant
            ),
        )
    )
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        loss='mean_squared_error',
        optimizer=RMSprop(lr=0.01),
    )

    return model


class BatchGenerator(Sequence):
    def __init__(self, X, y, sequence_length, batch_size):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __getitem__(self, idx):
        inputs = np.zeros((self.batch_size, self.sequence_length, self.X.shape[-1]))
        targets = np.zeros(self.batch_size)
        for idx_in_batch in range(self.batch_size):
            start = idx * self.batch_size + idx_in_batch
            end = start + self.sequence_length
            inputs[idx_in_batch] = self.X[start:end]
            targets[idx_in_batch] = self.y[end]

        return inputs, targets

    def __len__(self):
        return int((len(self.X) - self.sequence_length) / self.batch_size)


def main():
    X, ys = np.load('X.npy'), np.load('ys.npy')
    X = np.nan_to_num(X)

    for interactant in [0, 1]:
        model = prepare_model()
        batch_generator = BatchGenerator(X, ys[:, interactant], SEQUENCE_LENGTH, BATCH_SIZE)
        model.fit_generator(batch_generator, epochs=100)
        model.save(f'model_{interactant}.h5')


if __name__ == '__main__':
    main()
