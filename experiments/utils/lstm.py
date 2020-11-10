from keras import regularizers
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import Sequence

import numpy as np

FRAME_RATE = 20
_SEQUENCE_DURATION = 10
_FUTURE_DURATION = 3
SEQUENCE_LENGTH = FRAME_RATE * _SEQUENCE_DURATION
PREDICTION_LENGTH = FRAME_RATE * _FUTURE_DURATION
BATCH_SIZE = 512


def prepare_model(full=True):
    '''
    Return the replication of Skantze 2017 LSTM turn-taking model,
    implemented in Keras.
    '''
    model = Sequential()

    # Voice activation (if full), relative pitch, absolute pitch, voiced, power, and spectral flux
    n_features = 6 if full else 5

    model.add(
        LSTM(
            units=10,
            activation='tanh',
            kernel_regularizer=regularizers.l2(0.001),
            input_shape=(
                SEQUENCE_LENGTH,
                2 * n_features,  # Per interactant
            ),
        )
    )
    model.add(Dense(units=PREDICTION_LENGTH, activation='sigmoid'))

    model.compile(
        loss='mean_squared_error',
        optimizer=RMSprop(lr=0.01),
    )

    return model


def shift_predictions(x):
    '''
    The vector of predictions should be shifted by
    SEQUENCE_LENGTH + 1, because at index 0 the prediction
    is for the next frame after SEQUENCE_LENGTH.
    '''
    return np.hstack([[np.NaN] * (SEQUENCE_LENGTH + 1), x])


class BatchGenerator(Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        inputs = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, self.X.shape[-1]))
        targets = np.zeros((BATCH_SIZE, PREDICTION_LENGTH))
        for idx_in_batch in range(BATCH_SIZE):
            start = idx * BATCH_SIZE + idx_in_batch
            end = start + SEQUENCE_LENGTH
            prediction_end = end + PREDICTION_LENGTH
            inputs[idx_in_batch] = self.X[start:end]
            targets[idx_in_batch] = self.y[end:prediction_end]

        return inputs, targets

    def __len__(self):
        return int((len(self.X) - SEQUENCE_LENGTH - PREDICTION_LENGTH) / BATCH_SIZE)