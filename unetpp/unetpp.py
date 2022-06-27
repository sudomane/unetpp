import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras

def dropout_block(input_tensor, filters, i, j):
    x = Conv2D(filters[i - j], 3, activation = 'relu', padding = 'same', name=f"x_{i}_{j}", kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
    for depth in range(i - j):
        x = Conv2D(filters[i - j], 3, activation = 'relu', padding = 'same', name=f"x_{i}_{j}_{depth}", kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    if j == 0 and i != 0:
        x = Dropout(0.5, name=f"dropout_{i}_{j}")(x)
    return x

def unetpp(input_shape, L=4, deep_supervised=False):
    l = L + 1
    mat = []
    for i in range(l):
        mat.append([0] * l)
    inputs = Input(shape=input_shape)
    filters = [64]
    mat[0][0] = dropout_block(inputs, filters, 0, 0)

    for i in range(1, l):
        pool = MaxPooling2D((2, 2), strides=(2, 2))(mat[i - 1][0])
        filters.append(filters[i - 1] * 2)
        mat[i][0] = dropout_block(pool, filters, i, 0)

        for j in range(1, i + 1):
            up = Conv2DTranspose(filters[i - j], (2, 2), strides=(2, 2), padding='same', name=f"up_{i}_{j}")(
                mat[i][j - 1])
            previous_convs = [up]
            for k in range(j - 1, -1, -1):
                previous_convs.append(mat[k + i - j][k])
            concat = concatenate(previous_convs)
            mat[i][j] = dropout_block(concat, filters, i, j)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if deep_supervised:
        decoders = [mat[k][k] for k in range(L, 0, -1)]
        output = []
        for d in range(L):
            output.append(
                Conv2D(1, 1, activation="sigmoid", name=f"output_{d}", padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(decoders[d]))
        model = Model(inputs, output)
        model.compile(optimizer=opt,
                      loss={'output_0': "binary_crossentropy",
                            'output_1': "binary_crossentropy",
                            'output_2': "binary_crossentropy",
                            'output_3': "binary_crossentropy"},
                      metrics={'output_0': 'accuracy',
                               'output_1': 'accuracy',
                               'output_2': 'accuracy',
                               'output_3': 'accuracy'},
                      loss_weights={'output_0': 1.,
                                    'output_1': 1.,
                                    'output_2': 1.,
                                    'output_3': 1.})
    else:
        output = Conv2D(1, 1, activation="sigmoid", name=f"output", padding="same", kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(mat[L][L])
        model = Model(inputs, output)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    return model