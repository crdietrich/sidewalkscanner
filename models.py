"""Define models for image classification
Colin Dietrich 2019
"""

from keras import Model, applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.layers import Flatten, Dense

import config


def model_A1(verbose=False):
    """Custom model, based on:
    https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d"""

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=config.input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def VGG16_transfer(frozen_layers=0, verbose=False):
    """Build a model based on VGG16 and ImageNet weights.

    Parameters
    ----------
    frozen_layers : int, number of output layers to freeze
    verbose : bool, print debug statements and model layer graph
    """
    base_model = applications.VGG16(weights='imagenet',
                                    include_top=False,
                                    input_shape=config.input_shape)

    if frozen_layers != 0:
        for layer in base_model.layers[:(-1*frozen_layers)+1]:
            layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=config.learning_rate, momentum=config.momentum),
                  metrics=['accuracy'])

    if verbose:
        print('Trainable Layer Summary')
        print('=======================')
        for layer in model.layers:
            conf = layer.get_config()
            print(conf['name'], '\t : ', layer.trainable)

    return model


def inceptionV3_transfer(frozen_layers=0, verbose=False):
    """Build a model based on VGG16 and ImageNet weights.

    Parameters
    ----------
    frozen_layers : int, number of output layers to freeze
    verbose : bool, print debug statements and model layer graph
    """
    base_model = applications.inception_v3.InceptionV3(weights='imagenet',
                                                       include_top=False,
                                                       input_shape=config.input_shape)

    if frozen_layers != 0:
        for layer in base_model.layers[:(-1*frozen_layers)+1]:
            layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=config.learning_rate, momentum=config.momentum),
                  metrics=['accuracy'])

    if verbose:
        print('Trainable Layer Summary')
        print('=======================')
        for layer in model.layers:
            conf = layer.get_config()
            print(conf['name'], '\t : ', layer.trainable)

    return model