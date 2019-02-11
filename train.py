"""Train a model for image classification
Colin Dietrich 2019
"""

import os
import pickle
from datetime import datetime
from keras import Model, applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint

import config


def _struct_time():
    t = datetime.now()
    return (t.year, t.month, t.day, t.hour,
            t.minute, t.second, t.microsecond)


def std_time_ms(str_format='{:02d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:06}'):
    """Get time in stardard format '%Y-%m-%d %H:%M:%S.%f' and accurate
    to the millisecond
    """
    t = _struct_time()
    st = str_format
    return st.format(t[0], t[1], t[2], t[3], t[4], t[5], t[6])


def file_time():
    """Get time in a format compatible with filenames,
    '%Y_%m_%d_%H_%M_%S_%f' and accurate to the second
    """
    t = _struct_time()
    st = '{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{:06}'
    return st.format(t[0], t[1], t[2], t[3], t[4], t[5], t[6])


def make_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def find_max_batch(n_samples, max_size=33, verbose=False):
    """Find the maximum batch size that will divide the input data size without
    remainder.

    Parameters
    ----------
    n_samples : int, number of samples
    max_size : int, maximum number of samples per batch

    Returns
    -------
    int, maximum batch size
    """

    max_batch_size = 1
    for batch_size in range(1, max_size):
        if (n_samples % batch_size) == 0:
            max_batch_size = batch_size
            if verbose:
                print("Batch size found: {}".format(batch_size))
    return max_batch_size


def build_model():
    """Build a model based on VGG16 and ImageNet weights."""
    base_model = applications.VGG16(weights='imagenet',
                                    include_top=False,
                                    input_shape=config.input_shape)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    return model


def build_generators(tune_batch_size=False):
    """Build Data Generators for training, validation and testing

    TODO: add augmentation attributes to config file
    """

    if tune_batch_size:
        train_batch_size = find_max_batch(config.number_of_train_samples)
        validation_batch_size = find_max_batch(config.number_of_validation_samples)
        test_batch_size = find_max_batch(config.number_of_test_samples)
    else:
        train_batch_size = config.batch_size
        validation_batch_size = config.batch_size
        test_batch_size = config.batch_size

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        config.train_dir,
        target_size=(config.img_height, config.img_width),
        batch_size=train_batch_size,
        class_mode='binary',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        config.validation_dir,
        target_size=(config.img_height, config.img_width),
        batch_size=validation_batch_size,
        class_mode='binary',
        shuffle=False)

    test_generator = validation_datagen.flow_from_directory(
        config.test_dir,
        target_size=(config.img_height, config.img_width),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator, test_generator


def train_model(model, train_generator, validation_generator, model_description):
    """Name, train and save checkpoints, model, history and resulting weights."""

    training_output_dir = config.model_dir + os.path.sep + model_description + os.path.sep

    make_dir(training_output_dir)

    callback_file_path = os.path.normpath(training_output_dir + model_description +
                                          "_weights_improvement_{epoch:02d}-{val_acc:.2f}.hdf5")

    checkpoint = ModelCheckpoint(callback_file_path, monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model_history = model.fit_generator(
        train_generator,
        steps_per_epoch=config.number_of_train_samples // config.batch_size,
        epochs=config.epochs,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=config.number_of_validation_samples // config.batch_size,
        workers=12,
        callbacks=callbacks_list)

    model.save(os.path.normpath(training_output_dir + model_description +
                                '_model_complete_' + file_time() + '.h5'))

    model.save_weights(os.path.normpath(training_output_dir + model_description +
                                        '_weights_complete_' + file_time() + '.h5'))

    complete_history_filepath = os.path.normpath(training_output_dir + model_description +
                                                 '_history_complete_' + file_time() + '.pkl')
    with open(complete_history_filepath, 'wb') as f:
        pickle.dump(model_history, f)

    return model_history


def build_run(model_description):
    """Build and run model fit"""

    model = build_model()
    train_gen, validation_gen, test_gen = build_generators()
    history = train_model(model=model,
                          train_generator=train_gen,
                          validation_generator=validation_gen,
                          model_description=model_description)
    return history
