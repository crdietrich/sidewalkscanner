"""Train a model for image classification
Colin Dietrich 2019
"""

import os
import pickle
import numpy as np
import sklearn.metrics # import confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

import config, models


class CallbackMetrics(Callback):

    def __init__(self):
        super().__init__()
        self.val_f1s = None
        self.val_recalls = None
        self.val_precisions = None

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = sklearn.metrics.f1_score(val_targ, val_predict)
        _val_recall = sklearn.metrics.recall_score(val_targ, val_predict)
        _val_precision = sklearn.metrics.precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: {} — val_precision: {} — val_recall {}".format(_val_f1, _val_precision, _val_recall))
        return


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
        rotation_range=45,  # degree for random rotations
        shear_range=45,    # shear angle in counter-clockwise direction in degrees
        zoom_range=0.2,     # range for random zoom
        fill_mode='nearest',   # points outside boundaries of inputs are filled
        horizontal_flip=True,  # randomly horizontally flip
        vertical_flip=True,    # randomly vertically flip
        )

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

    callback_file_path = os.path.normpath(training_output_dir + os.path.sep + model_description +
                                          "_weights_improvement_{epoch:02d}-{val_acc:.2f}.hdf5")

    callback_checkpoint = ModelCheckpoint(callback_file_path,
                                          monitor='acc', mode='auto',  # or monitor='val_acc' ?
                                          verbose=1, save_best_only=True)

    callback_metrics = CallbackMetrics()

    callback_tensorboard = TensorBoard(log_dir=training_output_dir,
                                       histogram_freq=1,
                                       write_graph=True, write_images=True,
                                       embeddings_freq=1,
                                       )
                                       #update_freq='epoch')

    callbacks_list = [callback_checkpoint] #, callback_tensorboard]  # callback_metrics

    model_history = []
    for i in range(config.k_folds):

        m_history = model.fit_generator(
            train_generator,
            steps_per_epoch=config.number_of_train_samples // config.batch_size,
            epochs=config.epochs,
            shuffle=True,
            validation_data=validation_generator,
            validation_steps=config.number_of_validation_samples // config.batch_size,
            workers=12,
            callbacks=callbacks_list)
        model_history.append(m_history)

    model.save(os.path.normpath(training_output_dir + model_description +
                                '_model_complete_' + file_time() + '.h5'))

    model.save_weights(os.path.normpath(training_output_dir + model_description +
                                        '_weights_complete_' + file_time() + '.h5'))

    complete_history_filepath = os.path.normpath(training_output_dir + model_description +
                                                 '_history_complete_' + file_time() + '.pkl')
    with open(complete_history_filepath, 'wb') as f:
        pickle.dump(model_history, f)

    return model_history


def build_run(model_description, verbose=False):
    """Build and run model fit"""

    model = models.inceptionV3_transfer(verbose=verbose)
    train_gen, validation_gen, test_gen = build_generators()
    history = train_model(model=model,
                          train_generator=train_gen,
                          validation_generator=validation_gen,
                          model_description=model_description)
    return history
