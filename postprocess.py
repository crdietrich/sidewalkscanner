"""Post process methods for trained Keras Model
Colin Dietrich 2019
#TODO: most of this duplicates train, refactor into classes
"""


from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

import config


def binary_confusion_matrix(_test_generator, _predict_scores):
    # convert the binary classifier values to 1 if above 0.5, 0 if < 0.5
    # and compare to known label for class
    cnf_matrix = confusion_matrix(_test_generator.classes,
                                  (_predict_scores.T[0] > 0.5) * 1)
    return cnf_matrix


def build_predict_generator(predict_directory):
    predict_datagen = ImageDataGenerator(rescale=1. / 255)

    predict_generator = predict_datagen.flow_from_directory(
        predict_directory,
        target_size=(config.img_height, config.img_width),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

    return predict_generator
