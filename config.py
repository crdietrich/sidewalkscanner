"""Configuration settings for camera functions
Colin Dietrich 2019
"""

import os
from sys import platform

# ===== File Saving Locations =====
# assume Linux OS, i.e. if platform == "linux" or platform == "linux2":
source_dir = os.path.normpath("/home/colin/data/SDNET2018/P/")
input_dir = os.path.normpath("/home/colin/data/SDNET2018_balanced_2/")
model_dir = os.path.normpath("/home/colin/data/model_outputs/")

if platform == "darwin":  # OS X
    pass
if platform == "win32":
    source_dir = os.path.normpath("C://Users//colin//data//SDNET2018//P//")
    input_dir = os.path.normpath("C://Users//colin//data//SDNET2018_balanced_2//")
    model_dir = os.path.normpath("C://Users//colin//Google_Drive//ilocal//data//model_outputs//")

train_dir = input_dir + os.path.sep + "train" + os.path.sep
validation_dir = input_dir + os.path.sep + "validation" + os.path.sep
test_dir = input_dir + os.path.sep + "test" + os.path.sep

splitter = [['train', train_dir, 0, 1700],
            ['validation', validation_dir, 1700, 2300],
            ['test', test_dir, 2300, 2500]]

# ===== Class Identification =====
classes = ['CP', 'UP']

# ===== Input Image Size =====
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

# ===== Training Data Attributes =====
number_of_train_samples      = 3400
number_of_validation_samples = 1200
number_of_test_samples       = 400

# ===== Training Attributes =====
learning_rate = 1e-4
momentum = 0.2
k_folds = 1
epochs = 100
batch_size = 32
frozen_layers = 5  # number of layers to freeze, for none set -1
last_layers_frozen = 0
