"""Preprocess image files from a source collection for Keras CNN training
Colin Dietrich 2019
"""

import os
import shutil

import config


def make_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def build_folder_structure():
    # folders for balanced images for training, validation and test
    if not os.path.isdir(config.input_dir):
        os.mkdir(config.input_dir)
    for d in [config.train_dir, config.validation_dir, config.test_dir]:
        make_dir(d)
        make_dir(d + os.path.sep + 'CP')
        make_dir(d + os.path.sep + 'UP')
    # model checkpoints, output and weights
    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)


def split_images(n0, n1, src, dst, verbose=False):
    """Split training data into two directories for
    Keras flowfromdirectory method

    Parameters
    ----------
    n0 : int, index of first image in dir
    n1 : int, index of last image in dir
    src : source folder
    dst : destination folder
    verbose : bool, print debug statements
    """

    files = os.listdir(src)
    if verbose:
        print(len(files), type(files))
    for file in files[n0:n1]:
        shutil.copyfile(src + file, dst + file)


def apply_splitter(source_dir=config.source_dir,
                   verbose=False):
    """Split data according to splitter definition

    Parameters
    ----------
    source_dir :
    verbose : bool, print debug statements
    """

    for n in config.splitter:
        if verbose:
            print(n[0])
            print(n[1])
            print(n[2], n[3])
        for c in config.classes:
            split_images(n[2], n[3],
                         source_dir + os.path.sep + c + os.path.sep,
                         n[1] + os.path.sep + c + os.path.sep,
                         verbose=verbose)


if __name__ == "__main__":
    build_folder_structure()
    apply_splitter()
    print('Done creating folders and splitting image files!')
