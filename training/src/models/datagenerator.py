import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline."""

    def __init__(self, input_list, mode, batch_size, img_size, shuffle=True):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            input_list: Image paths, labels and track data.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.img_paths = input_list[0]
        self.labels = input_list[1]

        self.img_size = img_size

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._preprocessing, num_parallel_calls=multiprocessing.cpu_count())
        elif mode == 'inference':
            data = data.map(self._preprocessing, num_parallel_calls=multiprocessing.cpu_count())
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(np.expand_dims(labels[i], axis=0))

        self.labels = np.concatenate(self.labels)

    def _preprocessing(self, filename, label):
        """Input parser for samples of the training set."""
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        image_resized = tf.image.resize_images(img_decoded,
                                               [self.img_size[0], self.img_size[1]])
        image_resized = tf.cast(image_resized, dtype=tf.float32)
        image_resized = tf.divide(image_resized, 255.0)

        return image_resized, label
