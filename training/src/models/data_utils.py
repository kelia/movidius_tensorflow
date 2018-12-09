import numpy as np
import math
import re
import os
import cv2
import time

from tensorflow.python.keras.preprocessing.image import Iterator


class BaseDirectoryIterator(Iterator):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:
    root_folder/
           folder_1/
                    images/
                    labels
           folder_2/
                    images/
                    labels
           .
           .
           folder_n/
                    images/
                    labels

    # Arguments
       directory: Path to the root directory to read data from.
       target_size: tuple of integers, dimensions to resize input images to.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    """

    def __init__(self, directory, radius_normalization,
                 target_size=(224, 224), num_channels=3, resume_train=False,
                 batch_size=32, shuffle=False, seed=None, follow_links=False):
        self.directory = directory
        self.target_size = tuple(target_size)
        self.follow_links = follow_links
        self.radius_normalization = radius_normalization

        self.num_channels = num_channels
        self.image_shape = self.target_size + (self.num_channels,)

        self.filename_poses = 'labels.csv'

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        self.formats = {'png'}

        # Associate each filename with a corresponding label
        self.filenames = []
        self.poses_labels = []
        self.n_samples_per_exp = []
        self.poses_vio = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)

        # Conversion of list into array
        self.poses_labels = np.array(self.poses_labels, dtype=np.float32)
        self.poses_vio = np.array(self.poses_vio, dtype=np.float32)

        print('Found {} images belonging to {} experiments.'.format(
            self.samples, self.num_experiments))
        super(BaseDirectoryIterator, self).__init__(self.samples,
                                                    batch_size, shuffle, seed)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                      key=lambda tpl: tpl[0])

    def _load_img(self, path):
        """
        Load an image. Ans reshapes it to target size

        # Arguments
            path: Path to image file.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_width, img_height)`.

        # Returns
            Image as numpy array.
        """
        img = cv2.imread(path)
        if self.target_size:
            if (img.shape[0], img.shape[1]) != self.target_size:
                img = cv2.resize(img, self.target_size,
                                 interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class ReactiveDirIterator(BaseDirectoryIterator):
    def next(self):
        """
        Public function to fetch next batch. Note that this function
        will only be used for evaluation and testing, but not for training.

        # Returns
            The next batch of images and labels.
        """
        with self.lock:
            index_array = next(
                self.index_generator)
            current_batch_size = index_array.shape[0]

        # Image transformation is not under thread lock, so it can be done in
        # parallel
        img_batch = np.zeros((current_batch_size,) + self.image_shape,
                             dtype=np.uint8)
        batch_y = np.zeros((current_batch_size, 3),
                           dtype=np.float32)

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = self._load_img(os.path.join(fname))
            img_batch[i] = x

        poses_labels_batch = self.poses_labels[index_array]

        return img_batch, poses_labels_batch

    def _decode_experiment_dir(self, dir_subpath):
        poses_labels_fname = os.path.join(dir_subpath, self.filename_poses)
        assert os.path.isfile(poses_labels_fname)
        exp_samples = 0

        # Try load labels
        poses_labels = np.loadtxt(poses_labels_fname, delimiter=';')
        poses_labels[:, 0] = poses_labels[:, 0] / self.radius_normalization


        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                                  key=lambda fname: int(re.search(r'\d+', fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(absolute_path)
                    self.poses_labels.append(poses_labels[frame_number])
                    self.samples += 1
                    exp_samples += 1
        # encode how many samples this experiment has
        self.n_samples_per_exp.append(exp_samples)