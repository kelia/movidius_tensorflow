import os
import re

import numpy as np
from tensorflow.python.keras.preprocessing.image import Iterator


class DirectoryIterator(Iterator):
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
       follow_links: Bool, whether to follow symbolic links or not
    """

    def __init__(self, directory, radius_normalization, follow_links=False):
        self.directory = directory
        self.follow_links = follow_links
        self.radius_normalization = radius_normalization
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
        self.labels = []
        self.n_samples_per_exp = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)

        # Conversion of list into array
        self.labels = np.array(self.labels, dtype=np.float32)

        print('Found {} images belonging to {} experiments.'.format(
            self.samples, self.num_experiments))

    def _decode_experiment_dir(self, dir_subpath):
        labels_fname = os.path.join(dir_subpath, self.filename_poses)
        assert os.path.isfile(labels_fname)
        exp_samples = 0

        # Try load labels
        labels = np.loadtxt(labels_fname, delimiter=';')
        labels[:, 0] = labels[:, 0] / self.radius_normalization

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
                    self.labels.append(labels[frame_number])
                    self.samples += 1
                    exp_samples += 1
        # encode how many samples this experiment has
        self.n_samples_per_exp.append(exp_samples)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                      key=lambda tpl: tpl[0])
