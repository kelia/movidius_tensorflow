import tensorflow as tf
import pprint
import random
import numpy as np
from models.learner import Learner
import os
import gflags
import sys

from models.common_flags import FLAGS


def _main():
    # Set random seed for training
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    print_flags_dict = {}
    for key in FLAGS.__flags.keys():
        print_flags_dict[key] = getattr(FLAGS, key)

    pp.pprint(print_flags_dict)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    assert os.path.exists(FLAGS.train_dir), "Specify training data directory!"
    assert os.path.exists(FLAGS.val_dir), "Specify validation data directory!"
    trl = Learner(FLAGS)
    trl.train(FLAGS)


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
