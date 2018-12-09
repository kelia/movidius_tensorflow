# Base model for learning
import os
import numpy as np
import tensorflow as tf


class BaseLearner(object):
    def __init__(self, config):
        self.min_val_loss = np.inf
        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False)

        self.incr_global_step = tf.assign(self.global_step,
                                          self.global_step + 1)
        self.num_channels = 3
        self.action_dim = 4

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        if step == 'best':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.best'))
            print(" [*] Saving checkpoint to {}/{}.best".format(checkpoint_dir,
                                                                model_name))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
            print(" [*] Saving checkpoint to {}/{}-{}".format(checkpoint_dir,
                                                              model_name, step))

    def preprocess_image(self, image):
        """ Preprocess an input image
        Args:
            Image: A uint8 tensor
        Returns:
            image: A preprocessed float32 tensor.
        """
        # image = tf.image.resize_images(image,
        #                                [self.config.img_height, self.config.img_width])
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0)

        # convert to NCHW
        # image = tf.transpose(image, [0, 3, 1, 2])

        return image

    def setup_inference(self, config, mode):
        """Sets up the inference graph.
        """
        self.mode = mode
        self.config = config
        self.build_test_graph()