import os
import time
from itertools import count
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils import Progbar
from .nets import cnn, mean_predictor, variance_predictor, variance_predictor_no_exp
from .data_utils import DirectoryIterator
from .datagenerator import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Learner(object):
    def __init__(self):
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
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0)

        return image

    def setup_inference(self, config, mode):
        """Sets up the inference graph.
        """
        self.mode = mode
        self.config = config
        self.build_test_graph()

    def get_filenames_list(self, directory):
        assert os.path.isdir(directory)
        iterator = DirectoryIterator(directory, self.config.radius_normalization)
        return iterator.filenames, iterator.labels

    def build_train_graph(self):
        is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

        image_height = self.config.img_height
        image_width = self.config.img_width

        image_list_tr, label_list_tr = self.get_filenames_list(self.config.train_dir)
        image_list_val, label_list_val = self.get_filenames_list(self.config.val_dir)

        with tf.device('/cpu:0'):
            tr_data = ImageDataGenerator([image_list_tr, label_list_tr],
                                         mode='training',
                                         batch_size=self.config.batch_size,
                                         img_size=(image_height, image_width),
                                         shuffle=True)

            val_data = ImageDataGenerator([image_list_val, label_list_val],
                                          mode='inference',
                                          batch_size=self.config.batch_size,
                                          img_size=(image_height, image_width),
                                          shuffle=True)

            # create a reinitializable iterator given the dataset structure
            iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
            current_batch = iterator.get_next()

        training_init_op = iterator.make_initializer(tr_data.data)
        validation_init_op = iterator.make_initializer(val_data.data)

        with tf.name_scope("data_loading"):
            image_batch, pose_batch = current_batch[0], current_batch[1]

        with tf.name_scope("pose_prediction"):
            image_descriptors = cnn(image_batch, scope='CNN')

            mean_prediction = mean_predictor(image_descriptors=image_descriptors, output_dim=self.action_dim,
                                             is_training=is_training_ph, scope='Mean_Prediction')
            variance_prediction = variance_predictor(image_descriptors=image_descriptors, output_dim=self.action_dim,
                                                     is_training=is_training_ph, scope='Variance_Prediction')

            prediction = tf.concat([mean_prediction, variance_prediction], axis=1)
            prediction = tf.identity(prediction, name="final_prediction")

        with tf.name_scope("compute_loss"):
            sigma = prediction[:, self.action_dim:]
            tf.assert_greater(sigma, tf.constant(0.0, shape=[self.config.batch_size, self.action_dim]))
            mu = prediction[:, :self.action_dim]

            log_loss = tf.reduce_sum(tf.log(sigma) + tf.square(pose_batch - mu) / sigma)
            mean_loss = tf.losses.mean_squared_error(labels=pose_batch,
                                                     predictions=mu)

            # Adapt loss to training strategy
            train_loss = mean_loss  # + log_loss

            # compute norm of variance for debugging
            variance_norm = tf.reduce_sum(sigma[:, 0])

        with tf.name_scope("train_op"):
            train_vars = {
                'CNN': [var for var in tf.trainable_variables(scope='CNN')],
                'Mean_Prediction': [var for var in tf.trainable_variables(scope='Mean_Prediction')],
                'Variance_Prediction': [var for var in tf.trainable_variables(scope='Variance_Prediction')]
            }

            optimizer = tf.train.AdamOptimizer(self.config.learning_rate,
                                               self.config.beta1)

            grads_and_vars = {}
            for key, vars_list in train_vars.items():
                grads_and_vars[key] = optimizer.compute_gradients(train_loss,
                                                                  var_list=train_vars[key])
            self.grads_and_vars = grads_and_vars
            train_op = {}
            for key, grad_vars in self.grads_and_vars.items():
                train_op[key] = optimizer.apply_gradients(grad_vars)

            self.train_op = train_op

        # Collect tensors that are useful later (e.g. tf summary), maybe add
        # images
        self.train_steps_per_epoch = \
            int(np.ceil(len(image_list_tr) / self.config.batch_size))
        self.val_steps_per_epoch = \
            int(np.ceil(len(image_list_val) / self.config.batch_size))
        self.gt_pnt = pose_batch
        self.train_loss = train_loss
        self.mean_loss = mean_loss
        self.log_loss = log_loss
        self.variance_norm = variance_norm
        self.image_batch = image_batch
        self.is_training = is_training_ph
        self.training_init_iter = training_init_op
        self.validation_init_iter = validation_init_op
        self.tr_data = tr_data
        self.val_data = val_data

    def collect_summaries(self):
        train_loss_sum = tf.summary.scalar("train_loss", self.train_loss)
        mean_loss_sum = tf.summary.scalar("mean_loss", self.mean_loss)
        log_loss_sum = tf.summary.scalar("log_loss", self.log_loss)
        variance_norm = tf.summary.scalar("variance_norm", self.variance_norm)
        self.step_sum_op = tf.summary.merge(
            [train_loss_sum, mean_loss_sum, log_loss_sum, variance_norm])
        self.validation_loss = tf.placeholder(tf.float32, shape=(), name="validation_loss")
        self.val_loss_sum = tf.summary.scalar("Validation_Error", self.validation_loss)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
            descriptors_database_dict: dictionary of numpy arrays of descriptors, one array per track
            actions_database_dict: dictionary of numpy arrays of actions associated to descriptors, one array per track
            pos_database_dict: dictionary of numpy arrays of positions associated to descriptors, one array per track
        Returns:
            None
        """

        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in \
                                     tf.trainable_variables()] + [self.global_step], max_to_keep=20)

        sv = tf.train.Supervisor(logdir=config.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        with sv.managed_session() as sess:
            print("Number of params: {}".format(sess.run(parameter_count)))
            if config.resume_train:
                assert os.path.isdir(self.config.checkpoint_dir)
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(
                    self.config.checkpoint_dir)
                assert checkpoint, "Found no checkpoint in the given dir!"
                print("Restoring checkpoint: ")
                print(checkpoint)
                self.saver.restore(sess, checkpoint)

            progbar = Progbar(target=self.train_steps_per_epoch)

            # What to train?
            trainables = {'CNN': True,
                          'Mean_Prediction': True,
                          'Variance_Prediction': False}

            n_epochs = 0

            # (Re-)Initialize the iterator
            sess.run(self.training_init_iter)

            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                fetches = {
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                for key, trainable in trainables.items():
                    # print(key + " is trainable: " + str(trainable))
                    if trainable:
                        # fetches[key] = self.train_op[key]
                        fetches[key] = self.train_op[key]

                if step % config.summary_freq == 0:
                    fetches["train_loss"] = self.train_loss
                    fetches["summary"] = self.step_sum_op

                # Runs a series of operations
                results = sess.run(fetches, feed_dict={self.is_training: True})

                progbar.update(step % self.train_steps_per_epoch)

                gs = results["global_step"]

                if step % config.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    self.completed_epochs = int(gs / self.train_steps_per_epoch)
                    train_step = gs - (self.completed_epochs - 1) * self.train_steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it train_loss: %.3f "
                          % (self.completed_epochs, train_step, self.train_steps_per_epoch, \
                             time.time() - start_time, results["train_loss"]))

                if step % self.train_steps_per_epoch == 0:
                    n_epochs += 1
                    self.completed_epochs = int(gs / self.train_steps_per_epoch)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    done = self._epoch_end_callback(sess, sv, n_epochs)

                    # (Re-)Initialize the iterator
                    sess.run(self.training_init_iter)

                    if done:
                        break

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or for prediction of trajectories.
        """

        image_height = self.config.img_height_sim if self.config.sim_experiment else self.config.img_height_rw
        image_width = self.config.img_width_sim if self.config.sim_experiment else self.config.img_width_rw

        # input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
        #                                         image_width, self.num_channels],
        #                              name='raw_input')
        self.data_format = "NHWC"
        input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
                                                image_width, self.num_channels],
                                     name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        # flattened_input = tf.layers.flatten(input_mc)
        pnt_batch = tf.placeholder(tf.float32, [None, self.action_dim],
                                   name='gt_labels')

        pos_batch = tf.placeholder(tf.float32, [None, 3], name='pos_gt')

        image_descriptors = cnn(input_mc,
                                is_training=False,
                                scope='CNN', data_format=self.data_format)

        mean_prediction = mean_predictor(image_descriptors=image_descriptors, output_dim=self.action_dim,
                                         is_training=False, scope='Mean_Prediction')
        variance_prediction = variance_predictor_no_exp(image_descriptors=image_descriptors, output_dim=self.action_dim,
                                                        is_training=False, scope='Variance_Prediction')

        prediction = tf.concat([mean_prediction, variance_prediction], axis=1)
        # prediction = mean_prediction

        self.inputs_img = input_uint8
        self.action = prediction
        self.gt_action = pnt_batch
        self.pos_gt = pos_batch

    def inference(self, inputs, sess):
        results = {}
        fetches = {}

        feed_dict = {self.inputs_img: inputs['images']}

        if self.mode == 'prediction':
            results['predictions'] = sess.run(self.action, feed_dict)

        return results

    def _epoch_end_callback(self, sess, sv, n_epochs):
        # Evaluate val accuracy
        val_error = 0

        # Initialize iterator with the training dataset
        sess.run(self.validation_init_iter)

        for i in range(self.val_steps_per_epoch):
            loss = sess.run(self.train_loss, feed_dict={self.is_training: False})
            val_error += loss
        val_error = val_error / self.val_steps_per_epoch

        # Log to Tensorflow board
        val_sum = sess.run(self.val_loss_sum, feed_dict={
            self.validation_loss: val_error})
        sv.summary_writer.add_summary(val_sum, self.completed_epochs)
        print("Epoch [{}] Validation Loss: {}".format(
            self.completed_epochs, val_error))

        # Model Saving
        if val_error < self.min_val_loss:
            self.save(sess, self.config.checkpoint_dir, 'best')
            self.min_val_loss = val_error

        if n_epochs % self.config.save_freq == 0:
            self.save(sess, self.config.checkpoint_dir, self.completed_epochs)

        if (n_epochs == self.config.max_epochs):
            print("------------------------------")
            print("Training finished successfully")
            print("------------------------------")
            return True
