import os

import tensorflow as tf

from models.nets import cnn, mean_predictor, variance_predictor_no_exp

checkpoint_input_dir = '/home/elia/code/catkin_adr/src/adr_icra19/learning_code/gate_regression_learner/best_checkpoint/debug'
checkpoint_output_dir = '/home/elia/Desktop/movidius_checkpoint/181203'
checkpoint_filename = 'model.best'
test_graph_filename = 'test_graph'


def preprocess_image(image):
    """ Preprocess an input image
    Args:
        Image: A uint8 tensor
    Returns:
        image: A preprocessed float32 tensor.
    """
    # image = tf.image.resize_images(image,
    #                                [self.config.img_height, self.config.img_width])
    # image = tf.cast(image, dtype=tf.float32)
    # image = tf.divide(image, 255.0)
    return image


##############################
# build graph
##############################

image_height = 240
image_width = 320
num_channels = 3
action_dim = 4
# input_uint8 = tf.placeholder(tf.uint8, [1, image_height,
#                                         image_width, num_channels],
#                              name='raw_input')
#
# input_mc = preprocess_image(input_uint8)

input_float32 = tf.placeholder(tf.float32, [1, image_height,
                                            image_width, num_channels],
                               name='raw_input')

# input_float32 = tf.placeholder(tf.float32, [1, num_channels, image_height,
#                                             image_width],
#                                name='raw_input')

input_mc = preprocess_image(input_float32)
# flattened_input = tf.contrib.layers.flatten(input_mc)
image_descriptors = cnn(input_mc,
                        is_training=False,
                        scope='CNN')
# flattened_descriptors = tf.layers.flatten(image_descriptors)

mean_prediction = mean_predictor(image_descriptors=image_descriptors, output_dim=action_dim,
                                 is_training=False, scope='Mean_Prediction')
variance_prediction = variance_predictor_no_exp(image_descriptors=image_descriptors, output_dim=action_dim,
                                                is_training=False, scope='Variance_Prediction')

prediction = tf.concat([mean_prediction, variance_prediction], axis=1)
prediction = tf.identity(prediction, name="final_prediction")
# prediction = tf.identity(mean_prediction, name="final_prediction")


##############################
# load checkpoint
##############################
saver = tf.train.Saver([var for var in tf.trainable_variables()])

with tf.Session() as sess:
    # write the graph to file
    tf.train.write_graph(sess.graph_def, checkpoint_output_dir, 'graph.pbtxt')

    saver.restore(sess, os.path.join(checkpoint_input_dir, checkpoint_filename))
    print("--------------------------------------------------")
    print("Restored checkpoint file {}".format(os.path.join(checkpoint_input_dir, checkpoint_filename)))
    print("--------------------------------------------------")

    ##############################
    # save graph
    ##############################
    saver.save(sess, os.path.join(checkpoint_output_dir, test_graph_filename))
    print(" [*] Saving checkpoint to {}/{}.best".format(checkpoint_output_dir,
                                                        test_graph_filename))
