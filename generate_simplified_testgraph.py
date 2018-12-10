import os

import tensorflow as tf
from models.nets import cnn, mean_predictor, variance_predictor_no_exp

checkpoint_input_dir = 'training/results'
checkpoint_output_dir = 'graph_data'
checkpoint_filename = 'model.best'
test_graph_filename = 'test_graph'

##############################
# build graph
##############################

image_height = 240
image_width = 320
num_channels = 3
action_dim = 4

input_float32 = tf.placeholder(tf.float32, [1, image_height,
                                            image_width, num_channels],
                               name='raw_input')

image_descriptors = cnn(input_float32,
                        is_training=False,
                        scope='CNN')

mean_prediction = mean_predictor(image_descriptors=image_descriptors, output_dim=action_dim,
                                 is_training=False, scope='Mean_Prediction')
variance_prediction = variance_predictor_no_exp(image_descriptors=image_descriptors, output_dim=action_dim,
                                                is_training=False, scope='Variance_Prediction')

prediction = tf.concat([mean_prediction, variance_prediction], axis=1)
prediction = tf.identity(prediction, name="final_prediction")

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
