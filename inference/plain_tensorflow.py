import tensorflow as tf
import cv2
import numpy as np
from training.src.models.nets import cnn, mean_predictor, variance_predictor


def plain_tf_inference(checkpoint, query_img):
    print("plain TF inference...")

    # Build test graph
    image_height = 240
    image_width = 320
    num_channels = 3
    action_dim = 4

    input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
                                            image_width, num_channels],
                                 name='raw_input')

    image = tf.cast(input_uint8, dtype=tf.float32)
    image = tf.divide(image, 255.0)

    image_descriptors = cnn(image,
                            is_training=False,
                            scope='CNN')

    mean_prediction = mean_predictor(image_descriptors=image_descriptors, output_dim=action_dim,
                                     is_training=False, scope='Mean_Prediction')
    variance_prediction = variance_predictor(image_descriptors=image_descriptors, output_dim=action_dim,
                                             is_training=False, scope='Variance_Prediction')

    prediction = tf.concat([mean_prediction, variance_prediction], axis=1)

    # Load model
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint)
        print("--------------------------------------------------")
        print("Restored checkpoint file {}".format(checkpoint))
        print("--------------------------------------------------")
        # Do inference
        debug_img = cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB)

        ## debug stuff
        # self.predict_pose(cv2.cvtColor(cv2.imread("/home/elia/movidius_tutorials/frame_00000.png"), cv2.COLOR_BGR2RGB))
        inputs = {'images': debug_img[None]}
        feed_dict = {input_uint8: inputs['images']}
        results = sess.run(prediction, feed_dict=feed_dict)



        predictions = np.squeeze(results)
        print("flattened tensor")
        print(predictions.shape)

    return predictions