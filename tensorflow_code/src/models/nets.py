import tensorflow as tf


def cnn(img_input, is_training,
        scope='Prediction', reuse=False, padding='same', data_format='NHWC'):
    """
    Define model architecture.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output trajectories (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       model: Logits on output trajectories
    """
    with tf.variable_scope(scope, reuse=reuse):
        x1 = tf.layers.conv2d(inputs=img_input, filters=32, kernel_size=5, strides=[2, 2], padding=padding,
                              data_format='channels_last')
        x1 = tf.layers.max_pooling2d(x1, pool_size=2, strides=2)

        # First residual block
        x2 = tf.nn.leaky_relu(x1, alpha=0.2)
        # x2 = tf.pad(x2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x2 = tf.layers.conv2d(inputs=x2, filters=32, kernel_size=3, strides=[2, 2],
                              padding=padding,
                              data_format='channels_last')
        # x2 = tf.pad(x2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x2 = tf.layers.conv2d(inputs=x2, filters=32, kernel_size=3, strides=[1, 1],
                              padding=padding,
                              data_format='channels_last')
        x1 = tf.layers.conv2d(x1, filters=32, kernel_size=1, strides=[2, 2], padding=padding)
        x3 = x2 + x1

        # Second residual block
        x4 = tf.nn.leaky_relu(x3, alpha=0.2)
        # x4 = tf.pad(x4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x4 = tf.layers.conv2d(inputs=x4, filters=64, kernel_size=3, strides=[2, 2],
                              padding=padding,
                              data_format='channels_last')
        # x4 = tf.pad(x4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x4 = tf.layers.conv2d(inputs=x4, filters=64, kernel_size=3, strides=[1, 1],
                              padding=padding,
                              data_format='channels_last')
        x3 = tf.layers.conv2d(x3, filters=64, kernel_size=1, strides=[2, 2], padding=padding)
        x5 = x4 + x3

        # Third residual block
        x6 = tf.nn.leaky_relu(x5, alpha=0.2)
        # x6 = tf.pad(x6, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x6 = tf.layers.conv2d(inputs=x6, filters=64, kernel_size=3, strides=[2, 2],
                              padding=padding,
                              data_format='channels_last')
        # x6 = tf.pad(x6, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x6 = tf.layers.conv2d(inputs=x6, filters=64, kernel_size=3, strides=[1, 1],
                              padding=padding,
                              data_format='channels_last')
        x5 = tf.layers.conv2d(x5, filters=64, kernel_size=1, strides=[2, 2], padding=padding)
        x7 = x6 + x5

        image_descriptors = tf.nn.leaky_relu(x7, alpha=0.2)
        image_descriptors = tf.layers.flatten(image_descriptors)

    return image_descriptors


def mean_predictor(image_descriptors, output_dim, is_training,
                   scope='Prediction', reuse=False, l2_reg_scale=1.0, data_format='NCHW'):
    """
    Define model architecture.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output trajectories (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       model: Logits on output trajectories
    """
    with tf.variable_scope(scope, reuse=reuse):
        x1 = tf.layers.dense(image_descriptors, units=128)
        x1 = tf.layers.dropout(x1, rate=0.5, training=is_training)

        x2 = tf.nn.leaky_relu(x1, alpha=0.2)
        x2 = tf.layers.dense(x2, units=64)
        x2 = tf.layers.dropout(x2, rate=0.5, training=is_training)

        x3 = tf.nn.leaky_relu(x2, alpha=0.2)
        x3 = tf.layers.dense(x3, units=32)
        x3 = tf.layers.dropout(x3, rate=0.5, training=is_training)

        mean = tf.layers.dense(x3, units=output_dim, activation=None)

    return mean


def variance_predictor(image_descriptors, output_dim, is_training,
                       scope='Prediction', reuse=False, l2_reg_scale=1.0, data_format='NCHW'):
    """
    Define model architecture.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output trajectories (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       model: Logits on output trajectories
    """
    with tf.variable_scope(scope, reuse=reuse):
        x1 = tf.layers.dense(image_descriptors, units=128)
        x1 = tf.layers.dropout(x1, rate=0.5, training=is_training)

        x2 = tf.nn.leaky_relu(x1, alpha=0.2)
        x2 = tf.layers.dense(x2, units=64)
        x2 = tf.layers.dropout(x2, rate=0.5, training=is_training)

        x3 = tf.nn.leaky_relu(x2, alpha=0.2)
        x3 = tf.layers.dense(x3, units=32)
        x3 = tf.layers.dropout(x3, rate=0.5, training=is_training)

        variance_raw = tf.layers.dense(x3, units=output_dim, activation=None)

        # exp assures positive variance and has nice resolution properties
        variance = tf.exp(variance_raw)
        variance_radius = tf.slice(variance, [0, 0], [tf.shape(variance_raw)[0], 1])
        variance_angles = tf.slice(variance, [0, 1], [tf.shape(variance_raw)[0], 2])
        variance_yaw = tf.slice(variance, [0, 3], [tf.shape(variance_raw)[0], 1])
        epsilon_radius = tf.fill(tf.shape(variance_radius), 5.0e-3)
        epsilon_angles = tf.fill(tf.shape(variance_angles), 0.01)
        epsilon_yaw = tf.fill(tf.shape(variance_yaw), 0.17)  # corresponds to 10 degree
        variance_radius = variance_radius + epsilon_radius
        variance_angles = variance_angles + epsilon_angles
        variance_yaw = variance_yaw + epsilon_yaw
        predicted_variance = tf.concat([variance_radius, variance_angles, variance_yaw], axis=1)

    return predicted_variance


def variance_predictor_no_exp(image_descriptors, output_dim, is_training,
                              scope='Prediction', reuse=False, l2_reg_scale=1.0, data_format='NCHW'):
    """
    Define model architecture.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output trajectories (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       model: Logits on output trajectories
    """
    with tf.variable_scope(scope, reuse=reuse):
        x1 = tf.layers.dense(image_descriptors, units=128)
        x1 = tf.layers.dropout(x1, rate=0.5, training=is_training)

        x2 = tf.nn.leaky_relu(x1, alpha=0.2)
        x2 = tf.layers.dense(x2, units=64)
        x2 = tf.layers.dropout(x2, rate=0.5, training=is_training)

        x3 = tf.nn.leaky_relu(x2, alpha=0.2)
        x3 = tf.layers.dense(x3, units=32)
        x3 = tf.layers.dropout(x3, rate=0.5, training=is_training)

        variance_raw = tf.layers.dense(x3, units=output_dim, activation=None)

    return variance_raw
