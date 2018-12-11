import cv2
import tensorflow as tf
import time


def frozen_tf_inference(frozen_model, query_img, num_iterations, print_ops_in_graph=False):
    print("Feed frozen model from TF...")

    # load graph
    with tf.gfile.GFile(frozen_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # (optional) print all operations in computation graph
    if print_ops_in_graph:
        for op in graph.get_operations():
            print(op.name)

    # access the input and output nodes
    x = graph.get_tensor_by_name('raw_input:0')
    y = graph.get_tensor_by_name('final_prediction:0')

    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There are no variables in this graph, only hardcoded constants
        image = cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB)
        image = tf.expand_dims(image, 0)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0).eval()

        start_time = time.time()
        for i in range(num_iterations):
            prediction = sess.run(y, feed_dict={
                x: image
            })
        print("[{n}] inferences took [{s}] seconds.".format(n=num_iterations, s=(time.time() - start_time)))

        return prediction