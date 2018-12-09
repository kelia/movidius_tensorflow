import cv2
import argparse
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="/home/elia/Desktop/movidius_checkpoint/181203/frozen_graph.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name('raw_input:0')
    y = graph.get_tensor_by_name('final_prediction:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        debug_img = cv2.cvtColor(cv2.imread("/home/elia/movidius_tutorials/frame_00000.png"), cv2.COLOR_BGR2RGB)
        debug_img = tf.expand_dims(debug_img, 0)
        debug_img = tf.cast(debug_img, dtype=tf.float32)
        debug_img = tf.divide(debug_img, 255.0).eval()

        y_out = sess.run(y, feed_dict={
            x: debug_img  # < 45
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out[0][0:10])  # [[ False ]] Yay, it works!