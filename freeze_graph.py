from tensorflow.python.tools import freeze_graph

MODEL_DIR = 'graph_data/'

##########################3
import tensorflow as tf
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

gdef = gpb.GraphDef()

with open(MODEL_DIR + 'graph.pbtxt', 'r') as fh:
    graph_str = fh.read()

pbtf.Parse(graph_str, gdef)

tf.import_graph_def(gdef)

# print([n.name for n in tf.get_default_graph().as_graph_def().node])

###############################


# Freeze the graph
input_graph_path = MODEL_DIR + 'graph.pbtxt'
checkpoint_path = MODEL_DIR + 'test_graph'
input_saver_def_path = ""
input_binary = False
output_node_names = "final_prediction"
restore_op_name = ""
filename_tensor_name = ""
output_frozen_graph_name = MODEL_DIR + 'frozen_graph.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
