import inference.plain_tensorflow as plain_tf
import inference.use_frozen_graph as frozen_tf


print("Comparing the results of different inference methods...")


checkpoint = 'graph_data/model.best'
frozen_model = 'graph_data/frozen_graph.pb'
query_img = 'query_imgs/frame_00000.png'



print("Plain tensorflow...")
predictions_plain_tf = plain_tf.plain_tf_inference(checkpoint, query_img)
print(predictions_plain_tf)


print("Feed frozen graph using tensorflow...")
predictions_frozen_tf = frozen_tf.frozen_tf_inference(frozen_model, query_img, print_ops_in_graph=False)
print(predictions_frozen_tf)



print("Done.")