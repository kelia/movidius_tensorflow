import inference.plain_tensorflow as plain_tf
import inference.use_frozen_graph as frozen_tf
import inference.openvino_inference as openvino


print("Comparing the results of different inference methods...")


checkpoint = 'graph_data/model.best'
frozen_model = 'graph_data/frozen_graph.pb'
intermediate_rep_cpu = 'graph_data/optimized_graph_FP32'
intermediate_rep_myriad = 'graph_data/optimized_graph_FP16'
query_img = 'query_imgs/frame_00000.png'



print("Plain tensorflow...")
predictions_plain_tf = plain_tf.plain_tf_inference(checkpoint, query_img)
print(predictions_plain_tf)


print("Feed frozen graph using tensorflow...")
predictions_frozen_tf = frozen_tf.frozen_tf_inference(frozen_model, query_img, print_ops_in_graph=False)
print(predictions_frozen_tf)


print("OpenVino inference on CPU...")
predictions_openvino_cpu = openvino.openvino_inference(intermediate_rep_cpu, query_img, device="CPU")
print(predictions_openvino_cpu)


print("OpenVino inference on MYRIAD...")
predictions_openvino_myriad = openvino.openvino_inference(intermediate_rep_myriad, query_img, device="MYRIAD")
print(predictions_openvino_myriad)



print("Done.")