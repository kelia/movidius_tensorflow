from colored import fg, bg, attr

import inference.openvino_inference as openvino
import inference.plain_tensorflow as plain_tf
import inference.use_frozen_graph as frozen_tf

print("Comparing the results of different inference methods...")

checkpoint = 'graph_data/model.best'
frozen_model = 'graph_data/frozen_graph.pb'
intermediate_rep_cpu = 'graph_data/optimized_graph_FP32'
intermediate_rep_myriad = 'graph_data/optimized_graph_FP16'
query_img = 'query_imgs/frame_00000.png'

print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
print("Plain tensorflow inference")
print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
predictions_plain_tf = plain_tf.plain_tf_inference(checkpoint, query_img)
print(predictions_plain_tf)

print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
print("Feed frozen graph using tensorflow")
print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
predictions_frozen_tf = frozen_tf.frozen_tf_inference(frozen_model, query_img, print_ops_in_graph=False)
print(predictions_frozen_tf)

# We first test MYRIAD, then CPU (vice versa leads to segfault...)
print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
print("OpenVino inference on MYRIAD")
print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
predictions_openvino_myriad = openvino.openvino_inference(intermediate_rep_myriad, query_img, device="MYRIAD")
print(predictions_openvino_myriad)

print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
print("OpenVino inference on CPU")
print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
predictions_openvino_cpu = openvino.openvino_inference(intermediate_rep_cpu, query_img, device="CPU")
print(predictions_openvino_cpu)

print("Done.")
