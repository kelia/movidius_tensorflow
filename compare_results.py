from colored import fg, bg, attr

import inference.openvino_inference as openvino
import inference.plain_tensorflow as plain_tf
import inference.use_frozen_graph as frozen_tf


def check_result(prediction, ground_truth):
    error = ((prediction - ground_truth) ** 2).sum()
    if error > 0.05:
        print ('%s%sResult different from tf graph inference!%s' % (fg('white'), bg('red'), attr('reset')))


if __name__ == '__main__':
    print("Comparing the results of different inference methods...")

    checkpoint = 'graph_data/model.best'
    frozen_model = 'graph_data/frozen_graph.pb'
    intermediate_rep_cpu = 'graph_data/optimized_graph_FP32'
    intermediate_rep_myriad = 'graph_data/optimized_graph_FP16'
    query_img = 'query_imgs/frame_00000.png'
    num_iterations = 1000

    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
    print("Plain tensorflow inference")
    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
    predictions_plain_tf = plain_tf.plain_tf_inference(checkpoint, query_img, num_iterations)
    print(predictions_plain_tf)

    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
    print("Feed frozen graph using tensorflow")
    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
    predictions_frozen_tf = frozen_tf.frozen_tf_inference(frozen_model, query_img, num_iterations, print_ops_in_graph=False)
    print(predictions_frozen_tf)
    check_result(predictions_frozen_tf, predictions_plain_tf)

    # We first test MYRIAD, then CPU (vice versa leads to segfault...)
    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
    print("OpenVino inference on MYRIAD")
    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
    predictions_openvino_myriad = openvino.openvino_inference(intermediate_rep_myriad, query_img, num_iterations, device="MYRIAD")
    print(predictions_openvino_myriad)
    check_result(predictions_openvino_myriad, predictions_plain_tf)

    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('blink')))
    print("OpenVino inference on CPU")
    print ('%s%s--------------------------------------------------%s' % (fg('white'), bg('blue'), attr('reset')))
    predictions_openvino_cpu = openvino.openvino_inference(intermediate_rep_cpu, query_img, num_iterations, device="CPU")
    print(predictions_openvino_cpu)
    check_result(predictions_openvino_cpu, predictions_plain_tf)

    print("Done.")
