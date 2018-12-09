from __future__ import print_function

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


class OpenVinoDetector(object):
    """
    OpenVinoDetector
    """

    def __init__(self, model_name):
        model_xml = model_name + '.xml'
        model_bin = model_name + '.bin'

        # Load TFLite model and allocate tensors.
        plugin = IEPlugin(device='MYRIAD')
        # plugin = IEPlugin(device='CPU')
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))

        self.exec_net = plugin.load(network=net)

    def preprocess(self, input_image):
        """ Performs preprocessing on the input image to prepare it for inference.
       """
        preproc_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        preproc_image = preproc_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        preproc_image = preproc_image.astype(np.float16)
        preproc_image *= 0.00392156862745098  # 1.0 / 255.0
        return preproc_image

    def predict(self, image):
        """ """
        img = self.preprocess(image)
        infer_req_hdl = self.exec_net.start_async(request_id=0, inputs={self.input_blob: [img]})
        infer_req_hdl.wait()

        det_out = infer_req_hdl.outputs[self.out_blob]
        return det_out


t = OpenVinoDetector('/home/elia/Desktop/movidius_checkpoint/181203/optimized_graph_FP16')
# t = OpenVinoDetector('/home/elia/Desktop/movidius_checkpoint/181203/optimized_graph_FP32')
image = cv2.imread('/home/elia/movidius_tutorials/frame_00000.png')
prediction = t.predict(image)
print(prediction[0][0:10])
print('DONE')

