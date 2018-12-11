from __future__ import print_function

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
import time


class OpenVinoDetector(object):
    """
    OpenVinoDetector
    """

    def __init__(self, model_name, device='CPU'):
        model_xml = model_name + '.xml'
        model_bin = model_name + '.bin'

        plugin = IEPlugin(device=device)
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))

        self.exec_net = plugin.load(network=net)

    def preprocess(self, input_image):
        """ Performs preprocessing on the input image to prepare it for inference.
       """
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.astype(np.float16)
        image *= 0.00392157  # 1.0 / 255.0

        return image

    def predict(self, image):
        """ """
        img = self.preprocess(image)
        infer_req_hdl = self.exec_net.start_async(request_id=0, inputs={self.input_blob: [img]})
        infer_req_hdl.wait()

        det_out = infer_req_hdl.outputs[self.out_blob]
        return det_out


def openvino_inference(intermediate_rep, query_img, num_iterations, device):
    t = OpenVinoDetector(intermediate_rep, device)
    image = cv2.imread(query_img)

    start_time = time.time()
    for i in range(num_iterations):
        prediction = t.predict(image)
    print("[{n}] inferences took [{s}] seconds.".format(n=num_iterations, s=(time.time() - start_time)))

    return prediction
