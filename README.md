# Inference for tensorflow models on the Movidius Neural Compute Stick (NCS)

The documentation of openvino gives many examples, how to run specific network architectures by converting the model using their model optimizer. However, I could not find a single example that starts at the actual tensorflow code, and porting the model to the graph version needed for inference using the inference engine. 

This repo aims at giving two examples of the workflow, specifically the following two architectures will be covered:
 - simple, two-layer network (conv layer followed by fully connected layer, no activation functions)
 - simple convolutional network with some residual connections and two fully connected heads.


## 0. Train the network (optionally)
For completeness, I added code to generate your own checkpoints. To just check out the conversion from tensorflow checkpoints to the optimized intel representation, skip this section.

## 1. Save a frozen testgraph

### 1.1 Simplify the computation graph
When saving a model in tensorflow during training, the input pipeline is usually part of the graph and therefore included in the checkpoint. Since this part of the computation graph is not needed at inference time, in a first step, a reduced version of the graph is generated that only contains operations that are actually executed during inference. 

```
python generate_simplified_graph.py
```

### 1.2 Freeze the graph
Although the test graph is now simplified and only contains relevant operations for test time inference, the graph is still saved in classical tensorflow style (seperate files for graph topology and weights). To feed the graph to the model optimizer in the next step (and generally for ease of deployment), the model is saved in a single file containing both weights and graph topology. 
```
python freeze_graph.py
```

## Convert the model to an optimized Intermediate Representation (IR)
To perform this step, the openvino toolkit needs to be installed first.

1. Register and download the openvino toolkit from [here](https://software.intel.com/en-us/neural-compute-stick/get-started). You will receive an email with the download link. Choose the linux version (without FPGA support).

2. Follow the installation steps [here](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-2)
   * **Attention**: If you want to port a **new** network to the movidius stick, you need to configure the model optimizer. Else you can skip this step.  
   * Note: you can configure the model optimizer for multiple frameworks (TF, Caffe, ...) or only for a specific one. 
   * Perform [these additional steps](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-3-2) for the platform where the NCS is used.
