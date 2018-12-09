# Inference for tensorflow models on the Movidius Neural Compute Stick (NCS)

The documentation of openvino gives many examples, how to run specific network architectures by converting the model using their model optimizer. However, I could not find a single example that starts at the actual tensorflow code, and porting the model to the graph version needed for inference using the inference engine. 

This repo aims at giving two examples of the workflow, specifically the following two architectures will be covered:
 - simple, two-layer network (conv layer followed by fully connected layer, no activation functions)
 - simple convolutional network with some residual connections and two fully connected heads.


## Train the network (optionally)
For completeness, I added code to generate your own checkpoints. To just check out the conversion from tensorflow checkpoints to the optimized intel representation, skip this section.

## Save a frozen testgraph

### Simplify the computation graph
When saving a model in tensorflow during training, the input pipeline is usually part of the graph and therefore included in the checkpoint. Since this part of the computation graph is not needed at inference time, in a first step, a reduced version of the graph is generated that only contains operations that are actually executed during inference. 

```
python generate_simplified_graph.py
```

### Freeze the graph
