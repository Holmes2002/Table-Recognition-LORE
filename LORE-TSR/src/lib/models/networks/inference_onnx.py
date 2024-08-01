import onnx
import onnxruntime as ort
import torch
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path
)

# Define the custom operation
@onnx_op(op_type='_DCNv2', domain='ai.onnx.contrib',
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float], 
         outputs=[PyCustomOpDef.dt_float])
def _DCNv2(x, y, z, p, q):
    # Implement the actual functionality here
    return q

onnx_model_path = "model.onnx"
so = ort.SessionOptions()
so.register_custom_ops_library(_get_library_path())
so.log_severity_level = 0  # Set logging level to verbose

# Helper function to convert PyTorch tensor to NumPy array
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Function to perform inference
def infer(ort_session, input_tensor):
    inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, inputs)
    return ort_outs

# Load and check the ONNX model
try:
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is well-formed.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    raise

# Inspect the BatchNormalization node
for node in onnx_model.graph.node:
    if node.op_type == "BatchNormalization":
        print(f"Node name: {node.name}")
        for attr in node.attribute:
            print(f"{attr.name}: {attr}")

# Create an inference session
try:
    ort_session = ort.InferenceSession(onnx_model_path, so)
    print("Inference session created successfully.")
except Exception as e:
    print(f"Error creating inference session: {e}")
    raise

dummy_input = torch.randn(1, 3, 1024, 1024)  # Adjust shape if necessary

# Perform inference with detailed logging
try:
    outputs = infer(ort_session, dummy_input)
    print("Model output shape:", outputs[0].shape)
except Exception as e:
    print(f"Error during inference: {e}")
    raise

