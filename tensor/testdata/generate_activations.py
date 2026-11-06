"""Generate NN-06 fixtures with ONNX 1.16.2 and NumPy 1.26.4.

Development only; normal Go tests read activations.json without Python/ONNX.
"""

import json
from pathlib import Path

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def bits(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def evaluate(name, operation, values, axis=0):
    attributes = {"axis": axis} if operation == "Softmax" else {}
    graph = helper.make_graph(
        [helper.make_node(operation, ["input"], ["output"], **attributes)],
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(values.shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    reference_error = ""
    try:
        with np.errstate(all="ignore"):
            output = ReferenceEvaluator(model).run(None, {"input": values})[0]
    except ValueError as error:
        # The library deliberately accepts empty Softmax axes as validated
        # no-ops. Keep the pinned evaluator's rejection explicit in the data.
        if operation != "Softmax" or values.shape[axis] != 0:
            raise
        reference_error = str(error)
        output = np.empty_like(values)
    return {
        "name": name,
        "operation": operation,
        "axis": axis,
        "input_shape": list(values.shape),
        "input_bits": bits(values),
        "output_shape": list(output.shape),
        "output_bits": bits(output),
        "reference_error": reference_error,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

fixtures = []
elementwise = [
    ("scalar", np.array(0.5, dtype=np.float32)),
    ("vector", np.array([-100, -5, -1, -0.0, 0, 0.5, 1, 5, 100], dtype=np.float32)),
    ("rank two", np.array([[-3, 2, 0.25], [2, -4, 8]], dtype=np.float32)),
    ("rank three", np.linspace(-3, 3, 12, dtype=np.float32).reshape(2, 2, 3)),
    ("empty", np.empty([2, 0, 3], dtype=np.float32)),
    ("nonfinite", np.array([-np.inf, -0.0, 0, np.inf, np.nan], dtype=np.float32)),
]
for operation in ["Relu", "Sigmoid", "Tanh"]:
    for name, values in elementwise:
        fixtures.append(evaluate(f"{operation} {name}", operation, values))

for axis in [0, -1]:
    fixtures.append(evaluate(f"Softmax vector axis {axis}", "Softmax",
                             np.array([1, 2, 4, -3], dtype=np.float32), axis))
for axis in [0, 1, -2, -1]:
    fixtures.append(evaluate(f"Softmax rank two axis {axis}", "Softmax",
                             np.arange(6, dtype=np.float32).reshape(2, 3), axis))
for axis in [0, 1, 2, -3, -2, -1]:
    fixtures.append(evaluate(f"Softmax rank three axis {axis}", "Softmax",
                             np.linspace(-2, 2, 24, dtype=np.float32).reshape(2, 3, 4), axis))
fixtures.append(evaluate("Softmax large logits", "Softmax",
                         np.array([[1000, 1001, 999], [-1000, -999, -1001]], dtype=np.float32), 1))
fixtures.append(evaluate("Softmax singleton axis", "Softmax",
                         np.arange(6, dtype=np.float32).reshape(2, 1, 3), 1))
fixtures.append(evaluate("Softmax separate noncontiguous special slices", "Softmax",
                         np.array([[[1, np.inf], [2, 1], [-np.inf, 2]],
                                   [[np.nan, -np.inf], [0, -np.inf], [1, -np.inf]]], dtype=np.float32), 1))
fixtures.append(evaluate("Softmax special rows", "Softmax",
                         np.array([[1, np.nan, 3], [np.inf, 0, -np.inf],
                                   [-np.inf, -np.inf, -np.inf], [0, -np.inf, 1]], dtype=np.float32), 1))
for shape, axis in [([0, 3], 1), ([2, 0, 3], 1), ([2, 3, 0], 2), ([2, 0, 3], -1)]:
    fixtures.append(evaluate(f"Softmax empty {shape} axis {axis}", "Softmax",
                             np.empty(shape, dtype=np.float32), axis))

output_path = Path(__file__).with_name("activations.json")
output_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases to {output_path}")
