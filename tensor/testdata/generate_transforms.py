"""Generate NN-07 exact typed fixtures with pinned ONNX/NumPy references.

Development only; ordinary Go tests consume transforms.json offline.
"""

import json
import math
from pathlib import Path

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def payload(values):
    if values.dtype == np.float32:
        return {"bits": values.reshape(-1).view(np.uint32).tolist()}
    return {"integers": values.reshape(-1).tolist()}


def evaluate(name, operation, shape, argument, dtype):
    count = math.prod(shape)
    if dtype == np.float32:
        values = np.arange(count, dtype=np.float32) + np.float32(0.25)
        for index, pattern in enumerate([0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000]):
            if index < count:
                values.view(np.uint32)[index] = pattern
        tensor_type = TensorProto.FLOAT
    else:
        values = np.arange(count, dtype=np.int64) + np.int64(16777217)
        if count > 0:
            values[0] = np.iinfo(np.int64).max
        if count > 1:
            values[1] = np.iinfo(np.int64).min
        tensor_type = TensorProto.INT64
    values = values.reshape(shape)
    feeds = {"input": values}
    inputs = [helper.make_tensor_value_info("input", tensor_type, shape)]
    node_inputs = ["input"]
    attributes = {}
    reference = "ONNX 1.16.2 ReferenceEvaluator, IR 8/opset 13"
    reference_error = ""
    if operation == "Reshape":
        output = np.reshape(values, argument)
        feeds["argument"] = np.array(argument, dtype=np.int64)
        if 0 in argument:
            reference = "NumPy 1.26.4 concrete reshape; ONNX zero policy resolved by caller"
    elif operation == "Flatten":
        normalized = argument if argument >= 0 else argument + len(shape)
        target = [math.prod(shape[:normalized]), math.prod(shape[normalized:])]
        output = np.reshape(values, target)
        attributes["axis"] = argument
    elif operation == "Transpose":
        normalized = [axis % len(shape) for axis in argument]
        output = np.transpose(values, axes=normalized)
        if normalized:
            attributes["perm"] = normalized
    elif operation == "Squeeze":
        output = np.squeeze(values, axis=tuple(argument))
        feeds["argument"] = np.array(argument, dtype=np.int64)
    else:
        output = np.expand_dims(values, axis=tuple(argument))
        feeds["argument"] = np.array(argument, dtype=np.int64)
    if "argument" in feeds:
        node_inputs.append("argument")
        inputs.append(helper.make_tensor_value_info("argument", TensorProto.INT64, [len(argument)]))
    if reference.startswith("ONNX"):
        graph = helper.make_graph(
            [helper.make_node(operation, node_inputs, ["output"], **attributes)],
            name,
            inputs,
            [helper.make_tensor_value_info("output", tensor_type, None)],
        )
        model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
        try:
            checked = ReferenceEvaluator(model).run(None, feeds)[0]
        except ValueError as error:
            if operation != "Flatten" or math.prod(shape) != 0:
                raise
            reference_error = str(error)
            reference = "NumPy 1.26.4 concrete flatten products; ONNX reference inferred -1 rejects empty prefix"
        else:
            assert list(checked.shape) == list(output.shape)
            assert payload(checked) == payload(output)
    return {
        "name": f"{name} {np.dtype(dtype).name}",
        "operation": operation,
        "data_type": np.dtype(dtype).name,
        "input_shape": shape,
        "input": payload(values),
        "axis": argument if operation == "Flatten" else 0,
        "axes": argument if operation in ["Transpose", "Squeeze", "Unsqueeze"] else [],
        "target": argument if operation == "Reshape" else [],
        "output_shape": list(output.shape),
        "output": payload(output),
        "reference": reference,
        "reference_error": reference_error,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

cases = [
    ("reshape scalar to units", "Reshape", [], [1, 1]),
    ("reshape units to scalar", "Reshape", [1, 1], []),
    ("reshape rectangular", "Reshape", [2, 3, 2], [3, 4]),
    ("reshape literal empty target", "Reshape", [0, 3], [3, 0]),
    ("flatten scalar", "Flatten", [], 0),
    ("flatten first endpoint", "Flatten", [2, 3], 0),
    ("flatten final endpoint", "Flatten", [2, 3], 2),
    ("flatten negative split", "Flatten", [2, 3, 2], -1),
    ("flatten empty suffix", "Flatten", [2, 0, 3], 1),
    ("flatten empty prefix", "Flatten", [2, 0, 3], 2),
    ("transpose scalar", "Transpose", [], []),
    ("transpose rectangular", "Transpose", [2, 3], [1, 0]),
    ("transpose rank three", "Transpose", [2, 3, 2], [2, 0, 1]),
    ("transpose negative permutation", "Transpose", [2, 3, 2], [-1, -3, -2]),
    ("transpose empty", "Transpose", [2, 0, 3], [2, 0, 1]),
    ("squeeze scalar identity", "Squeeze", [], []),
    ("squeeze selected negative axes", "Squeeze", [1, 2, 1, 3, 1], [0, -1, 2]),
    ("squeeze to scalar", "Squeeze", [1, 1], [1, 0]),
    ("squeeze empty axes identity", "Squeeze", [1, 2, 1], []),
    ("squeeze empty value", "Squeeze", [1, 0, 1, 3], [0, 2]),
    ("unsqueeze scalar", "Unsqueeze", [], [0]),
    ("unsqueeze scalar two axes", "Unsqueeze", [], [1, 0]),
    ("unsqueeze final rank axes", "Unsqueeze", [2, 3], [0, -1]),
    ("unsqueeze unsorted axes", "Unsqueeze", [2, 3], [3, 0]),
    ("unsqueeze negative axes", "Unsqueeze", [2, 3], [-3, -1]),
    ("unsqueeze empty axes identity", "Unsqueeze", [1, 2, 1], []),
    ("unsqueeze empty value", "Unsqueeze", [0, 3], [1, -1]),
]
fixtures = [evaluate(*case, dtype) for dtype in [np.float32, np.int64] for case in cases]
output_path = Path(__file__).with_name("transforms.json")
output_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases to {output_path}")
