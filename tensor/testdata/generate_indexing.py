"""Generate NN-08 exact typed Concat/Gather fixtures with pinned references.

Development only; ordinary Go tests consume indexing.json offline.
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


def values_for(shape, dtype, seed):
    count = math.prod(shape)
    values = np.arange(count, dtype=dtype) + dtype(seed)
    if seed == 0 and dtype == np.float32:
        for index, pattern in enumerate([0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000]):
            if index < count:
                values.view(np.uint32)[index] = pattern
    elif dtype == np.int64:
        values += np.int64(16777217)
        if seed == 0 and count > 0:
            values[0] = np.iinfo(np.int64).max
        if seed == 0 and count > 1:
            values[1] = np.iinfo(np.int64).min
    return values.reshape(shape)


def evaluate(name, operation, shapes, axis, indices_shape, indices, dtype):
    tensor_type = TensorProto.FLOAT if dtype == np.float32 else TensorProto.INT64
    arrays = [values_for(shape, dtype, index * 100) for index, shape in enumerate(shapes)]
    feeds = {f"input{index}": array for index, array in enumerate(arrays)}
    inputs = [helper.make_tensor_value_info(key, tensor_type, list(value.shape)) for key, value in feeds.items()]
    node_inputs = list(feeds)
    if operation == "Concat":
        expected = np.concatenate(arrays, axis=axis)
    else:
        index_values = np.array(indices, dtype=np.int64).reshape(indices_shape)
        expected = np.take(arrays[0], index_values, axis=axis)
        feeds["indices"] = index_values
        inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape))
        node_inputs.append("indices")
    graph = helper.make_graph(
        [helper.make_node(operation, node_inputs, ["output"], axis=axis)],
        name,
        inputs,
        [helper.make_tensor_value_info("output", tensor_type, None)],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    checked = ReferenceEvaluator(model).run(None, feeds)[0]
    difference = ""
    reference = "ONNX 1.16.2 ReferenceEvaluator, IR 8/opset 13; NumPy 1.26.4 cross-check"
    if list(checked.shape) != list(expected.shape):
        assert operation == "Gather" and len(indices) == 0
        assert list(checked.shape) == [0]
        difference = "ONNX reference empty-indices shortcut returns [0] instead of prefix + indices shape + suffix"
        reference = "NumPy 1.26.4 take; ONNX 1.16.2 empty-indices shape difference recorded"
    assert payload(checked) == payload(expected)
    return {
        "name": f"{name} {np.dtype(dtype).name}",
        "operation": operation,
        "data_type": np.dtype(dtype).name,
        "inputs": [{"shape": list(array.shape), **payload(array)} for array in arrays],
        "axis": axis,
        "indices_shape": indices_shape,
        "indices": indices,
        "output_shape": list(expected.shape),
        "output": payload(expected),
        "reference": reference,
        "reference_shape": list(checked.shape),
        "reference_difference": difference,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

concat_cases = [
    ("concat feature channels", [[1, 1, 2, 2], [1, 2, 2, 2]], 1),
    ("concat batches", [[1, 2, 2, 1], [2, 2, 2, 1]], 0),
    ("concat final negative axis", [[2, 2], [2, 1]], -1),
    ("concat middle axis", [[2, 1, 2], [2, 3, 2]], 1),
    ("concat shape vectors", [[2], [0], [3]], 0),
    ("concat single input", [[2, 3]], -2),
    ("concat empty axis input", [[2, 0, 3], [2, 2, 3]], 1),
    ("concat all empty axis", [[2, 0, 3], [2, 0, 3]], 1),
    ("concat empty prefix", [[0, 2, 3], [0, 4, 3]], 1),
    ("concat empty suffix", [[2, 3, 0], [2, 1, 0]], 1),
    ("concat empty vectors", [[0], [0]], 0),
]
gather_cases = [
    ("gather vector scalar negative index", [4], 0, [], [-1]),
    ("gather matrix scalar index", [2, 3], 1, [], [1]),
    ("gather first axis", [2, 3, 2], 0, [2], [1, 0]),
    ("gather middle multidimensional", [2, 3, 2], 1, [2, 2], [-1, 0, 1, 1]),
    ("gather final negative axis", [2, 2, 3], -1, [3], [-3, -1, 0]),
    ("gather repeated rows", [2, 3], 0, [3], [1, 1, 0]),
    ("gather empty vector indices", [2, 3], 1, [0], []),
    ("gather empty multidimensional indices", [2, 3, 4], 1, [2, 0, 1], []),
    ("gather zero data axis", [2, 0, 3], 1, [0], []),
    ("gather empty prefix", [0, 3, 2], 1, [2], [2, 0]),
    ("gather empty suffix", [2, 3, 0], 1, [2], [-3, -1]),
    ("gather scalar index empty prefix", [0, 3, 2], 1, [], [1]),
    ("gather row index matrix", [2, 3], 0, [2, 1], [1, 0]),
]
fixtures = []
for dtype in [np.float32, np.int64]:
    fixtures.extend(evaluate(name, "Concat", shapes, axis, [], [], dtype) for name, shapes, axis in concat_cases)
    fixtures.extend(
        evaluate(name, "Gather", [shape], axis, indices_shape, indices, dtype)
        for name, shape, axis, indices_shape, indices in gather_cases
    )
output_path = Path(__file__).with_name("indexing.json")
output_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases to {output_path}")
