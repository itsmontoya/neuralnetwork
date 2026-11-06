"""Generate NN-05 fixtures with ONNX 1.16.2 and NumPy 1.26.4.

Development only; normal Go tests read products.json without Python/ONNX.
"""

import json
from pathlib import Path

import google.protobuf
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def bits(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def sequence(shape, start=0):
    count = int(np.prod(shape, dtype=np.int64))
    return ((np.arange(count, dtype=np.float32) + start) % 13 - 6).reshape(shape)


def evaluate(name, operation, left, right, bias=None, trans_a=False,
             trans_b=False, alpha=1.0, beta=1.0, constant_weights=False):
    feeds = {"left": left, "right": right}
    inputs = [helper.make_tensor_value_info("left", TensorProto.FLOAT, list(left.shape))]
    initializers = []
    if constant_weights:
        initializers.append(numpy_helper.from_array(right, "right"))
        del feeds["right"]
    else:
        inputs.append(helper.make_tensor_value_info("right", TensorProto.FLOAT, list(right.shape)))
    names = ["left", "right"]
    attributes = {}
    if operation == "Gemm":
        attributes = {"transA": int(trans_a), "transB": int(trans_b),
                      "alpha": float(alpha), "beta": float(beta)}
        if bias is not None:
            names.append("bias")
            inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(bias.shape)))
            feeds["bias"] = bias
    graph = helper.make_graph(
        [helper.make_node(operation, names, ["output"], **attributes)], name,
        inputs, [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    with np.errstate(all="ignore"):
        output = ReferenceEvaluator(model).run(None, feeds)[0]
    return {
        "name": name, "operation": operation, "constant_weights": constant_weights,
        "left_shape": list(left.shape), "left_bits": bits(left),
        "right_shape": list(right.shape), "right_bits": bits(right),
        "has_bias": bias is not None,
        "bias_shape": list(bias.shape) if bias is not None else [],
        "bias_bits": bits(bias) if bias is not None else [],
        "transpose_a": trans_a, "transpose_b": trans_b,
        "alpha_bits": bits([alpha])[0], "beta_bits": bits([beta])[0],
        "output_shape": list(output.shape), "output_bits": bits(output),
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

fixtures = []
matmul_shapes = [
    ("vector dot", [3], [3]),
    ("matrix vector", [2, 3], [3]),
    ("vector matrix", [3], [3, 4]),
    ("rectangular", [2, 3], [3, 4]),
    ("vector batched matrix", [3], [2, 3, 4]),
    ("batched matrix vector", [2, 2, 3], [3]),
    ("two broadcast axes", [2, 1, 2, 3], [1, 3, 3, 4]),
    ("right batch", [2, 3], [2, 3, 4]),
    ("unequal batch ranks", [3, 2, 3], [2, 1, 3, 4]),
    ("empty vector dot", [0], [0]),
    ("empty contraction", [2, 0], [0, 3]),
    ("empty batched contraction", [2, 1, 3, 0], [1, 4, 0, 2]),
    ("empty batch", [0, 2, 3], [1, 3, 4]),
    ("empty rows", [2, 0, 3], [1, 3, 4]),
    ("empty columns", [2, 3], [3, 0]),
]
for index, (name, left_shape, right_shape) in enumerate(matmul_shapes):
    fixtures.append(evaluate(name, "MatMul", sequence(left_shape, 1), sequence(right_shape, 3),
                             constant_weights=index % 2 == 0))
fixtures.append(evaluate("fractional contraction", "MatMul", sequence([3, 17]) / np.float32(17),
                         sequence([17, 4], 4) / np.float32(9)))
fixtures.append(evaluate("nonfinite product", "MatMul",
                         np.array([[1, np.inf], [np.nan, 2]], dtype=np.float32),
                         np.array([[0, 1], [2, 3]], dtype=np.float32)))

left = sequence([2, 3], 1) / np.float32(2)
right = sequence([3, 4], 4) / np.float32(3)
biases = [
    ("no bias", None), ("scalar bias", np.array(2, dtype=np.float32)),
    ("vector bias", sequence([4], 2)), ("row bias", sequence([1, 4], 2)),
    ("column bias", sequence([2, 1], 2)), ("full bias", sequence([2, 4], 2)),
]
for trans_a in [False, True]:
    for trans_b in [False, True]:
        for name, bias in biases:
            fixtures.append(evaluate(
                f"transpose {int(trans_a)}{int(trans_b)} {name}", "Gemm",
                left.T.copy() if trans_a else left,
                right.T.copy() if trans_b else right, bias, trans_a, trans_b,
                alpha=0.5, beta=-2, constant_weights=trans_b,
            ))

fixtures.append(evaluate("zero beta ignores nonfinite bias", "Gemm", left, right,
                         np.array([np.inf, np.nan, -np.inf, -0.0], dtype=np.float32), beta=0))
fixtures.append(evaluate("negative zero beta ignores nonfinite bias", "Gemm", left, right,
                         np.array(np.nan, dtype=np.float32), beta=-0.0))
fixtures.append(evaluate("omitted bias ignores nonfinite beta", "Gemm", left, right, beta=np.nan))
fixtures.append(evaluate("zero alpha retains nonfinite product", "Gemm",
                         np.array([[np.inf]], dtype=np.float32), np.array([[1]], dtype=np.float32),
                         alpha=0))
fixtures.append(evaluate("empty contraction with bias", "Gemm", sequence([2, 0]), sequence([0, 3]),
                         sequence([3], 1), alpha=2, beta=0.5))
fixtures.append(evaluate("empty contraction infinite alpha", "Gemm", sequence([2, 0]), sequence([0, 3]),
                         alpha=np.inf))
fixtures.append(evaluate("empty Gemm rows", "Gemm", sequence([0, 3]), sequence([3, 4]), sequence([4])))
fixtures.append(evaluate("empty Gemm columns", "Gemm", sequence([2, 3]), sequence([3, 0]), sequence([0])))

output_path = Path(__file__).with_name("products.json")
output_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases to {output_path}")
