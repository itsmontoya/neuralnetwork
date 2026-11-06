"""Generate NN-04 fixtures with the pinned ONNX reference evaluator.

Development only: onnx==1.16.2, numpy==1.26.4, protobuf==4.25.3.
Ordinary Go tests only read arithmetic.json and need no Python installation.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()


def case(name, left_shape, left_bits, right_shape, right_bits):
    fixtures = []
    left = np.asarray(left_bits, dtype=np.uint32).view(np.float32).reshape(left_shape)
    right = np.asarray(right_bits, dtype=np.uint32).view(np.float32).reshape(right_shape)
    for operation in ["Add", "Sub", "Mul", "Div"]:
        node = helper.make_node(operation, ["left", "right"], ["output"])
        graph = helper.make_graph(
            [node],
            name,
            [
                helper.make_tensor_value_info("left", TensorProto.FLOAT, left_shape),
                helper.make_tensor_value_info("right", TensorProto.FLOAT, right_shape),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(
            graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)]
        )
        with np.errstate(all="ignore"):
            output = ReferenceEvaluator(model).run(None, {"left": left, "right": right})[0]
        fixtures.append(
            {
                "name": f"{name}/{operation}",
                "operation": operation,
                "left_shape": left_shape,
                "left_bits": left_bits,
                "right_shape": right_shape,
                "right_bits": right_bits,
                "output_shape": list(output.shape),
                "output_bits": output.reshape(-1).view(np.uint32).tolist(),
            }
        )
    return fixtures


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__

fixtures = []
fixtures += case("channel bias", [2, 2, 1, 2], bits([1, 2, 3, 4, 5, 6, 7, 8]), [1, 2, 1, 1], bits([10, 20]))
fixtures += case("residual", [2, 3], bits([1, -2, 3, 4.5, -5.5, 0]), [2, 3], bits([6, 7, -8, 9, 1, -2]))
fixtures += case("multi-axis broadcast", [2, 1, 3], bits([1, 2, 3, 4, 5, 6]), [1, 2, 1], bits([2, -4]))
fixtures += case("left scalar", [], bits([-2]), [2, 2], bits([1, -2, 0, 4]))
fixtures += case("right scalar", [2, 1, 2], bits([0, -0.0, 3, -4]), [], bits([2]))
fixtures += case("scalar result", [], bits([3.5]), [], bits([2]))
fixtures += case("empty leading axis", [0, 3], [], [1, 3], bits([1, 2, 3]))
fixtures += case("empty middle axis", [2, 0, 3], [], [3], bits([1, 2, 3]))

# Deliberate literal IEEE 754 pairs exercise signed zero, infinities, NaNs, and
# SIMD-sized matching shapes without depending on Python arithmetic for inputs.
left_special = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x3F800000]
right_special = [0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x7F800000, 0x3F800000, 0x40000000, 0x7FC05678]
fixtures += case("special matching", [4, 8], left_special * 4, [4, 8], right_special * 4)
fixtures += case("special broadcast", [2, 1], [0x7F800000, 0x80000000], [1, 4], [0x00000000, 0x80000000, 0x7F800000, 0x7FC01234])

output_path = Path(__file__).with_name("arithmetic.json")
output_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases to {output_path}")
