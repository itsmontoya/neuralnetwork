"""Generate NN-09 fixtures with pinned ONNX 1.16.2 and NumPy 1.26.4.

Development only; ordinary Go tests read convolution.json offline.
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


def sequence(shape, start=0, divisor=4):
    count = int(np.prod(shape, dtype=np.int64))
    return (((np.arange(count, dtype=np.float32) + start) % 13 - 6)
            / np.float32(divisor)).reshape(shape)


def numpy_convolution(x, w, bias, strides, dilations, pads, groups):
    # Independent vectorized padded-window contraction, not the library loop.
    kh, kw = w.shape[2:]
    eh, ew = (kh - 1) * dilations[0] + 1, (kw - 1) * dilations[1] + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    oh = max(0, (padded.shape[2] - eh) // strides[0] + 1)
    ow = max(0, (padded.shape[3] - ew) // strides[1] + 1)
    output = np.zeros((x.shape[0], w.shape[0], oh, ow), dtype=np.float32)
    if output.size == 0:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(padded, (eh, ew), axis=(2, 3))
    windows = windows[:, :, ::strides[0], ::strides[1], ::dilations[0], ::dilations[1]]
    cg, mg = x.shape[1] // groups, w.shape[0] // groups
    for group in range(groups):
        output[:, group * mg:(group + 1) * mg] = np.einsum(
            "nchwkl,mckl->nmhw", windows[:, group * cg:(group + 1) * cg],
            w[group * mg:(group + 1) * mg], optimize=False,
        )
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)
    return output


def evaluate(name, input_shape, weight_shape, strides=(1, 1), dilations=(1, 1),
             pads=(0, 0, 0, 0), groups=1, has_bias=True, constant_weights=False,
             divisor=4):
    x, w = sequence(input_shape, 1, divisor), sequence(weight_shape, 4, divisor)
    bias = sequence([weight_shape[0]], 2) if has_bias else None
    feeds = {"input": x}
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(x.shape))]
    initializers = []
    if constant_weights:
        initializers.append(numpy_helper.from_array(w, "weights"))
    else:
        feeds["weights"] = w
        inputs.append(helper.make_tensor_value_info("weights", TensorProto.FLOAT, list(w.shape)))
    names = ["input", "weights"]
    if bias is not None:
        names.append("bias")
        feeds["bias"] = bias
        inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(bias.shape)))
    graph = helper.make_graph(
        [helper.make_node("Conv", names, ["output"], strides=list(strides),
                          dilations=list(dilations), pads=list(pads), group=groups)],
        name, inputs, [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    output = numpy_convolution(x, w, bias, strides, dilations, pads, groups)
    reference, reference_error, reference_shape = "onnx", "", []
    try:
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        reference_shape = list(actual.shape)
        if actual.shape != output.shape:
            reference = "numpy"
        else:
            np.testing.assert_allclose(actual, output, rtol=1e-5, atol=1e-5)
    except (ValueError, RuntimeError, IndexError) as error:
        reference, reference_error = "numpy", str(error)
    return {
        "name": name, "input_shape": list(x.shape), "input_bits": bits(x),
        "weights_shape": list(w.shape), "weights_bits": bits(w),
        "has_bias": has_bias, "bias_bits": bits(bias) if has_bias else [],
        "strides": list(strides), "dilations": list(dilations), "pads": list(pads),
        "groups": groups, "constant_weights": constant_weights,
        "output_shape": list(output.shape), "output_bits": bits(output),
        "reference": reference, "reference_error": reference_error,
        "reference_shape": reference_shape,
    }


assert onnx.__version__ == "1.16.2", onnx.__version__
assert np.__version__ == "1.26.4", np.__version__
assert google.protobuf.__version__ == "4.25.3", google.protobuf.__version__

cases = [
    ("pointwise batches", [2, 3, 2, 3], [4, 3, 1, 1], {}),
    ("rectangular valid", [1, 2, 4, 5], [3, 2, 2, 3], {}),
    ("symmetric padding", [2, 2, 3, 4], [3, 2, 3, 3], {"pads": [1, 1, 1, 1]}),
    ("asymmetric padding", [1, 2, 3, 4], [2, 2, 2, 3], {"pads": [0, 2, 1, 0]}),
    ("unequal strides", [2, 2, 5, 6], [3, 2, 2, 3], {"strides": [2, 1]}),
    ("horizontal stride", [1, 1, 4, 7], [2, 1, 2, 2], {"strides": [1, 3]}),
    ("dilated height", [1, 2, 5, 4], [2, 2, 2, 2], {"dilations": [2, 1]}),
    ("dilated width", [1, 2, 4, 7], [2, 2, 2, 3], {"dilations": [1, 2]}),
    ("dilated both padded", [2, 1, 3, 4], [2, 1, 2, 2],
     {"dilations": [2, 3], "pads": [1, 2, 0, 1], "strides": [2, 1]}),
    ("grouped batches", [2, 4, 5, 4], [6, 2, 2, 3], {"groups": 2}),
    ("grouped dilated padded", [1, 4, 4, 5], [4, 2, 2, 2],
     {"groups": 2, "dilations": [2, 1], "pads": [1, 0, 2, 1], "strides": [1, 2]}),
    ("depthwise", [2, 3, 4, 4], [3, 1, 3, 3], {"groups": 3}),
    ("depthwise multiplier", [1, 3, 4, 5], [6, 1, 2, 3], {"groups": 3, "pads": [1, 0, 0, 1]}),
    ("whole input kernel", [1, 2, 3, 4], [2, 2, 3, 4], {}),
    ("stride beyond extent", [1, 1, 2, 2], [1, 1, 1, 1], {"strides": [4, 3]}),
    ("fractional accumulation", [1, 3, 4, 4], [2, 3, 3, 3], {"divisor": 7}),
    ("empty batch", [0, 2, 3, 4], [3, 2, 2, 2], {}),
    ("empty grouped batch", [0, 4, 3, 4], [6, 2, 2, 2], {"groups": 2}),
    ("empty height", [1, 2, 0, 3], [2, 2, 1, 1], {}),
    ("padded empty height", [1, 2, 0, 3], [2, 2, 1, 2], {"pads": [1, 0, 1, 0]}),
    ("padded empty spatial", [2, 1, 0, 0], [2, 1, 1, 1], {"pads": [1, 1, 1, 1]}),
    ("kernel larger than input", [1, 1, 1, 2], [1, 1, 3, 2], {}),
    ("negative fractional geometry", [1, 1, 2, 2], [1, 1, 3, 3], {"strides": [2, 2]}),
]
fixtures = []
for position, (name, input_shape, weight_shape, attributes) in enumerate(cases):
    for has_bias in [False, True]:
        fixtures.append(evaluate(name + (" bias" if has_bias else " no bias"), input_shape,
                                 weight_shape, has_bias=has_bias,
                                 constant_weights=position % 2 == 0, **attributes))

path = Path(__file__).with_name("convolution.json")
path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(fixtures)} cases; {sum(f['reference'] != 'onnx' for f in fixtures)} explicit NumPy references")
