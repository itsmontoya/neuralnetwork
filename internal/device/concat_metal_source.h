#ifndef NN_INTERNAL_DEVICE_CONCAT_METAL_SOURCE_H
#define NN_INTERNAL_DEVICE_CONCAT_METAL_SOURCE_H

// Strided tensor concatenation kernels for 4-byte (Float32) and 2-byte
// (Float16/BFloat16) elements. Copies raw bits without float conversion.
// Compiled lazily with fastMathEnabled = NO.
static const char *nnMetalTensorConcatSource =
	"struct NNMetalConcatParams {\n"
	"    uint outer;\n"
	"    uint srcAxisLen;\n"
	"    uint dstAxisLen;\n"
	"    uint dstAxisOffset;\n"
	"    uint inner;\n"
	"};\n"
	"kernel void nn_tensor_concat_u32(\n"
	"    const device uint *input [[buffer(0)]],\n"
	"    device uint *result [[buffer(1)]],\n"
	"    constant NNMetalConcatParams &params [[buffer(2)]],\n"
	"    uint3 gid [[thread_position_in_grid]]\n"
	") {\n"
	"    if (gid.x >= params.inner || gid.y >= params.srcAxisLen || gid.z >= params.outer) {\n"
	"        return;\n"
	"    }\n"
	"    ulong srcIdx = (ulong(gid.z) * ulong(params.srcAxisLen) + ulong(gid.y)) * ulong(params.inner) + ulong(gid.x);\n"
	"    ulong dstIdx = (ulong(gid.z) * ulong(params.dstAxisLen) + ulong(params.dstAxisOffset + gid.y)) * ulong(params.inner) + ulong(gid.x);\n"
	"    result[dstIdx] = input[srcIdx];\n"
	"}\n"
	"kernel void nn_tensor_concat_u16(\n"
	"    const device ushort *input [[buffer(0)]],\n"
	"    device ushort *result [[buffer(1)]],\n"
	"    constant NNMetalConcatParams &params [[buffer(2)]],\n"
	"    uint3 gid [[thread_position_in_grid]]\n"
	") {\n"
	"    if (gid.x >= params.inner || gid.y >= params.srcAxisLen || gid.z >= params.outer) {\n"
	"        return;\n"
	"    }\n"
	"    ulong srcIdx = (ulong(gid.z) * ulong(params.srcAxisLen) + ulong(gid.y)) * ulong(params.inner) + ulong(gid.x);\n"
	"    ulong dstIdx = (ulong(gid.z) * ulong(params.dstAxisLen) + ulong(params.dstAxisOffset + gid.y)) * ulong(params.inner) + ulong(gid.x);\n"
	"    result[dstIdx] = input[srcIdx];\n"
	"}\n";

#endif
