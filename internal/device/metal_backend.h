#ifndef NN_INTERNAL_DEVICE_METAL_BACKEND_H
#define NN_INTERNAL_DEVICE_METAL_BACKEND_H

#include <stdint.h>

enum {
	NNMetalStatusError = -1,
	NNMetalStatusUnavailable = 0,
	NNMetalStatusSuccess = 1,
};

typedef void *NNMetalBuffer;
typedef void *NNMetalScope;

typedef struct {
	uint32_t leftRows;
	uint32_t leftCols;
	uint32_t rightRows;
	uint32_t rightCols;
	uint32_t resultRows;
	uint32_t resultCols;
	uint32_t variant;
} NNMetalMatMulDimensions;

typedef struct {
	uint64_t liveBuffers;
	uint64_t liveBufferBytes;
	uint64_t peakBuffers;
	uint64_t peakBufferBytes;
	uint64_t liveScopes;
	uint64_t peakScopes;
	uint64_t createdBuffers;
	uint64_t releasedBuffers;
	uint64_t createdScopes;
	uint64_t releasedScopes;
	uint64_t submittedCommands;
	uint64_t completedCommands;
} NNMetalResourceSnapshot;

int nn_metal_runtime_available(void);
uint64_t nn_metal_max_buffer_length(void);
NNMetalBuffer nn_metal_buffer_new(uint64_t bytes);
int nn_metal_buffer_upload(NNMetalBuffer buffer, const float *values, uint64_t count);
int nn_metal_buffer_download(NNMetalBuffer buffer, float *values, uint64_t count);
int nn_metal_buffer_upload_bytes(NNMetalBuffer buffer, const void *values, uint64_t bytes);
int nn_metal_buffer_download_bytes(NNMetalBuffer buffer, void *values, uint64_t bytes);
void nn_metal_buffer_release(NNMetalBuffer buffer);
NNMetalScope nn_metal_scope_new(void);
int nn_metal_scope_encode_copy(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t bytes
);
int nn_metal_scope_encode_gather_rows(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t rowBytes,
	const int64_t *rows,
	uint64_t count
);
int nn_metal_scope_encode_gather_rows_mixed(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t columns,
	const int64_t *rows,
	uint64_t count,
	uint32_t sourceDType
);
int nn_metal_scope_encode_fill(
	NNMetalScope scope,
	NNMetalBuffer buffer,
	float value,
	uint64_t count
);
int nn_metal_scope_encode_tensor_elementwise(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	uint32_t count,
	uint32_t operation
);
int nn_metal_scope_encode_tensor_softmax(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t outer,
	uint32_t axisSize,
	uint32_t inner
);
int nn_metal_scope_encode_tensor_layernorm(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer weight,
	NNMetalBuffer bias,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t width,
	float epsilon,
	uint32_t weightDType,
	uint32_t biasDType
);
int nn_metal_scope_encode_tensor_batchnorm(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer scale,
	NNMetalBuffer bias,
	NNMetalBuffer mean,
	NNMetalBuffer variance,
	NNMetalBuffer result,
	uint32_t batch,
	uint32_t channels,
	uint32_t spatial,
	float epsilon
);
typedef struct {
	uint32_t channelsTotal;
	uint32_t inH;
	uint32_t inW;
	uint32_t outH;
	uint32_t outW;
	uint32_t kernelH;
	uint32_t kernelW;
	uint32_t strideH;
	uint32_t strideW;
	uint32_t padTop;
	uint32_t padLeft;
	uint32_t dilationH;
	uint32_t dilationW;
	uint32_t countIncludePad;
} NNMetalPool2DParams;
int nn_metal_scope_encode_tensor_pool2d(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	NNMetalPool2DParams params,
	uint32_t average
);
typedef struct {
	uint32_t rank;
	uint32_t count;
	uint32_t dims[6];
	uint32_t outStrides[6];
	uint32_t inStrides[6];
} NNMetalTransposeParams;
int nn_metal_scope_encode_tensor_transpose(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	NNMetalTransposeParams params,
	uint32_t elemBytes
);
typedef struct {
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t totalBatches;
	uint32_t batchRank;
	uint32_t rightTransposed;
	uint32_t outBatchStrides[4];
	uint32_t leftBatchStrides[4];
	uint32_t rightBatchStrides[4];
} NNMetalBatchedMatMulParams;
int nn_metal_scope_encode_matmul_left_transpose(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	uint32_t m,
	uint32_t n,
	uint32_t k,
	uint32_t rightDType
);
int nn_metal_scope_encode_batched_matmul(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalBatchedMatMulParams params,
	uint32_t rightDType
);
typedef struct {
	uint32_t op;
	uint32_t rank;
	uint32_t count;
	uint32_t mode;
	uint32_t dims[6];
	uint32_t outStrides[6];
	uint32_t leftStrides[6];
	uint32_t rightStrides[6];
} NNMetalBroadcastParams;
int nn_metal_scope_encode_tensor_scale(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	float scale,
	uint32_t count
);
int nn_metal_scope_encode_tensor_broadcast(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalBroadcastParams params
);
int nn_metal_scope_encode_kv_append(
	NNMetalScope scope, NNMetalBuffer keys, NNMetalBuffer values,
	NNMetalBuffer target, uint32_t plane, uint32_t prefix
);
int nn_metal_scope_encode_kv_append_f16(
	NNMetalScope scope, NNMetalBuffer keys, NNMetalBuffer values,
	NNMetalBuffer target, uint32_t plane, uint32_t prefix
);
int nn_metal_scope_encode_attention_math(
	NNMetalScope scope, uint32_t operation, NNMetalBuffer first, NNMetalBuffer second,
	NNMetalBuffer target, NNMetalBuffer extra, const uint32_t params[6], uint32_t threads
);
int nn_metal_scope_encode_add_row_vector(
	NNMetalScope scope,
	NNMetalBuffer values,
	NNMetalBuffer rowVector,
	uint32_t rows,
	uint32_t cols
);
int nn_metal_scope_encode_add_scaled(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	float scale,
	uint32_t count
);
int nn_metal_scope_encode_relu(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t count
);
int nn_metal_scope_encode_relu_backward(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer outputGradient,
	NNMetalBuffer result,
	uint32_t count
);
int nn_metal_scope_encode_softmax_rows(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols
);
int nn_metal_scope_encode_softmax_rows_backward(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer outputGradient,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols
);
int nn_metal_scope_encode_column_sums(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	uint32_t accumulate
);
int nn_metal_scope_encode_categorical_cross_entropy(
	NNMetalScope scope,
	NNMetalBuffer predictions,
	NNMetalBuffer targets,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	float epsilon
);
int nn_metal_scope_encode_categorical_cross_entropy_gradient(
	NNMetalScope scope,
	NNMetalBuffer predictions,
	NNMetalBuffer targets,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	float epsilon
);
enum {
	NNMetalWeightFloat16 = 1,
	NNMetalWeightBFloat16 = 2,
	NNMetalWeightQ8_0 = 3,
	NNMetalWeightQ4_0 = 4,
	NNMetalWeightQ4_1 = 5,
};

int nn_metal_scope_encode_matmul(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions,
	int tensorPipeline
);
int nn_metal_scope_encode_matmul_mixed(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions,
	uint32_t weightDType
);
int nn_metal_scope_encode_matmul_quant(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions,
	uint32_t weightDType
);
int nn_metal_scope_commit(NNMetalScope scope);
int nn_metal_scope_completed(NNMetalScope scope);
int nn_metal_scope_wait(NNMetalScope scope);
void nn_metal_scope_release(NNMetalScope scope);
void nn_metal_resource_snapshot(NNMetalResourceSnapshot *snapshot);
int nn_metal_resource_reset(void);
const char *nn_metal_last_error(void);

int nn_metal_test_missing_kernel(const char *name);
int nn_metal_test_compile_source(const char *source);
int nn_metal_test_scope_fail(NNMetalScope scope);

#endif
