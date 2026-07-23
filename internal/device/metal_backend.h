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
NNMetalBuffer nn_metal_buffer_new(uint64_t bytes);
int nn_metal_buffer_upload(NNMetalBuffer buffer, const float *values, uint64_t count);
int nn_metal_buffer_download(NNMetalBuffer buffer, float *values, uint64_t count);
void nn_metal_buffer_release(NNMetalBuffer buffer);
NNMetalScope nn_metal_scope_new(void);
int nn_metal_scope_encode_copy(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t bytes
);
int nn_metal_scope_encode_fill(
	NNMetalScope scope,
	NNMetalBuffer buffer,
	float value,
	uint64_t count
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
int nn_metal_scope_encode_matmul(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions
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
