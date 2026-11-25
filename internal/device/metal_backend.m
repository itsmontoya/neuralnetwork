//go:build darwin && cgo && metal && !purego

#include "metal_backend.h"
#include "attention_metal_source.h"
#include "matmul_mixed_metal_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	id<MTLBuffer> value;
	uint64_t bytes;
} NNMetalBufferRecord;

typedef struct {
	id<MTLCommandBuffer> commandBuffer;
	NSMutableArray *retainedResources;
	int committed;
	int completionRecorded;
	int forcedFailure;
} NNMetalScopeRecord;

static const char *nnMetalShaderHeader =
	"#include <metal_stdlib>\n"
	"using namespace metal;\n";

typedef enum {
	NNMetalTensorAdd,
	NNMetalTensorMul,
	NNMetalTensorSiLU,
	NNMetalTensorSwiGLU,
	NNMetalTensorElementwiseCount,
} NNMetalTensorElementwiseOperation;

static const char *nnMetalTensorElementwiseSource =
	"static float nn_silu(float x) {\n"
	"    float exponent = exp(-fabs(x));\n"
	"    float sigmoid = (x < 0.0f ? exponent : 1.0f) / (1.0f + exponent);\n"
	"    return x * sigmoid;\n"
	"}\n"
	"kernel void nn_tensor_add(\n"
	"    const device float *left [[buffer(0)]],\n"
	"    const device float *right [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    result[index] = left[index] + right[index];\n"
	"}\n"
	"kernel void nn_tensor_mul(\n"
	"    const device float *left [[buffer(0)]],\n"
	"    const device float *right [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    result[index] = left[index] * right[index];\n"
	"}\n"
	"kernel void nn_tensor_silu(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    result[index] = nn_silu(input[index]);\n"
	"}\n"
	"kernel void nn_tensor_swiglu(\n"
	"    const device float *gate [[buffer(0)]],\n"
	"    const device float *up [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    result[index] = nn_silu(gate[index]) * up[index];\n"
	"}\n";

static const char *nnMetalMatMulDimensionsSource =
	"struct NNMetalMatMulDimensions {\n"
	"    uint leftRows;\n"
	"    uint leftCols;\n"
	"    uint rightRows;\n"
	"    uint rightCols;\n"
	"    uint resultRows;\n"
	"    uint resultCols;\n"
	"    uint variant;\n"
	"};\n";

static const char *nnMetalShaderSource =
	"kernel void nn_fill(\n"
	"    device float *values [[buffer(0)]],\n"
	"    constant float &value [[buffer(1)]],\n"
	"    constant uint &count [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index < count) {\n"
	"        values[index] = value;\n"
	"    }\n"
	"}\n"
	"kernel void nn_add_row_vector(\n"
	"    device float *values [[buffer(0)]],\n"
	"    const device float *rowVector [[buffer(1)]],\n"
	"    constant uint &count [[buffer(2)]],\n"
	"    constant uint &cols [[buffer(3)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index < count) {\n"
	"        values[index] += rowVector[index % cols];\n"
	"    }\n"
	"}\n"
	"kernel void nn_add_scaled(\n"
	"    const device float *left [[buffer(0)]],\n"
	"    const device float *right [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant float &scale [[buffer(3)]],\n"
	"    constant uint &count [[buffer(4)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index < count) {\n"
	"        result[index] = left[index] + scale * right[index];\n"
	"    }\n"
	"}\n"
	"kernel void nn_relu(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    device float *result [[buffer(1)]],\n"
	"    constant uint &count [[buffer(2)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index < count) {\n"
	"        float value = input[index];\n"
	"        result[index] = value > 0.0f ? value : 0.0f;\n"
	"    }\n"
	"}\n"
	"kernel void nn_relu_backward(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    const device float *outputGradient [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant uint &count [[buffer(3)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index < count) {\n"
	"        float derivative = input[index] > 0.0f ? 1.0f : 0.0f;\n"
	"        result[index] = derivative * outputGradient[index];\n"
	"    }\n"
	"}\n"
	"kernel void nn_softmax_rows(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    device float *result [[buffer(1)]],\n"
	"    constant uint &rows [[buffer(2)]],\n"
	"    constant uint &cols [[buffer(3)]],\n"
	"    uint row [[thread_position_in_grid]]\n"
	") {\n"
	"    if (row >= rows) {\n"
	"        return;\n"
	"    }\n"
	"    uint offset = row * cols;\n"
	"    float maxValue = input[offset];\n"
	"    for (uint col = 1; col < cols; col++) {\n"
	"        float value = input[offset + col];\n"
	"        if (value > maxValue) {\n"
	"            maxValue = value;\n"
	"        }\n"
	"    }\n"
	"    float sum = 0.0f;\n"
	"    for (uint col = 0; col < cols; col++) {\n"
	"        float value = exp(input[offset + col] - maxValue);\n"
	"        result[offset + col] = value;\n"
	"        sum += value;\n"
	"    }\n"
	"    for (uint col = 0; col < cols; col++) {\n"
	"        result[offset + col] /= sum;\n"
	"    }\n"
	"}\n"
	"kernel void nn_softmax_rows_backward(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    const device float *outputGradient [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant uint &rows [[buffer(3)]],\n"
	"    constant uint &cols [[buffer(4)]],\n"
	"    uint row [[thread_position_in_grid]]\n"
	") {\n"
	"    if (row >= rows) {\n"
	"        return;\n"
	"    }\n"
	"    uint offset = row * cols;\n"
	"    float maxValue = input[offset];\n"
	"    for (uint col = 1; col < cols; col++) {\n"
	"        float value = input[offset + col];\n"
	"        if (value > maxValue) {\n"
	"            maxValue = value;\n"
	"        }\n"
	"    }\n"
	"    float sum = 0.0f;\n"
	"    for (uint col = 0; col < cols; col++) {\n"
	"        float probability = exp(input[offset + col] - maxValue);\n"
	"        result[offset + col] = probability;\n"
	"        sum += probability;\n"
	"    }\n"
	"    float dot = 0.0f;\n"
	"    for (uint col = 0; col < cols; col++) {\n"
	"        float probability = result[offset + col] / sum;\n"
	"        result[offset + col] = probability;\n"
	"        dot += outputGradient[offset + col] * probability;\n"
	"    }\n"
	"    for (uint col = 0; col < cols; col++) {\n"
	"        result[offset + col] *= outputGradient[offset + col] - dot;\n"
	"    }\n"
	"}\n"
	"kernel void nn_column_sums(\n"
	"    const device float *input [[buffer(0)]],\n"
	"    device float *result [[buffer(1)]],\n"
	"    constant uint &rows [[buffer(2)]],\n"
	"    constant uint &cols [[buffer(3)]],\n"
	"    constant uint &accumulate [[buffer(4)]],\n"
	"    uint col [[thread_position_in_grid]]\n"
	") {\n"
	"    if (col >= cols) {\n"
	"        return;\n"
	"    }\n"
	"    float sum = 0.0f;\n"
	"    for (uint row = 0; row < rows; row++) {\n"
	"        sum += input[row * cols + col];\n"
	"    }\n"
	"    result[col] = accumulate != 0 ? result[col] + sum : sum;\n"
	"}\n"
	"kernel void nn_categorical_cross_entropy(\n"
	"    const device float *predictions [[buffer(0)]],\n"
	"    const device float *targets [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant uint &rows [[buffer(3)]],\n"
	"    constant uint &cols [[buffer(4)]],\n"
	"    constant float &epsilon [[buffer(5)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index != 0) {\n"
	"        return;\n"
	"    }\n"
	"    result[0] = 0.0f;\n"
	"    result[1] = as_type<float>(0u);\n"
	"    result[2] = as_type<float>(0u);\n"
	"    result[3] = as_type<float>(0u);\n"
	"    result[4] = 0.0f;\n"
	"    float loss = 0.0f;\n"
	"    for (uint row = 0; row < rows; row++) {\n"
	"        uint offset = row * cols;\n"
	"        uint ones = 0;\n"
	"        for (uint col = 0; col < cols; col++) {\n"
	"            float target = targets[offset + col];\n"
	"            if (target == 1.0f) {\n"
	"                ones++;\n"
	"                float prediction = predictions[offset + col];\n"
	"                if (prediction < epsilon) {\n"
	"                    prediction = epsilon;\n"
	"                } else if (prediction > 1.0f - epsilon) {\n"
	"                    prediction = 1.0f - epsilon;\n"
	"                }\n"
	"                loss -= log(prediction);\n"
	"            } else if (target != 0.0f) {\n"
	"                result[1] = as_type<float>(1u);\n"
	"                result[2] = as_type<float>(row);\n"
	"                result[3] = as_type<float>(col);\n"
	"                result[4] = target;\n"
	"                return;\n"
	"            }\n"
	"        }\n"
	"        if (ones != 1) {\n"
	"            result[1] = as_type<float>(2u);\n"
	"            result[2] = as_type<float>(row);\n"
	"            result[3] = as_type<float>(ones);\n"
	"            return;\n"
	"        }\n"
	"    }\n"
	"    result[0] = loss / float(rows);\n"
	"}\n"
	"kernel void nn_categorical_cross_entropy_gradient(\n"
	"    const device float *predictions [[buffer(0)]],\n"
	"    const device float *targets [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant uint &count [[buffer(3)]],\n"
	"    constant uint &rows [[buffer(4)]],\n"
	"    constant float &epsilon [[buffer(5)]],\n"
	"    uint index [[thread_position_in_grid]]\n"
	") {\n"
	"    if (index >= count) {\n"
	"        return;\n"
	"    }\n"
	"    float target = targets[index];\n"
	"    if (target == 0.0f) {\n"
	"        result[index] = 0.0f;\n"
	"        return;\n"
	"    }\n"
	"    float prediction = predictions[index];\n"
	"    if (prediction < epsilon) {\n"
	"        prediction = epsilon;\n"
	"    } else if (prediction > 1.0f - epsilon) {\n"
	"        prediction = 1.0f - epsilon;\n"
	"    }\n"
	"    result[index] = -target / prediction / float(rows);\n"
	"}\n";

static const char *nnMetalMatMulSource =
	"kernel void nn_matmul(\n"
	"    const device float *left [[buffer(0)]],\n"
	"    const device float *right [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant NNMetalMatMulDimensions &params [[buffer(3)]],\n"
	"    uint2 gid [[thread_position_in_grid]]\n"
	") {\n"
	"    uint col = gid.x;\n"
	"    uint row = gid.y;\n"
	"    if (row >= params.resultRows || col >= params.resultCols) {\n"
	"        return;\n"
	"    }\n"
	"    float sum = 0.0;\n"
	"    if (params.variant == 0) {\n"
	"        for (uint inner = 0; inner < params.leftCols; inner++) {\n"
	"            sum += left[row * params.leftCols + inner] * right[inner * params.rightCols + col];\n"
	"        }\n"
	"    } else if (params.variant == 1) {\n"
	"        for (uint inner = 0; inner < params.leftRows; inner++) {\n"
	"            sum += left[inner * params.leftCols + row] * right[inner * params.rightCols + col];\n"
	"        }\n"
	"    } else {\n"
	"        for (uint inner = 0; inner < params.leftCols; inner++) {\n"
	"            sum += left[row * params.leftCols + inner] * right[col * params.rightCols + inner];\n"
	"        }\n"
	"    }\n"
	"    result[row * params.resultCols + col] = sum;\n"
	"}\n";

enum {
	NNMetalRuntimeUninitialized = 0,
	NNMetalRuntimeReady = 1,
	NNMetalRuntimeUnavailable = 2,
	NNMetalRuntimeFailed = 3,
};

static pthread_mutex_t nnMetalRuntimeMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t nnMetalResourceMutex = PTHREAD_MUTEX_INITIALIZER;
static int nnMetalRuntimeState = NNMetalRuntimeUninitialized;
static char nnMetalRuntimeError[1024];
static _Thread_local char nnMetalLastError[1024];
static id<MTLDevice> nnMetalDevice = nil;
static id<MTLCommandQueue> nnMetalCommandQueue = nil;
static id<MTLLibrary> nnMetalLibrary = nil;
static id<MTLComputePipelineState> nnMetalFillPipeline = nil;
static id<MTLComputePipelineState> nnMetalAddRowVectorPipeline = nil;
static id<MTLComputePipelineState> nnMetalAddScaledPipeline = nil;
static id<MTLComputePipelineState> nnMetalReLUPipeline = nil;
static id<MTLComputePipelineState> nnMetalReLUBackwardPipeline = nil;
static id<MTLComputePipelineState> nnMetalSoftmaxRowsPipeline = nil;
static id<MTLComputePipelineState> nnMetalSoftmaxRowsBackwardPipeline = nil;
static id<MTLComputePipelineState> nnMetalColumnSumsPipeline = nil;
static id<MTLComputePipelineState> nnMetalCategoricalCrossEntropyPipeline = nil;
static id<MTLComputePipelineState> nnMetalCategoricalCrossEntropyGradientPipeline = nil;
static id<MTLComputePipelineState> nnMetalMatMulPipeline = nil;
static pthread_once_t nnMetalTensorMatMulOnce = PTHREAD_ONCE_INIT;
static id<MTLComputePipelineState> nnMetalTensorMatMulPipeline = nil;
static char nnMetalTensorMatMulError[1024];
typedef enum {
	NNMetalMatMulMixedF16,
	NNMetalMatMulMixedF16RowRT,
	NNMetalMatMulMixedBF16,
	NNMetalMatMulMixedBF16RowRT,
	NNMetalMatMulMixedCount,
} NNMetalMatMulMixedPipeline;
static pthread_once_t nnMetalTensorMatMulMixedOnce = PTHREAD_ONCE_INIT;
static id<MTLComputePipelineState> nnMetalTensorMatMulMixedPipelines[NNMetalMatMulMixedCount];
static char nnMetalTensorMatMulMixedError[1024];
static pthread_once_t nnMetalTensorElementwiseOnce = PTHREAD_ONCE_INIT;
static id<MTLComputePipelineState> nnMetalTensorElementwisePipelines[NNMetalTensorElementwiseCount];
static char nnMetalTensorElementwiseError[1024];
typedef enum {
	NNMetalTensorRMSNorm,
	NNMetalTensorRotatePairs,
	NNMetalTensorCausalAttention,
	NNMetalAttentionMathCount,
} NNMetalAttentionMathOperation;

static pthread_once_t nnMetalAttentionMathOnce = PTHREAD_ONCE_INIT;
static id<MTLComputePipelineState> nnMetalAttentionMathPipelines[NNMetalAttentionMathCount];
static char nnMetalAttentionMathError[1024];
static NNMetalResourceSnapshot nnMetalResources;

static void nn_metal_set_error(const char *message) {
	if (message == NULL || message[0] == '\0') {
		message = "metal: unknown error";
	}

	strncpy(nnMetalLastError, message, sizeof(nnMetalLastError) - 1);
	nnMetalLastError[sizeof(nnMetalLastError) - 1] = '\0';
}

static void nn_metal_set_error_ns(NSString *message) {
	const char *text = NULL;

	if (message != nil) {
		text = [message UTF8String];
	}

	nn_metal_set_error(text);
}

static void nn_metal_cache_runtime_error(void) {
	strncpy(nnMetalRuntimeError, nnMetalLastError, sizeof(nnMetalRuntimeError) - 1);
	nnMetalRuntimeError[sizeof(nnMetalRuntimeError) - 1] = '\0';
}

static void nn_metal_release_runtime_resources(void) {
	[nnMetalMatMulPipeline release];
	[nnMetalCategoricalCrossEntropyGradientPipeline release];
	[nnMetalCategoricalCrossEntropyPipeline release];
	[nnMetalColumnSumsPipeline release];
	[nnMetalSoftmaxRowsBackwardPipeline release];
	[nnMetalSoftmaxRowsPipeline release];
	[nnMetalReLUBackwardPipeline release];
	[nnMetalReLUPipeline release];
	[nnMetalAddScaledPipeline release];
	[nnMetalAddRowVectorPipeline release];
	[nnMetalFillPipeline release];
	[nnMetalLibrary release];
	[nnMetalCommandQueue release];
	[nnMetalDevice release];
	nnMetalMatMulPipeline = nil;
	nnMetalCategoricalCrossEntropyGradientPipeline = nil;
	nnMetalCategoricalCrossEntropyPipeline = nil;
	nnMetalColumnSumsPipeline = nil;
	nnMetalSoftmaxRowsBackwardPipeline = nil;
	nnMetalSoftmaxRowsPipeline = nil;
	nnMetalReLUBackwardPipeline = nil;
	nnMetalReLUPipeline = nil;
	nnMetalAddScaledPipeline = nil;
	nnMetalAddRowVectorPipeline = nil;
	nnMetalFillPipeline = nil;
	nnMetalLibrary = nil;
	nnMetalCommandQueue = nil;
	nnMetalDevice = nil;
}

static id<MTLComputePipelineState> nn_metal_new_pipeline(id<MTLLibrary> library, NSString *name) {
	NSError *error = nil;
	id<MTLFunction> function = nil;
	id<MTLComputePipelineState> pipeline = nil;

	function = [library newFunctionWithName:name];
	if (function == nil) {
		nn_metal_set_error_ns([NSString stringWithFormat:@"metal: kernel %@ not found", name]);
		return nil;
	}

	pipeline = [nnMetalDevice newComputePipelineStateWithFunction:function error:&error];
	[function release];
	if (pipeline == nil) {
		nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile pipeline %@: %@", name, [error localizedDescription]]);
		return nil;
	}

	return pipeline;
}

// This pipeline shares the runtime owner and is compiled only on first tensor matmul.
static void nn_metal_initialize_tensor_matmul(void) {
	@autoreleasepool {
		NSError *error = nil;
		NSString *source = [NSString stringWithFormat:@"%s%s%s",
			nnMetalShaderHeader, nnMetalMatMulDimensionsSource, nnMetalMatMulSource];
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		[options setFastMathEnabled:NO];
		id<MTLLibrary> library = [nnMetalDevice newLibraryWithSource:source options:options error:&error];
		[options release];
		if (library == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile tensor matmul library: %@", [error localizedDescription]]);
		} else {
			nnMetalTensorMatMulPipeline = nn_metal_new_pipeline(library, @"nn_matmul");
			[library release];
		}
		if (nnMetalTensorMatMulPipeline == nil) {
			strlcpy(nnMetalTensorMatMulError, nnMetalLastError, sizeof(nnMetalTensorMatMulError));
		}
	}
}

// Mixed-precision FP16/BF16 matmul pipelines compile lazily with fastMathEnabled = NO.
static void nn_metal_initialize_tensor_matmul_mixed(void) {
	@autoreleasepool {
		NSError *error = nil;
		NSString *source = [NSString stringWithFormat:@"%s%s%s",
			nnMetalShaderHeader, nnMetalMatMulDimensionsSource, nnMetalMatMulMixedSource];
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		[options setFastMathEnabled:NO];
		id<MTLLibrary> library = [nnMetalDevice newLibraryWithSource:source options:options error:&error];
		[options release];
		if (library == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile mixed matmul library: %@", [error localizedDescription]]);
			strlcpy(nnMetalTensorMatMulMixedError, nnMetalLastError, sizeof(nnMetalTensorMatMulMixedError));
			return;
		}

		NSString *names[NNMetalMatMulMixedCount] = {
			[NNMetalMatMulMixedF16] = @"nn_matmul_f16",
			[NNMetalMatMulMixedF16RowRT] = @"nn_matmul_f16_row_right_transpose",
			[NNMetalMatMulMixedBF16] = @"nn_matmul_bf16",
			[NNMetalMatMulMixedBF16RowRT] = @"nn_matmul_bf16_row_right_transpose",
		};
		for (NSUInteger i = 0; i < NNMetalMatMulMixedCount; i++) {
			nnMetalTensorMatMulMixedPipelines[i] = nn_metal_new_pipeline(library, names[i]);
			if (nnMetalTensorMatMulMixedPipelines[i] == nil) {
				strlcpy(nnMetalTensorMatMulMixedError, nnMetalLastError, sizeof(nnMetalTensorMatMulMixedError));
				for (NSUInteger j = 0; j < i; j++) {
					[nnMetalTensorMatMulMixedPipelines[j] release];
					nnMetalTensorMatMulMixedPipelines[j] = nil;
				}
				break;
			}
		}
		[library release];
	}
}

// Tensor elementwise pipelines share the device and compile only on first use.
static void nn_metal_initialize_tensor_elementwise(void) {
	@autoreleasepool {
		NSError *error = nil;
		NSString *source = [NSString stringWithFormat:@"%s%s", nnMetalShaderHeader, nnMetalTensorElementwiseSource];
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		[options setFastMathEnabled:NO];
		id<MTLLibrary> library = [nnMetalDevice newLibraryWithSource:source options:options error:&error];
		[options release];
		if (library == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile tensor elementwise library: %@", [error localizedDescription]]);
			strlcpy(nnMetalTensorElementwiseError, nnMetalLastError, sizeof(nnMetalTensorElementwiseError));
			return;
		}

		NSString *names[NNMetalTensorElementwiseCount] = {
			[NNMetalTensorAdd] = @"nn_tensor_add",
			[NNMetalTensorMul] = @"nn_tensor_mul",
			[NNMetalTensorSiLU] = @"nn_tensor_silu",
			[NNMetalTensorSwiGLU] = @"nn_tensor_swiglu",
		};
		for (NSUInteger i = 0; i < NNMetalTensorElementwiseCount; i++) {
			nnMetalTensorElementwisePipelines[i] = nn_metal_new_pipeline(library, names[i]);
			if (nnMetalTensorElementwisePipelines[i] == nil) {
				strlcpy(nnMetalTensorElementwiseError, nnMetalLastError, sizeof(nnMetalTensorElementwiseError));
				for (NSUInteger j = 0; j < i; j++) {
					[nnMetalTensorElementwisePipelines[j] release];
					nnMetalTensorElementwisePipelines[j] = nil;
				}
				break;
			}
		}
		[library release];
	}
}

// These kernels compile lazily on the existing device, outside ANN startup.
static void nn_metal_initialize_attention_math(void) {
	@autoreleasepool {
		NSError *error = nil;
		NSString *source = [NSString stringWithFormat:@"%s%s", nnMetalShaderHeader, nnMetalAttentionMathSource];
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		[options setFastMathEnabled:NO];
		id<MTLLibrary> library = [nnMetalDevice newLibraryWithSource:source options:options error:&error];
		[options release];
		if (library == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile attention math library: %@", [error localizedDescription]]);
			strlcpy(nnMetalAttentionMathError, nnMetalLastError, sizeof(nnMetalAttentionMathError));
			return;
		}

		NSString *names[NNMetalAttentionMathCount] = {
			[NNMetalTensorRMSNorm] = @"nn_tensor_rms_norm",
			[NNMetalTensorRotatePairs] = @"nn_tensor_rotate_pairs",
			[NNMetalTensorCausalAttention] = @"nn_tensor_causal_attention",
		};
		for (NSUInteger i = 0; i < NNMetalAttentionMathCount; i++) {
			nnMetalAttentionMathPipelines[i] = nn_metal_new_pipeline(library, names[i]);
			if (nnMetalAttentionMathPipelines[i] == nil) {
				strlcpy(nnMetalAttentionMathError, nnMetalLastError, sizeof(nnMetalAttentionMathError));
				for (NSUInteger j = 0; j < i; j++) {
					[nnMetalAttentionMathPipelines[j] release];
					nnMetalAttentionMathPipelines[j] = nil;
				}
				break;
			}
		}
		[library release];
	}
}

static int nn_metal_initialize(void) {
	int status = NNMetalStatusError;

	pthread_mutex_lock(&nnMetalRuntimeMutex);
	if (nnMetalRuntimeState != NNMetalRuntimeUninitialized) {
		if (nnMetalRuntimeState == NNMetalRuntimeReady) {
			status = NNMetalStatusSuccess;
		} else if (nnMetalRuntimeState == NNMetalRuntimeUnavailable) {
			nn_metal_set_error(nnMetalRuntimeError);
			status = NNMetalStatusUnavailable;
		} else {
			nn_metal_set_error(nnMetalRuntimeError);
			status = NNMetalStatusError;
		}
		pthread_mutex_unlock(&nnMetalRuntimeMutex);
		return status;
	}

	@autoreleasepool {
		NSError *error = nil;
		NSString *source = nil;

		nnMetalDevice = MTLCreateSystemDefaultDevice();
		if (nnMetalDevice == nil) {
			nn_metal_set_error("metal: no default device");
			nn_metal_cache_runtime_error();
			nnMetalRuntimeState = NNMetalRuntimeUnavailable;
			status = NNMetalStatusUnavailable;
		} else {
			nnMetalCommandQueue = [nnMetalDevice newCommandQueue];
			if (nnMetalCommandQueue == nil) {
				nn_metal_set_error("metal: create command queue: returned nil");
			}

			if (nnMetalCommandQueue != nil) {
				source = [NSString stringWithFormat:@"%s%s%s%s",
					nnMetalShaderHeader, nnMetalMatMulDimensionsSource, nnMetalShaderSource, nnMetalMatMulSource];
				nnMetalLibrary = [nnMetalDevice newLibraryWithSource:source options:nil error:&error];
				if (nnMetalLibrary == nil) {
					nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile shader library: %@", [error localizedDescription]]);
				}
			}

			if (nnMetalLibrary != nil) {
				nnMetalFillPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_fill");
			}
			if (nnMetalFillPipeline != nil) {
				nnMetalAddRowVectorPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_add_row_vector");
			}
			if (nnMetalAddRowVectorPipeline != nil) {
				nnMetalAddScaledPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_add_scaled");
			}
			if (nnMetalAddScaledPipeline != nil) {
				nnMetalReLUPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_relu");
			}
			if (nnMetalReLUPipeline != nil) {
				nnMetalReLUBackwardPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_relu_backward");
			}
			if (nnMetalReLUBackwardPipeline != nil) {
				nnMetalSoftmaxRowsPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_softmax_rows");
			}
			if (nnMetalSoftmaxRowsPipeline != nil) {
				nnMetalSoftmaxRowsBackwardPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_softmax_rows_backward");
			}
			if (nnMetalSoftmaxRowsBackwardPipeline != nil) {
				nnMetalColumnSumsPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_column_sums");
			}
			if (nnMetalColumnSumsPipeline != nil) {
				nnMetalCategoricalCrossEntropyPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_categorical_cross_entropy");
			}
			if (nnMetalCategoricalCrossEntropyPipeline != nil) {
				nnMetalCategoricalCrossEntropyGradientPipeline =
					nn_metal_new_pipeline(nnMetalLibrary, @"nn_categorical_cross_entropy_gradient");
			}
			if (nnMetalCategoricalCrossEntropyGradientPipeline != nil) {
				nnMetalMatMulPipeline = nn_metal_new_pipeline(nnMetalLibrary, @"nn_matmul");
			}

			if (nnMetalCommandQueue != nil && nnMetalLibrary != nil &&
				nnMetalFillPipeline != nil && nnMetalAddRowVectorPipeline != nil &&
				nnMetalAddScaledPipeline != nil && nnMetalReLUPipeline != nil &&
				nnMetalReLUBackwardPipeline != nil && nnMetalSoftmaxRowsPipeline != nil &&
				nnMetalSoftmaxRowsBackwardPipeline != nil && nnMetalColumnSumsPipeline != nil &&
				nnMetalCategoricalCrossEntropyPipeline != nil &&
				nnMetalCategoricalCrossEntropyGradientPipeline != nil &&
				nnMetalMatMulPipeline != nil) {
				nnMetalRuntimeState = NNMetalRuntimeReady;
				status = NNMetalStatusSuccess;
			} else {
				nn_metal_cache_runtime_error();
				nn_metal_release_runtime_resources();
				nnMetalRuntimeState = NNMetalRuntimeFailed;
				status = NNMetalStatusError;
			}
		}
	}

	pthread_mutex_unlock(&nnMetalRuntimeMutex);
	return status;
}

static NNMetalBufferRecord *nn_metal_buffer_record(NNMetalBuffer buffer) {
	return (NNMetalBufferRecord *)buffer;
}

static NNMetalScopeRecord *nn_metal_scope_record(NNMetalScope scope) {
	return (NNMetalScopeRecord *)scope;
}

static void nn_metal_record_buffer_created(uint64_t bytes) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	nnMetalResources.liveBuffers++;
	nnMetalResources.liveBufferBytes += bytes;
	nnMetalResources.createdBuffers++;
	if (nnMetalResources.liveBuffers > nnMetalResources.peakBuffers) {
		nnMetalResources.peakBuffers = nnMetalResources.liveBuffers;
	}
	if (nnMetalResources.liveBufferBytes > nnMetalResources.peakBufferBytes) {
		nnMetalResources.peakBufferBytes = nnMetalResources.liveBufferBytes;
	}
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static void nn_metal_record_buffer_released(uint64_t bytes) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	if (nnMetalResources.liveBuffers > 0) {
		nnMetalResources.liveBuffers--;
	}
	if (nnMetalResources.liveBufferBytes >= bytes) {
		nnMetalResources.liveBufferBytes -= bytes;
	} else {
		nnMetalResources.liveBufferBytes = 0;
	}
	nnMetalResources.releasedBuffers++;
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static void nn_metal_record_scope_created(void) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	nnMetalResources.liveScopes++;
	nnMetalResources.createdScopes++;
	if (nnMetalResources.liveScopes > nnMetalResources.peakScopes) {
		nnMetalResources.peakScopes = nnMetalResources.liveScopes;
	}
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static void nn_metal_record_scope_released(void) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	if (nnMetalResources.liveScopes > 0) {
		nnMetalResources.liveScopes--;
	}
	nnMetalResources.releasedScopes++;
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static void nn_metal_record_command_submitted(void) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	nnMetalResources.submittedCommands++;
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static void nn_metal_record_command_completed(NNMetalScopeRecord *record) {
	if (record->completionRecorded) {
		return;
	}

	record->completionRecorded = 1;
	pthread_mutex_lock(&nnMetalResourceMutex);
	nnMetalResources.completedCommands++;
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

static int nn_metal_scope_completion_status(NNMetalScopeRecord *record, int wait) {
	NSError *error = nil;
	MTLCommandBufferStatus status;

	if (record == NULL || record->commandBuffer == nil) {
		nn_metal_set_error("metal: inspect command scope: nil handle");
		return NNMetalStatusError;
	}
	if (!record->committed) {
		nn_metal_set_error("metal: inspect command scope: command not committed");
		return NNMetalStatusError;
	}

	if (wait) {
		[record->commandBuffer waitUntilCompleted];
	}
	status = [record->commandBuffer status];
	if (status == MTLCommandBufferStatusCompleted) {
		nn_metal_record_command_completed(record);
		if (record->forcedFailure) {
			nn_metal_set_error("metal: execute command scope: injected failure");
			return NNMetalStatusError;
		}
		return NNMetalStatusSuccess;
	}
	if (!wait && (status == MTLCommandBufferStatusCommitted || status == MTLCommandBufferStatusScheduled)) {
		return NNMetalStatusUnavailable;
	}

	nn_metal_record_command_completed(record);
	error = [record->commandBuffer error];
	if (error != nil) {
		nn_metal_set_error_ns([NSString stringWithFormat:@"metal: execute command scope: %@", [error localizedDescription]]);
	} else {
		nn_metal_set_error("metal: execute command scope: command failed");
	}
	return NNMetalStatusError;
}

int nn_metal_runtime_available(void) {
	return nn_metal_initialize();
}

uint64_t nn_metal_max_buffer_length(void) {
	if (nn_metal_initialize() != NNMetalStatusSuccess) {
		return 0;
	}
	return (uint64_t)[nnMetalDevice maxBufferLength];
}

NNMetalBuffer nn_metal_buffer_new(uint64_t bytes) {
	NNMetalBufferRecord *record = NULL;
	id<MTLBuffer> buffer = nil;

	if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) {
		nn_metal_set_error("metal: allocate buffer: invalid byte length");
		return NULL;
	}
	if (nn_metal_initialize() != NNMetalStatusSuccess) {
		return NULL;
	}
	if (bytes > (uint64_t)[nnMetalDevice maxBufferLength]) {
		nn_metal_set_error("metal: allocate buffer: byte length exceeds device limit");
		return NULL;
	}

	@autoreleasepool {
		buffer = [nnMetalDevice newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
		if (buffer == nil) {
			nn_metal_set_error("metal: allocate buffer: returned nil");
			return NULL;
		}

		record = calloc(1, sizeof(*record));
		if (record == NULL) {
			[buffer release];
			nn_metal_set_error("metal: allocate buffer record: out of memory");
			return NULL;
		}
		record->value = buffer;
		record->bytes = bytes;
	}

	nn_metal_record_buffer_created(bytes);
	return (NNMetalBuffer)record;
}

int nn_metal_buffer_upload(NNMetalBuffer buffer, const float *values, uint64_t count) {
	NNMetalBufferRecord *record = nn_metal_buffer_record(buffer);
	uint64_t bytes = 0;

	if (record == NULL || record->value == nil || values == NULL) {
		nn_metal_set_error("metal: upload buffer: nil handle or values");
		return NNMetalStatusError;
	}
	if (count > UINT64_MAX / sizeof(float)) {
		nn_metal_set_error("metal: upload buffer: byte length overflow");
		return NNMetalStatusError;
	}
	bytes = count * sizeof(float);
	if (bytes != record->bytes) {
		nn_metal_set_error("metal: upload buffer: length mismatch");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		memcpy([record->value contents], values, (size_t)bytes);
	}
	return NNMetalStatusSuccess;
}

int nn_metal_buffer_download(NNMetalBuffer buffer, float *values, uint64_t count) {
	NNMetalBufferRecord *record = nn_metal_buffer_record(buffer);
	uint64_t bytes = 0;

	if (record == NULL || record->value == nil || values == NULL) {
		nn_metal_set_error("metal: download buffer: nil handle or destination");
		return NNMetalStatusError;
	}
	if (count > UINT64_MAX / sizeof(float)) {
		nn_metal_set_error("metal: download buffer: byte length overflow");
		return NNMetalStatusError;
	}
	bytes = count * sizeof(float);
	if (bytes != record->bytes) {
		nn_metal_set_error("metal: download buffer: length mismatch");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		memcpy(values, [record->value contents], (size_t)bytes);
	}
	return NNMetalStatusSuccess;
}

int nn_metal_buffer_upload_bytes(NNMetalBuffer buffer, const void *values, uint64_t bytes) {
	NNMetalBufferRecord *record = nn_metal_buffer_record(buffer);

	if (record == NULL || record->value == nil || values == NULL) {
		nn_metal_set_error("metal: upload buffer bytes: nil handle or values");
		return NNMetalStatusError;
	}
	if (bytes == 0 || bytes != record->bytes) {
		nn_metal_set_error("metal: upload buffer bytes: length mismatch");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		memcpy([record->value contents], values, (size_t)bytes);
	}
	return NNMetalStatusSuccess;
}

int nn_metal_buffer_download_bytes(NNMetalBuffer buffer, void *values, uint64_t bytes) {
	NNMetalBufferRecord *record = nn_metal_buffer_record(buffer);

	if (record == NULL || record->value == nil || values == NULL) {
		nn_metal_set_error("metal: download buffer bytes: nil handle or destination");
		return NNMetalStatusError;
	}
	if (bytes == 0 || bytes != record->bytes) {
		nn_metal_set_error("metal: download buffer bytes: length mismatch");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		memcpy(values, [record->value contents], (size_t)bytes);
	}
	return NNMetalStatusSuccess;
}

void nn_metal_buffer_release(NNMetalBuffer buffer) {
	NNMetalBufferRecord *record = nn_metal_buffer_record(buffer);

	if (record == NULL) {
		return;
	}

	@autoreleasepool {
		[record->value release];
		record->value = nil;
	}
	nn_metal_record_buffer_released(record->bytes);
	free(record);
}

NNMetalScope nn_metal_scope_new(void) {
	NNMetalScopeRecord *record = NULL;
	id<MTLCommandBuffer> commandBuffer = nil;

	if (nn_metal_initialize() != NNMetalStatusSuccess) {
		return NULL;
	}

	@autoreleasepool {
		commandBuffer = [nnMetalCommandQueue commandBuffer];
		if (commandBuffer == nil) {
			nn_metal_set_error("metal: create command scope: returned nil command buffer");
			return NULL;
		}

		record = calloc(1, sizeof(*record));
		if (record == NULL) {
			nn_metal_set_error("metal: allocate command scope record: out of memory");
			return NULL;
		}
		record->commandBuffer = [commandBuffer retain];
		record->retainedResources = [[NSMutableArray alloc] init];
		if (record->retainedResources == nil) {
			[record->commandBuffer release];
			free(record);
			nn_metal_set_error("metal: allocate command scope resources: returned nil");
			return NULL;
		}
	}

	nn_metal_record_scope_created();
	return (NNMetalScope)record;
}

int nn_metal_scope_encode_copy(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t bytes
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *sourceRecord = nn_metal_buffer_record(source);
	NNMetalBufferRecord *destinationRecord = nn_metal_buffer_record(destination);
	id<MTLBlitCommandEncoder> encoder = nil;

	if (scopeRecord == NULL || sourceRecord == NULL || destinationRecord == NULL) {
		nn_metal_set_error("metal: encode copy: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || bytes == 0 || bytes > sourceRecord->bytes || bytes > destinationRecord->bytes) {
		nn_metal_set_error("metal: encode copy: invalid state or byte length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer blitCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode copy: create blit encoder returned nil");
			return NNMetalStatusError;
		}
		[encoder copyFromBuffer:sourceRecord->value sourceOffset:0 toBuffer:destinationRecord->value destinationOffset:0 size:(NSUInteger)bytes];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:sourceRecord->value];
		[scopeRecord->retainedResources addObject:destinationRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_gather_rows(
	NNMetalScope scope,
	NNMetalBuffer source,
	NNMetalBuffer destination,
	uint64_t rowBytes,
	const int64_t *rows,
	uint64_t count
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *sourceRecord = nn_metal_buffer_record(source);
	NNMetalBufferRecord *destinationRecord = nn_metal_buffer_record(destination);

	if (scopeRecord == NULL || sourceRecord == NULL || destinationRecord == NULL || rows == NULL) {
		nn_metal_set_error("metal: encode gather rows: nil handle or indices");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || sourceRecord == destinationRecord || rowBytes == 0 ||
		rowBytes % sizeof(float) != 0 || sourceRecord->bytes % rowBytes != 0 ||
		count == 0 || destinationRecord->bytes % rowBytes != 0 || destinationRecord->bytes / rowBytes != count) {
		nn_metal_set_error("metal: encode gather rows: invalid state or row size");
		return NNMetalStatusError;
	}
	uint64_t sourceRows = sourceRecord->bytes / rowBytes;
	for (uint64_t i = 0; i < count; i++) {
		if (rows[i] < 0 || (uint64_t)rows[i] >= sourceRows) {
			nn_metal_set_error("metal: encode gather rows: row index outside source");
			return NNMetalStatusError;
		}
	}

	@autoreleasepool {
		id<MTLBlitCommandEncoder> encoder = [scopeRecord->commandBuffer blitCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode gather rows: create blit encoder returned nil");
			return NNMetalStatusError;
		}
		for (uint64_t i = 0; i < count; i++) {
			[encoder copyFromBuffer:sourceRecord->value sourceOffset:(NSUInteger)((uint64_t)rows[i] * rowBytes)
				toBuffer:destinationRecord->value destinationOffset:(NSUInteger)(i * rowBytes) size:(NSUInteger)rowBytes];
		}
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:sourceRecord->value];
		[scopeRecord->retainedResources addObject:destinationRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_fill(
	NNMetalScope scope,
	NNMetalBuffer buffer,
	float value,
	uint64_t count
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *bufferRecord = nn_metal_buffer_record(buffer);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint32_t shaderCount = 0;

	if (scopeRecord == NULL || bufferRecord == NULL) {
		nn_metal_set_error("metal: encode fill: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || count == 0 || count > UINT32_MAX || count * sizeof(float) != bufferRecord->bytes) {
		nn_metal_set_error("metal: encode fill: invalid state or element count");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode fill: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		shaderCount = (uint32_t)count;
		threadWidth = MIN((NSUInteger)256, [nnMetalFillPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalFillPipeline];
		[encoder setBuffer:bufferRecord->value offset:0 atIndex:0];
		[encoder setBytes:&value length:sizeof(value) atIndex:1];
		[encoder setBytes:&shaderCount length:sizeof(shaderCount) atIndex:2];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:bufferRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_tensor_elementwise(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	uint32_t count,
	uint32_t operation
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *leftRecord = nn_metal_buffer_record(left);
	NNMetalBufferRecord *rightRecord = nn_metal_buffer_record(right);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	uint64_t bytes = (uint64_t)count * sizeof(float);

	if (scopeRecord == NULL || leftRecord == NULL || resultRecord == NULL ||
		(rightRecord == NULL) != (operation == NNMetalTensorSiLU)) {
		nn_metal_set_error("metal: encode tensor elementwise operation: invalid handles");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || operation >= NNMetalTensorElementwiseCount ||
		count == 0 || bytes != leftRecord->bytes || bytes != resultRecord->bytes ||
		(rightRecord != NULL && bytes != rightRecord->bytes) ||
		leftRecord == resultRecord || rightRecord == resultRecord) {
		nn_metal_set_error("metal: encode tensor elementwise operation: invalid state, operation or buffer length");
		return NNMetalStatusError;
	}

	pthread_once(&nnMetalTensorElementwiseOnce, nn_metal_initialize_tensor_elementwise);
	id<MTLComputePipelineState> pipeline = nnMetalTensorElementwisePipelines[operation];
	if (pipeline == nil) {
		nn_metal_set_error(nnMetalTensorElementwiseError);
		return NNMetalStatusError;
	}

	@autoreleasepool {
		id<MTLComputeCommandEncoder> encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode tensor elementwise operation: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:leftRecord->value offset:0 atIndex:0];
		if (rightRecord != NULL) {
			[encoder setBuffer:rightRecord->value offset:0 atIndex:1];
		}
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		NSUInteger threadWidth = MIN((NSUInteger)256, [pipeline maxTotalThreadsPerThreadgroup]);
		[encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:leftRecord->value];
		if (rightRecord != NULL) {
			[scopeRecord->retainedResources addObject:rightRecord->value];
		}
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_kv_append(
	NNMetalScope scope, NNMetalBuffer history, NNMetalBuffer keys, NNMetalBuffer values,
	NNMetalBuffer target, uint32_t plane, uint32_t prefix
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *previous = nn_metal_buffer_record(history);
	NNMetalBufferRecord *keyRows = nn_metal_buffer_record(keys);
	NNMetalBufferRecord *valueRows = nn_metal_buffer_record(values);
	NNMetalBufferRecord *result = nn_metal_buffer_record(target);
	uint64_t planeBytes = (uint64_t)plane * sizeof(float);
	uint64_t prefixBytes = (uint64_t)prefix * sizeof(float);

	if (scopeRecord == NULL || keyRows == NULL || valueRows == NULL || result == NULL ||
		scopeRecord->committed || keyRows == result || valueRows == result ||
		keyRows->bytes != valueRows->bytes || prefixBytes > planeBytes ||
		keyRows->bytes > planeBytes - prefixBytes || result->bytes != 2 * planeBytes ||
		(prefix != 0 && (previous == NULL || previous == result || previous->bytes != result->bytes))) {
		nn_metal_set_error("metal: encode KV append: invalid state, buffers or range");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		id<MTLBlitCommandEncoder> encoder = [scopeRecord->commandBuffer blitCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode KV append: create blit encoder returned nil");
			return NNMetalStatusError;
		}
		if (prefix != 0) {
			[encoder copyFromBuffer:previous->value sourceOffset:0 toBuffer:result->value destinationOffset:0 size:(NSUInteger)prefixBytes];
			[encoder copyFromBuffer:previous->value sourceOffset:(NSUInteger)planeBytes toBuffer:result->value destinationOffset:(NSUInteger)planeBytes size:(NSUInteger)prefixBytes];
			[scopeRecord->retainedResources addObject:previous->value];
		}
		[encoder copyFromBuffer:keyRows->value sourceOffset:0 toBuffer:result->value destinationOffset:(NSUInteger)prefixBytes size:(NSUInteger)keyRows->bytes];
		[encoder copyFromBuffer:valueRows->value sourceOffset:0 toBuffer:result->value destinationOffset:(NSUInteger)(planeBytes + prefixBytes) size:(NSUInteger)valueRows->bytes];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:keyRows->value];
		[scopeRecord->retainedResources addObject:valueRows->value];
		[scopeRecord->retainedResources addObject:result->value];
	}
	return NNMetalStatusSuccess;
}

static int nn_metal_attention_math_valid(
	uint32_t operation, NNMetalBufferRecord *first, NNMetalBufferRecord *second,
	NNMetalBufferRecord *target, NNMetalBufferRecord *extra, const uint32_t params[6], uint32_t threads
) {
	if (params[0] == 0 || threads == 0 || first->bytes != target->bytes ||
		target->bytes / sizeof(float) > UINT32_MAX) {
		return 0;
	}
	uint64_t firstCount = first->bytes / sizeof(float);
	uint64_t secondCount = second->bytes / sizeof(float);
	switch (operation) {
	case NNMetalTensorRMSNorm:
		return (uint64_t)threads * params[0] == firstCount && params[0] == secondCount;
	case NNMetalTensorRotatePairs: {
		uint64_t rowPairs = (uint64_t)params[0] * params[1];
		return rowPairs > 0 && threads % rowPairs == 0 && threads * (uint64_t)2 == firstCount &&
			second->bytes == extra->bytes && secondCount <= UINT32_MAX &&
			secondCount % params[0] == 0 && (uint64_t)params[2] + threads / rowPairs <= secondCount / params[0];
	}
	case NNMetalTensorCausalAttention: {
		uint64_t kvRow = (uint64_t)params[0] * params[2];
		return kvRow > 0 && params[1] > 0 && params[1] % params[2] == 0 &&
			threads % params[1] == 0 && (uint64_t)threads * params[0] == firstCount &&
			params[3] <= UINT32_MAX / 2 && secondCount == 2 * (uint64_t)params[3] &&
			params[3] % kvRow == 0 && params[4] > 0 && params[4] <= params[3] / kvRow && params[5] <= params[4];
	}
	default:
		return 0;
	}
}

int nn_metal_scope_encode_attention_math(
	NNMetalScope scope, uint32_t operation, NNMetalBuffer first, NNMetalBuffer second,
	NNMetalBuffer target, NNMetalBuffer extra, const uint32_t params[6], uint32_t threads
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *buffers[4] = {
		nn_metal_buffer_record(first), nn_metal_buffer_record(second),
		nn_metal_buffer_record(target), nn_metal_buffer_record(extra),
	};
	if (scopeRecord == NULL || scopeRecord->committed || buffers[0] == NULL ||
		buffers[1] == NULL || buffers[2] == NULL || (buffers[3] != NULL) != (operation == NNMetalTensorRotatePairs) ||
		buffers[0] == buffers[2] || buffers[1] == buffers[2] || buffers[3] == buffers[2] ||
		params == NULL || !nn_metal_attention_math_valid(operation, buffers[0], buffers[1], buffers[2], buffers[3], params, threads)) {
		nn_metal_set_error("metal: encode attention math: invalid state, buffers or range");
		return NNMetalStatusError;
	}

	pthread_once(&nnMetalAttentionMathOnce, nn_metal_initialize_attention_math);
	id<MTLComputePipelineState> pipeline = nnMetalAttentionMathPipelines[operation];
	if (pipeline == nil) {
		nn_metal_set_error(nnMetalAttentionMathError);
		return NNMetalStatusError;
	}

	@autoreleasepool {
		id<MTLComputeCommandEncoder> encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode attention math: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		[encoder setComputePipelineState:pipeline];
		for (NSUInteger index = 0; index < 4 && buffers[index] != NULL; index++) {
			[encoder setBuffer:buffers[index]->value offset:0 atIndex:index];
			[scopeRecord->retainedResources addObject:buffers[index]->value];
		}
		[encoder setBytes:params length:6 * sizeof(uint32_t) atIndex:4];
		NSUInteger threadWidth = MIN([pipeline threadExecutionWidth], [pipeline maxTotalThreadsPerThreadgroup]);
		[encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_add_row_vector(
	NNMetalScope scope,
	NNMetalBuffer values,
	NNMetalBuffer rowVector,
	uint32_t rows,
	uint32_t cols
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *valuesRecord = nn_metal_buffer_record(values);
	NNMetalBufferRecord *rowVectorRecord = nn_metal_buffer_record(rowVector);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t countValue = (uint64_t)rows * cols;
	uint32_t count = 0;

	if (scopeRecord == NULL || valuesRecord == NULL || rowVectorRecord == NULL) {
		nn_metal_set_error("metal: encode row-vector addition: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 || countValue > UINT32_MAX ||
		countValue * sizeof(float) != valuesRecord->bytes ||
		(uint64_t)cols * sizeof(float) != rowVectorRecord->bytes) {
		nn_metal_set_error("metal: encode row-vector addition: invalid state, dimensions, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		count = (uint32_t)countValue;
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode row-vector addition: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalAddRowVectorPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalAddRowVectorPipeline];
		[encoder setBuffer:valuesRecord->value offset:0 atIndex:0];
		[encoder setBuffer:rowVectorRecord->value offset:0 atIndex:1];
		[encoder setBytes:&count length:sizeof(count) atIndex:2];
		[encoder setBytes:&cols length:sizeof(cols) atIndex:3];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:valuesRecord->value];
		[scopeRecord->retainedResources addObject:rowVectorRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_add_scaled(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	float scale,
	uint32_t count
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *leftRecord = nn_metal_buffer_record(left);
	NNMetalBufferRecord *rightRecord = nn_metal_buffer_record(right);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t bytes = (uint64_t)count * sizeof(float);

	if (scopeRecord == NULL || leftRecord == NULL || rightRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode scaled addition: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || count == 0 || bytes != leftRecord->bytes ||
		bytes != rightRecord->bytes || bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode scaled addition: invalid state, element count, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode scaled addition: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalAddScaledPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalAddScaledPipeline];
		[encoder setBuffer:leftRecord->value offset:0 atIndex:0];
		[encoder setBuffer:rightRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&scale length:sizeof(scale) atIndex:3];
		[encoder setBytes:&count length:sizeof(count) atIndex:4];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:leftRecord->value];
		[scopeRecord->retainedResources addObject:rightRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_relu(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t count
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *inputRecord = nn_metal_buffer_record(input);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t bytes = (uint64_t)count * sizeof(float);

	if (scopeRecord == NULL || inputRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode ReLU: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || count == 0 || bytes != inputRecord->bytes || bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode ReLU: invalid state, element count, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode ReLU: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalReLUPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalReLUPipeline];
		[encoder setBuffer:inputRecord->value offset:0 atIndex:0];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:1];
		[encoder setBytes:&count length:sizeof(count) atIndex:2];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:inputRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_relu_backward(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer outputGradient,
	NNMetalBuffer result,
	uint32_t count
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *inputRecord = nn_metal_buffer_record(input);
	NNMetalBufferRecord *outputGradientRecord = nn_metal_buffer_record(outputGradient);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t bytes = (uint64_t)count * sizeof(float);

	if (scopeRecord == NULL || inputRecord == NULL || outputGradientRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode ReLU backward: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || count == 0 || bytes != inputRecord->bytes ||
		bytes != outputGradientRecord->bytes || bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode ReLU backward: invalid state, element count, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode ReLU backward: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalReLUBackwardPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalReLUBackwardPipeline];
		[encoder setBuffer:inputRecord->value offset:0 atIndex:0];
		[encoder setBuffer:outputGradientRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&count length:sizeof(count) atIndex:3];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:inputRecord->value];
		[scopeRecord->retainedResources addObject:outputGradientRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_softmax_rows(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *inputRecord = nn_metal_buffer_record(input);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t count = (uint64_t)rows * cols;
	uint64_t bytes = count * sizeof(float);

	if (scopeRecord == NULL || inputRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode Softmax: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 || count > UINT32_MAX ||
		bytes != inputRecord->bytes || bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode Softmax: invalid state, dimensions, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode Softmax: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalSoftmaxRowsPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)rows + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalSoftmaxRowsPipeline];
		[encoder setBuffer:inputRecord->value offset:0 atIndex:0];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:1];
		[encoder setBytes:&rows length:sizeof(rows) atIndex:2];
		[encoder setBytes:&cols length:sizeof(cols) atIndex:3];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:inputRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_softmax_rows_backward(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer outputGradient,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *inputRecord = nn_metal_buffer_record(input);
	NNMetalBufferRecord *outputGradientRecord = nn_metal_buffer_record(outputGradient);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t count = (uint64_t)rows * cols;
	uint64_t bytes = count * sizeof(float);

	if (scopeRecord == NULL || inputRecord == NULL || outputGradientRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode Softmax backward: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 || count > UINT32_MAX ||
		bytes != inputRecord->bytes || bytes != outputGradientRecord->bytes ||
		bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode Softmax backward: invalid state, dimensions, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode Softmax backward: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalSoftmaxRowsBackwardPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)rows + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalSoftmaxRowsBackwardPipeline];
		[encoder setBuffer:inputRecord->value offset:0 atIndex:0];
		[encoder setBuffer:outputGradientRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&rows length:sizeof(rows) atIndex:3];
		[encoder setBytes:&cols length:sizeof(cols) atIndex:4];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:inputRecord->value];
		[scopeRecord->retainedResources addObject:outputGradientRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_column_sums(
	NNMetalScope scope,
	NNMetalBuffer input,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	uint32_t accumulate
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *inputRecord = nn_metal_buffer_record(input);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t inputBytes = (uint64_t)rows * cols * sizeof(float);
	uint64_t resultBytes = (uint64_t)cols * sizeof(float);

	if (scopeRecord == NULL || inputRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode column sums: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 || accumulate > 1 ||
		inputBytes != inputRecord->bytes || resultBytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode column sums: invalid state, dimensions, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode column sums: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN((NSUInteger)256, [nnMetalColumnSumsPipeline maxTotalThreadsPerThreadgroup]);
		groupCount = ((NSUInteger)cols + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalColumnSumsPipeline];
		[encoder setBuffer:inputRecord->value offset:0 atIndex:0];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:1];
		[encoder setBytes:&rows length:sizeof(rows) atIndex:2];
		[encoder setBytes:&cols length:sizeof(cols) atIndex:3];
		[encoder setBytes:&accumulate length:sizeof(accumulate) atIndex:4];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:inputRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_categorical_cross_entropy(
	NNMetalScope scope,
	NNMetalBuffer predictions,
	NNMetalBuffer targets,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	float epsilon
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *predictionRecord = nn_metal_buffer_record(predictions);
	NNMetalBufferRecord *targetRecord = nn_metal_buffer_record(targets);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	uint64_t valueBytes = (uint64_t)rows * cols * sizeof(float);

	if (scopeRecord == NULL || predictionRecord == NULL ||
		targetRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode categorical cross entropy: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 ||
		!(epsilon > 0.0f && epsilon < 0.5f) ||
		valueBytes != predictionRecord->bytes ||
		valueBytes != targetRecord->bytes ||
		5 * sizeof(float) != resultRecord->bytes) {
		nn_metal_set_error("metal: encode categorical cross entropy: invalid state, dimensions, epsilon, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode categorical cross entropy: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		[encoder setComputePipelineState:nnMetalCategoricalCrossEntropyPipeline];
		[encoder setBuffer:predictionRecord->value offset:0 atIndex:0];
		[encoder setBuffer:targetRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&rows length:sizeof(rows) atIndex:3];
		[encoder setBytes:&cols length:sizeof(cols) atIndex:4];
		[encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
		[encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:predictionRecord->value];
		[scopeRecord->retainedResources addObject:targetRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_categorical_cross_entropy_gradient(
	NNMetalScope scope,
	NNMetalBuffer predictions,
	NNMetalBuffer targets,
	NNMetalBuffer result,
	uint32_t rows,
	uint32_t cols,
	float epsilon
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *predictionRecord = nn_metal_buffer_record(predictions);
	NNMetalBufferRecord *targetRecord = nn_metal_buffer_record(targets);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 0;
	NSUInteger groupCount = 0;
	uint64_t countValue = (uint64_t)rows * cols;
	uint64_t bytes = countValue * sizeof(float);
	uint32_t count = 0;

	if (scopeRecord == NULL || predictionRecord == NULL ||
		targetRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode categorical cross entropy gradient: nil handle");
		return NNMetalStatusError;
	}
	if (scopeRecord->committed || rows == 0 || cols == 0 ||
		countValue > UINT32_MAX ||
		!(epsilon > 0.0f && epsilon < 0.5f) ||
		bytes != predictionRecord->bytes ||
		bytes != targetRecord->bytes ||
		bytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode categorical cross entropy gradient: invalid state, dimensions, epsilon, or buffer length");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		count = (uint32_t)countValue;
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode categorical cross entropy gradient: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		threadWidth = MIN(
			(NSUInteger)256,
			[nnMetalCategoricalCrossEntropyGradientPipeline maxTotalThreadsPerThreadgroup]
		);
		groupCount = ((NSUInteger)count + threadWidth - 1) / threadWidth;
		[encoder setComputePipelineState:nnMetalCategoricalCrossEntropyGradientPipeline];
		[encoder setBuffer:predictionRecord->value offset:0 atIndex:0];
		[encoder setBuffer:targetRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&count length:sizeof(count) atIndex:3];
		[encoder setBytes:&rows length:sizeof(rows) atIndex:4];
		[encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
		[encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:predictionRecord->value];
		[scopeRecord->retainedResources addObject:targetRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_matmul(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions,
	int tensorPipeline
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *leftRecord = nn_metal_buffer_record(left);
	NNMetalBufferRecord *rightRecord = nn_metal_buffer_record(right);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputePipelineState> pipeline = nnMetalMatMulPipeline;
	id<MTLComputeCommandEncoder> encoder = nil;
	NSUInteger threadWidth = 16;
	NSUInteger threadHeight = 16;
	NSUInteger groupWidth = 0;
	NSUInteger groupHeight = 0;
	uint64_t leftBytes = 0;
	uint64_t rightBytes = 0;
	uint64_t resultBytes = 0;

	if (scopeRecord == NULL || leftRecord == NULL || rightRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode matrix multiplication: nil handle");
		return NNMetalStatusError;
	}
	leftBytes = (uint64_t)dimensions.leftRows * dimensions.leftCols * sizeof(float);
	rightBytes = (uint64_t)dimensions.rightRows * dimensions.rightCols * sizeof(float);
	resultBytes = (uint64_t)dimensions.resultRows * dimensions.resultCols * sizeof(float);
	if (scopeRecord->committed || dimensions.variant > 2 ||
		leftBytes != leftRecord->bytes || rightBytes != rightRecord->bytes || resultBytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode matrix multiplication: invalid state, dimensions, or buffer length");
		return NNMetalStatusError;
	}
	if (tensorPipeline) {
		pthread_once(&nnMetalTensorMatMulOnce, nn_metal_initialize_tensor_matmul);
		pipeline = nnMetalTensorMatMulPipeline;
		if (pipeline == nil) {
			nn_metal_set_error(nnMetalTensorMatMulError);
			return NNMetalStatusError;
		}
		MTLSize limits = [nnMetalDevice maxThreadsPerThreadgroup];
		if (threadWidth > limits.width || threadHeight > limits.height ||
			threadWidth * threadHeight > [pipeline maxTotalThreadsPerThreadgroup]) {
			nn_metal_set_error("metal: encode tensor matmul: threadgroup size exceeds Metal limits");
			return NNMetalStatusError;
		}
	}

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode matrix multiplication: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		groupWidth = ((NSUInteger)dimensions.resultCols + threadWidth - 1) / threadWidth;
		groupHeight = ((NSUInteger)dimensions.resultRows + threadHeight - 1) / threadHeight;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:leftRecord->value offset:0 atIndex:0];
		[encoder setBuffer:rightRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:3];
		[encoder dispatchThreadgroups:MTLSizeMake(groupWidth, groupHeight, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:leftRecord->value];
		[scopeRecord->retainedResources addObject:rightRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_encode_matmul_mixed(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions,
	uint32_t weightDType
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *leftRecord = nn_metal_buffer_record(left);
	NNMetalBufferRecord *rightRecord = nn_metal_buffer_record(right);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
	id<MTLComputePipelineState> pipeline = nil;
	id<MTLComputeCommandEncoder> encoder = nil;
	uint64_t leftBytes = 0;
	uint64_t rightBytes = 0;
	uint64_t resultBytes = 0;
	int useRowRT = 0;
	NSUInteger pipelineIndex = 0;

	if (scopeRecord == NULL || leftRecord == NULL || rightRecord == NULL || resultRecord == NULL) {
		nn_metal_set_error("metal: encode mixed matrix multiplication: nil handle");
		return NNMetalStatusError;
	}
	leftBytes = (uint64_t)dimensions.leftRows * dimensions.leftCols * sizeof(float);
	rightBytes = (uint64_t)dimensions.rightRows * dimensions.rightCols * sizeof(uint16_t);
	resultBytes = (uint64_t)dimensions.resultRows * dimensions.resultCols * sizeof(float);
	if (scopeRecord->committed ||
		(dimensions.variant != 0 && dimensions.variant != 2) ||
		(weightDType != NNMetalWeightFloat16 && weightDType != NNMetalWeightBFloat16) ||
		leftBytes == 0 || rightBytes == 0 || resultBytes == 0 ||
		leftBytes != leftRecord->bytes || rightBytes != rightRecord->bytes || resultBytes != resultRecord->bytes) {
		nn_metal_set_error("metal: encode mixed matrix multiplication: invalid state, dimensions, dtype, or buffer length");
		return NNMetalStatusError;
	}

	pthread_once(&nnMetalTensorMatMulMixedOnce, nn_metal_initialize_tensor_matmul_mixed);
	useRowRT = (dimensions.resultRows == 1 && dimensions.variant == 2);
	if (weightDType == NNMetalWeightFloat16) {
		pipelineIndex = useRowRT ? NNMetalMatMulMixedF16RowRT : NNMetalMatMulMixedF16;
	} else {
		pipelineIndex = useRowRT ? NNMetalMatMulMixedBF16RowRT : NNMetalMatMulMixedBF16;
	}
	pipeline = nnMetalTensorMatMulMixedPipelines[pipelineIndex];
	if (pipeline == nil) {
		nn_metal_set_error(nnMetalTensorMatMulMixedError);
		return NNMetalStatusError;
	}

	MTLSize limits = [nnMetalDevice maxThreadsPerThreadgroup];
	if (useRowRT) {
		NSUInteger threads = 128;
		if (threads > limits.width || threads > [pipeline maxTotalThreadsPerThreadgroup]) {
			nn_metal_set_error("metal: encode mixed row matmul: threadgroup size exceeds Metal limits");
			return NNMetalStatusError;
		}
		@autoreleasepool {
			encoder = [scopeRecord->commandBuffer computeCommandEncoder];
			if (encoder == nil) {
				nn_metal_set_error("metal: encode mixed row matmul: create compute encoder returned nil");
				return NNMetalStatusError;
			}
			NSUInteger groups = ((NSUInteger)dimensions.resultCols + 7) / 8;
			[encoder setComputePipelineState:pipeline];
			[encoder setBuffer:leftRecord->value offset:0 atIndex:0];
			[encoder setBuffer:rightRecord->value offset:0 atIndex:1];
			[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
			[encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:3];
			[encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
			[encoder endEncoding];
			[scopeRecord->retainedResources addObject:leftRecord->value];
			[scopeRecord->retainedResources addObject:rightRecord->value];
			[scopeRecord->retainedResources addObject:resultRecord->value];
		}
		return NNMetalStatusSuccess;
	}

	NSUInteger threadWidth = 16;
	NSUInteger threadHeight = 16;
	if (threadWidth > limits.width || threadHeight > limits.height ||
		threadWidth * threadHeight > [pipeline maxTotalThreadsPerThreadgroup]) {
		nn_metal_set_error("metal: encode mixed matmul: threadgroup size exceeds Metal limits");
		return NNMetalStatusError;
	}
	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode mixed matmul: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		NSUInteger groupWidth = ((NSUInteger)dimensions.resultCols + threadWidth - 1) / threadWidth;
		NSUInteger groupHeight = ((NSUInteger)dimensions.resultRows + threadHeight - 1) / threadHeight;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:leftRecord->value offset:0 atIndex:0];
		[encoder setBuffer:rightRecord->value offset:0 atIndex:1];
		[encoder setBuffer:resultRecord->value offset:0 atIndex:2];
		[encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:3];
		[encoder dispatchThreadgroups:MTLSizeMake(groupWidth, groupHeight, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
		[encoder endEncoding];
		[scopeRecord->retainedResources addObject:leftRecord->value];
		[scopeRecord->retainedResources addObject:rightRecord->value];
		[scopeRecord->retainedResources addObject:resultRecord->value];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_scope_commit(NNMetalScope scope) {
	NNMetalScopeRecord *record = nn_metal_scope_record(scope);

	if (record == NULL || record->commandBuffer == nil) {
		nn_metal_set_error("metal: commit command scope: nil handle");
		return NNMetalStatusError;
	}
	if (record->committed) {
		nn_metal_set_error("metal: commit command scope: already committed");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		[record->commandBuffer commit];
		record->committed = 1;
	}
	nn_metal_record_command_submitted();
	return NNMetalStatusSuccess;
}

int nn_metal_scope_completed(NNMetalScope scope) {
	int status = NNMetalStatusError;

	@autoreleasepool {
		status = nn_metal_scope_completion_status(nn_metal_scope_record(scope), 0);
	}
	return status;
}

int nn_metal_scope_wait(NNMetalScope scope) {
	int status = NNMetalStatusError;

	@autoreleasepool {
		status = nn_metal_scope_completion_status(nn_metal_scope_record(scope), 1);
	}
	return status;
}

void nn_metal_scope_release(NNMetalScope scope) {
	NNMetalScopeRecord *record = nn_metal_scope_record(scope);

	if (record == NULL) {
		return;
	}

	@autoreleasepool {
		if (record->committed && [record->commandBuffer status] != MTLCommandBufferStatusCompleted &&
			[record->commandBuffer status] != MTLCommandBufferStatusError) {
			[record->commandBuffer waitUntilCompleted];
			nn_metal_record_command_completed(record);
		}
		[record->retainedResources release];
		[record->commandBuffer release];
		record->retainedResources = nil;
		record->commandBuffer = nil;
	}
	nn_metal_record_scope_released();
	free(record);
}

void nn_metal_resource_snapshot(NNMetalResourceSnapshot *snapshot) {
	if (snapshot == NULL) {
		return;
	}

	pthread_mutex_lock(&nnMetalResourceMutex);
	memcpy(snapshot, &nnMetalResources, sizeof(*snapshot));
	pthread_mutex_unlock(&nnMetalResourceMutex);
}

int nn_metal_resource_reset(void) {
	pthread_mutex_lock(&nnMetalResourceMutex);
	if (nnMetalResources.liveBuffers != 0 || nnMetalResources.liveScopes != 0) {
		pthread_mutex_unlock(&nnMetalResourceMutex);
		nn_metal_set_error("metal: reset resource counters: runtime has live resources");
		return NNMetalStatusError;
	}
	memset(&nnMetalResources, 0, sizeof(nnMetalResources));
	pthread_mutex_unlock(&nnMetalResourceMutex);
	return NNMetalStatusSuccess;
}

const char *nn_metal_last_error(void) {
	return nnMetalLastError;
}

int nn_metal_test_missing_kernel(const char *name) {
	id<MTLFunction> function = nil;
	NSString *kernelName = nil;

	if (name == NULL || nn_metal_initialize() != NNMetalStatusSuccess) {
		nn_metal_set_error("metal: test missing kernel: invalid name or unavailable runtime");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		kernelName = [NSString stringWithUTF8String:name];
		function = [nnMetalLibrary newFunctionWithName:kernelName];
		if (function == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: kernel %@ not found", kernelName]);
			return NNMetalStatusError;
		}
		[function release];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_test_compile_source(const char *sourceText) {
	NSError *error = nil;
	id<MTLLibrary> library = nil;
	NSString *source = nil;

	if (sourceText == NULL || nn_metal_initialize() != NNMetalStatusSuccess) {
		nn_metal_set_error("metal: test shader compilation: invalid source or unavailable runtime");
		return NNMetalStatusError;
	}

	@autoreleasepool {
		source = [NSString stringWithUTF8String:sourceText];
		library = [nnMetalDevice newLibraryWithSource:source options:nil error:&error];
		if (library == nil) {
			nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile test shader: %@", [error localizedDescription]]);
			return NNMetalStatusError;
		}
		[library release];
	}
	return NNMetalStatusSuccess;
}

int nn_metal_test_scope_fail(NNMetalScope scope) {
	NNMetalScopeRecord *record = nn_metal_scope_record(scope);

	if (record == NULL || record->committed) {
		nn_metal_set_error("metal: inject command failure: nil or committed scope");
		return NNMetalStatusError;
	}
	record->forcedFailure = 1;
	return NNMetalStatusSuccess;
}
