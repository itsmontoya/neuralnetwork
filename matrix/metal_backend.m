//go:build darwin && cgo && metal && !purego

#include "metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
	uint32_t leftRows;
	uint32_t leftCols;
	uint32_t rightRows;
	uint32_t rightCols;
	uint32_t resultRows;
	uint32_t resultCols;
	uint32_t variant;
} NNMatMulParams;

static const char *nnMetalShaderSource =
	"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"struct NNMatMulParams {\n"
	"    uint leftRows;\n"
	"    uint leftCols;\n"
	"    uint rightRows;\n"
	"    uint rightCols;\n"
	"    uint resultRows;\n"
	"    uint resultCols;\n"
	"    uint variant;\n"
	"};\n"
	"kernel void nn_matmul(\n"
	"    const device float *left [[buffer(0)]],\n"
	"    const device float *right [[buffer(1)]],\n"
	"    device float *result [[buffer(2)]],\n"
	"    constant NNMatMulParams &params [[buffer(3)]],\n"
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
	"    } else if (params.variant == 2) {\n"
	"        for (uint inner = 0; inner < params.leftCols; inner++) {\n"
	"            sum += left[row * params.leftCols + inner] * right[col * params.rightCols + inner];\n"
	"        }\n"
	"    }\n"
	"    result[row * params.resultCols + col] = sum;\n"
	"}\n";

static pthread_mutex_t nnMetalInitMutex = PTHREAD_MUTEX_INITIALIZER;
static int nnMetalInitialized = 0;
static char nnMetalLastError[1024];
static id<MTLDevice> nnMetalDevice = nil;
static id<MTLCommandQueue> nnMetalCommandQueue = nil;
static id<MTLComputePipelineState> nnMetalMatMulPipeline = nil;

static void nn_metal_set_error(const char *message) {
	if (message == NULL) {
		message = "metal: unknown error";
	}

	strncpy(nnMetalLastError, message, sizeof(nnMetalLastError)-1);
	nnMetalLastError[sizeof(nnMetalLastError)-1] = '\0';
}

static void nn_metal_set_error_ns(NSString *message) {
	const char *text = NULL;

	if (message != nil) {
		text = [message UTF8String];
	}

	nn_metal_set_error(text);
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
		nn_metal_set_error_ns([error localizedDescription]);
		return nil;
	}

	return pipeline;
}

static int nn_metal_initialize(void) {
	int ready = 0;

	pthread_mutex_lock(&nnMetalInitMutex);
	if (nnMetalInitialized) {
		ready = nnMetalDevice != nil &&
			nnMetalCommandQueue != nil &&
			nnMetalMatMulPipeline != nil;
		pthread_mutex_unlock(&nnMetalInitMutex);
		return ready;
	}

	nnMetalInitialized = 1;
	@autoreleasepool {
		NSError *error = nil;
		NSString *source = nil;
		id<MTLLibrary> library = nil;

		nnMetalDevice = MTLCreateSystemDefaultDevice();
		if (nnMetalDevice == nil) {
			nn_metal_set_error("metal: no default device");
			pthread_mutex_unlock(&nnMetalInitMutex);
			return 0;
		}
		[nnMetalDevice retain];

		nnMetalCommandQueue = [nnMetalDevice newCommandQueue];
		if (nnMetalCommandQueue == nil) {
			nn_metal_set_error("metal: command queue creation failed");
			pthread_mutex_unlock(&nnMetalInitMutex);
			return 0;
		}

		source = [NSString stringWithUTF8String:nnMetalShaderSource];
		library = [nnMetalDevice newLibraryWithSource:source options:nil error:&error];
		if (library == nil) {
			nn_metal_set_error_ns([error localizedDescription]);
			pthread_mutex_unlock(&nnMetalInitMutex);
			return 0;
		}

		nnMetalMatMulPipeline = nn_metal_new_pipeline(library, @"nn_matmul");
		[library release];

		ready = nnMetalMatMulPipeline != nil;
	}
	pthread_mutex_unlock(&nnMetalInitMutex);

	return ready;
}

static int nn_metal_bytes_for_count(uint64_t count, size_t *bytes) {
	if (count > SIZE_MAX / sizeof(float)) {
		nn_metal_set_error("metal: buffer length overflow");
		return 0;
	}

	*bytes = (size_t)count * sizeof(float);
	return 1;
}

static id<MTLBuffer> nn_metal_new_buffer_with_bytes(const float *values, size_t bytes) {
	id<MTLBuffer> buffer = nil;

	buffer = [nnMetalDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	if (buffer == nil) {
		nn_metal_set_error("metal: buffer allocation failed");
		return nil;
	}

	memcpy([buffer contents], values, bytes);

	return buffer;
}

static void nn_metal_copy_buffer_to_floats(id<MTLBuffer> buffer, float *values, size_t bytes) {
	memcpy(values, [buffer contents], bytes);
}

static id<MTLBuffer> nn_metal_new_buffer(size_t bytes) {
	id<MTLBuffer> buffer = nil;

	buffer = [nnMetalDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	if (buffer == nil) {
		nn_metal_set_error("metal: buffer allocation failed");
		return nil;
	}

	return buffer;
}

static int nn_metal_wait(id<MTLCommandBuffer> commandBuffer, NNMetalCounters *counters) {
	NSError *error = nil;

	[commandBuffer commit];
	if (counters != NULL) {
		counters->commandSubmissions++;
	}

	[commandBuffer waitUntilCompleted];
	if (counters != NULL) {
		counters->waits++;
	}

	if ([commandBuffer status] == MTLCommandBufferStatusCompleted) {
		return 1;
	}

	error = [commandBuffer error];
	if (error != nil) {
		nn_metal_set_error_ns([error localizedDescription]);
		return 0;
	}

	nn_metal_set_error("metal: command failed");
	return 0;
}

int nn_metal_available(void) {
	return nn_metal_initialize();
}

int nn_metal_matmul(
	const float *left,
	const float *right,
	float *result,
	uint32_t leftRows,
	uint32_t leftCols,
	uint32_t rightRows,
	uint32_t rightCols,
	uint32_t resultRows,
	uint32_t resultCols,
	uint32_t variant,
	NNMetalCounters *counters
) {
	NNMatMulParams params;
	uint64_t leftCount = 0;
	uint64_t rightCount = 0;
	uint64_t resultCount = 0;
	size_t leftBytes = 0;
	size_t rightBytes = 0;
	size_t resultBytes = 0;
	NSUInteger threadWidth = 16;
	NSUInteger threadHeight = 16;
	NSUInteger groupWidth = 0;
	NSUInteger groupHeight = 0;
	id<MTLBuffer> leftBuffer = nil;
	id<MTLBuffer> rightBuffer = nil;
	id<MTLBuffer> resultBuffer = nil;
	id<MTLCommandBuffer> commandBuffer = nil;
	id<MTLComputeCommandEncoder> encoder = nil;
	int ok = 0;

	if (counters != NULL) {
		memset(counters, 0, sizeof(*counters));
	}

	if (left == NULL || right == NULL || result == NULL) {
		nn_metal_set_error("metal: nil matrix-multiply pointer");
		return 0;
	}
	if (!nn_metal_initialize()) {
		return 0;
	}

	leftCount = (uint64_t)leftRows * (uint64_t)leftCols;
	rightCount = (uint64_t)rightRows * (uint64_t)rightCols;
	resultCount = (uint64_t)resultRows * (uint64_t)resultCols;
	if (!nn_metal_bytes_for_count(leftCount, &leftBytes) ||
		!nn_metal_bytes_for_count(rightCount, &rightBytes) ||
		!nn_metal_bytes_for_count(resultCount, &resultBytes)) {
		return 0;
	}

	@autoreleasepool {
		leftBuffer = nn_metal_new_buffer_with_bytes(left, leftBytes);
		if (leftBuffer != nil && counters != NULL) {
			counters->bufferCreations++;
			counters->inputUploads++;
		}

		rightBuffer = nn_metal_new_buffer_with_bytes(right, rightBytes);
		if (rightBuffer != nil && counters != NULL) {
			counters->bufferCreations++;
			counters->inputUploads++;
		}

		resultBuffer = nn_metal_new_buffer(resultBytes);
		if (resultBuffer != nil && counters != NULL) {
			counters->bufferCreations++;
		}

		commandBuffer = [nnMetalCommandQueue commandBuffer];
		encoder = [commandBuffer computeCommandEncoder];
		if (leftBuffer == nil || rightBuffer == nil || resultBuffer == nil || commandBuffer == nil || encoder == nil) {
			nn_metal_set_error("metal: matrix-multiply command setup failed");
			ok = 0;
		} else {
			params.leftRows = leftRows;
			params.leftCols = leftCols;
			params.rightRows = rightRows;
			params.rightCols = rightCols;
			params.resultRows = resultRows;
			params.resultCols = resultCols;
			params.variant = variant;

			groupWidth = ((NSUInteger)resultCols + threadWidth - 1) / threadWidth;
			groupHeight = ((NSUInteger)resultRows + threadHeight - 1) / threadHeight;

			[encoder setComputePipelineState:nnMetalMatMulPipeline];
			[encoder setBuffer:leftBuffer offset:0 atIndex:0];
			[encoder setBuffer:rightBuffer offset:0 atIndex:1];
			[encoder setBuffer:resultBuffer offset:0 atIndex:2];
			[encoder setBytes:&params length:sizeof(params) atIndex:3];
			[encoder dispatchThreadgroups:MTLSizeMake(groupWidth, groupHeight, 1) threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
			[encoder endEncoding];

			ok = nn_metal_wait(commandBuffer, counters);
			if (ok) {
				nn_metal_copy_buffer_to_floats(resultBuffer, result, resultBytes);
				if (counters != NULL) {
					counters->resultDownloads++;
				}
			}
		}

		[leftBuffer release];
		[rightBuffer release];
		[resultBuffer release];
	}

	return ok;
}

const char *nn_metal_last_error(void) {
	return nnMetalLastError;
}
