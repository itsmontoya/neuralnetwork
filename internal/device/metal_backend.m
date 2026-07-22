//go:build darwin && cgo && metal && !purego

#include "metal_backend.h"

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

static const char *nnMetalShaderSource =
	"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"struct NNMetalMatMulDimensions {\n"
	"    uint leftRows;\n"
	"    uint leftCols;\n"
	"    uint rightRows;\n"
	"    uint rightCols;\n"
	"    uint resultRows;\n"
	"    uint resultCols;\n"
	"    uint variant;\n"
	"};\n"
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
static id<MTLComputePipelineState> nnMetalMatMulPipeline = nil;
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
	[nnMetalFillPipeline release];
	[nnMetalLibrary release];
	[nnMetalCommandQueue release];
	[nnMetalDevice release];
	nnMetalMatMulPipeline = nil;
	nnMetalFillPipeline = nil;
	nnMetalLibrary = nil;
	nnMetalCommandQueue = nil;
	nnMetalDevice = nil;
}

static id<MTLComputePipelineState> nn_metal_new_pipeline(NSString *name) {
	NSError *error = nil;
	id<MTLFunction> function = nil;
	id<MTLComputePipelineState> pipeline = nil;

	function = [nnMetalLibrary newFunctionWithName:name];
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
				source = [NSString stringWithUTF8String:nnMetalShaderSource];
				nnMetalLibrary = [nnMetalDevice newLibraryWithSource:source options:nil error:&error];
				if (nnMetalLibrary == nil) {
					nn_metal_set_error_ns([NSString stringWithFormat:@"metal: compile shader library: %@", [error localizedDescription]]);
				}
			}

			if (nnMetalLibrary != nil) {
				nnMetalFillPipeline = nn_metal_new_pipeline(@"nn_fill");
			}
			if (nnMetalFillPipeline != nil) {
				nnMetalMatMulPipeline = nn_metal_new_pipeline(@"nn_matmul");
			}

			if (nnMetalCommandQueue != nil && nnMetalLibrary != nil && nnMetalFillPipeline != nil && nnMetalMatMulPipeline != nil) {
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

int nn_metal_scope_encode_matmul(
	NNMetalScope scope,
	NNMetalBuffer left,
	NNMetalBuffer right,
	NNMetalBuffer result,
	NNMetalMatMulDimensions dimensions
) {
	NNMetalScopeRecord *scopeRecord = nn_metal_scope_record(scope);
	NNMetalBufferRecord *leftRecord = nn_metal_buffer_record(left);
	NNMetalBufferRecord *rightRecord = nn_metal_buffer_record(right);
	NNMetalBufferRecord *resultRecord = nn_metal_buffer_record(result);
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

	@autoreleasepool {
		encoder = [scopeRecord->commandBuffer computeCommandEncoder];
		if (encoder == nil) {
			nn_metal_set_error("metal: encode matrix multiplication: create compute encoder returned nil");
			return NNMetalStatusError;
		}
		groupWidth = ((NSUInteger)dimensions.resultCols + threadWidth - 1) / threadWidth;
		groupHeight = ((NSUInteger)dimensions.resultRows + threadHeight - 1) / threadHeight;
		[encoder setComputePipelineState:nnMetalMatMulPipeline];
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
