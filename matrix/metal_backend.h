#ifndef NN_MATRIX_METAL_BACKEND_H
#define NN_MATRIX_METAL_BACKEND_H

#include <stdint.h>

enum {
	NNMetalMatMulStandard = 0,
	NNMetalMatMulLeftTranspose = 1,
	NNMetalMatMulRightTranspose = 2,
};

int nn_metal_available(void);
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
	uint32_t variant
);
const char *nn_metal_last_error(void);

#endif
