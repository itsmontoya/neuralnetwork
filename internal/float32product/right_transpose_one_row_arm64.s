//go:build arm64 && !purego && !race

#include "textflag.h"

TEXT ·rightTransposeOneRowTileARM64(SB), NOSPLIT, $0-40
	MOVD left_base+0(FP), R0
	MOVD left_len+8(FP), R1
	MOVD weights+24(FP), R2
	MOVD output+32(FP), R3
	LSL $2, R1, R4
	LSR $2, R1, R1
	ADD R4, R2, R5
	ADD R4, R5, R6
	ADD R4, R6, R7
	ADD R4, R7, R8
	ADD R4, R8, R9
	ADD R4, R9, R10
	ADD R4, R10, R11

	// Each accumulator lane is one output column, initialized to +0.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16

contraction:
	VLD1.P 16(R0), [V2.S4]
	VDUP V2.S[0], V3.S4
	VDUP V2.S[1], V4.S4
	VDUP V2.S[2], V5.S4
	VDUP V2.S[3], V6.S4

	// Transpose four adjacent K values from each weight row in registers.
	// V15–V18 each hold the weights for a single, consecutive K index.
	VLD1.P 16(R2), [V7.S4]
	VLD1.P 16(R5), [V8.S4]
	VLD1.P 16(R6), [V9.S4]
	VLD1.P 16(R7), [V10.S4]
	VZIP1 V8.S4, V7.S4, V11.S4
	VZIP2 V8.S4, V7.S4, V12.S4
	VZIP1 V10.S4, V9.S4, V13.S4
	VZIP2 V10.S4, V9.S4, V14.S4
	VZIP1 V13.D2, V11.D2, V15.D2
	VZIP2 V13.D2, V11.D2, V16.D2
	VZIP1 V14.D2, V12.D2, V17.D2
	VZIP2 V14.D2, V12.D2, V18.D2
	VFMLA V15.S4, V3.S4, V0.S4
	VFMLA V16.S4, V4.S4, V0.S4
	VFMLA V17.S4, V5.S4, V0.S4
	VFMLA V18.S4, V6.S4, V0.S4

	// The next four columns reuse the same input values and K order.
	VLD1.P 16(R8), [V7.S4]
	VLD1.P 16(R9), [V8.S4]
	VLD1.P 16(R10), [V9.S4]
	VLD1.P 16(R11), [V10.S4]
	VZIP1 V8.S4, V7.S4, V11.S4
	VZIP2 V8.S4, V7.S4, V12.S4
	VZIP1 V10.S4, V9.S4, V13.S4
	VZIP2 V10.S4, V9.S4, V14.S4
	VZIP1 V13.D2, V11.D2, V15.D2
	VZIP2 V13.D2, V11.D2, V16.D2
	VZIP1 V14.D2, V12.D2, V17.D2
	VZIP2 V14.D2, V12.D2, V18.D2
	VFMLA V15.S4, V3.S4, V1.S4
	VFMLA V16.S4, V4.S4, V1.S4
	VFMLA V17.S4, V5.S4, V1.S4
	VFMLA V18.S4, V6.S4, V1.S4
	SUBS $1, R1, R1
	BNE contraction

	VST1 [V0.S4, V1.S4], (R3)
	RET
