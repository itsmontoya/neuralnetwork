//go:build arm64 && !purego

#include "textflag.h"

// Complete spans are checked by conv2DFourByFour. Each vector holds four
// independent columns; channel, row and column loops retain scalar order.
TEXT ·conv2DInteriorTile(SB), NOSPLIT, $0-104
	MOVD input_base+0(FP), R0
	MOVD weights_base+24(FP), R1
	MOVD sums+48(FP), R2
	MOVD channels+56(FP), R3
	MOVD rows+64(FP), R4
	MOVD kernelW+72(FP), R5
	MOVD inputPlane+80(FP), R6
	MOVD inputW+88(FP), R7
	MOVD kernelPlane+96(FP), R8
	LSL $2, R6, R6
	LSL $2, R7, R7
	LSL $2, R8, R8
	MUL R3, R8, R9 // byte distance between filters
	LSL $2, R5, R22
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

channel_loop:
	MOVD R0, R10
	MOVD R1, R11
	MOVD R4, R12
row_loop:
	MOVD R10, R16
	MOVD R11, R17
	ADD R9, R17, R19
	ADD R9, R19, R20
	ADD R9, R20, R21
	MOVD R5, R15
column_loop:
	VLD1 (R16), [V4.S4]
	ADD $4, R16, R16
	VLD1R.P 4(R17), [V5.S4]
	VLD1R.P 4(R19), [V6.S4]
	VLD1R.P 4(R20), [V7.S4]
	VLD1R.P 4(R21), [V8.S4]
	// Match the scalar spatial helper's weight-first NaN operand priority.
	VFMLA V4.S4, V5.S4, V0.S4
	VFMLA V4.S4, V6.S4, V1.S4
	VFMLA V4.S4, V7.S4, V2.S4
	VFMLA V4.S4, V8.S4, V3.S4
	SUBS $1, R15, R15
	BNE column_loop
	ADD R7, R10, R10
	ADD R22, R11, R11
	SUBS $1, R12, R12
	BNE row_loop
	ADD R6, R0, R0
	ADD R8, R1, R1
	SUBS $1, R3, R3
	BNE channel_loop
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	RET
