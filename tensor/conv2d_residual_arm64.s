//go:build arm64 && !purego

#include "textflag.h"

// The caller proves the clipped input and four-filter spans. Every load reads
// one float32; reduction order and multiplicand order match conv2DFourChannels.
TEXT ·conv2DResidualSums(SB), NOSPLIT, $0-112
	MOVD input_base+0(FP), R0
	MOVD weights_base+24(FP), R1
	MOVD sums+48(FP), R2
	MOVD channels+56(FP), R3
	MOVD rows+64(FP), R4
	MOVD columns+72(FP), R5
	MOVD inputPlane+80(FP), R6
	MOVD inputW+88(FP), R7
	MOVD kernelPlane+96(FP), R8
	MOVD kernelW+104(FP), R22
	LSL $2, R6, R6
	LSL $2, R7, R7
	LSL $2, R8, R8
	LSL $2, R22, R22
	MUL R3, R8, R9 // byte distance between filters
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
	FMOVS.P 4(R16), F4
	FMOVS.P 4(R17), F5
	FMOVS.P 4(R19), F6
	FMOVS.P 4(R20), F7
	FMOVS.P 4(R21), F8
	FMADDS F5, F0, F4, F0
	FMADDS F6, F1, F4, F1
	FMADDS F7, F2, F4, F2
	FMADDS F8, F3, F4, F3
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
	FMOVS F0, (R2)
	FMOVS F1, 4(R2)
	FMOVS F2, 8(R2)
	FMOVS F3, 12(R2)
	RET

// Bias is added after reduction, with the same operand priority as the released
// scalar four-filter caller. The Go wrapper bounds all four bias loads.
TEXT ·conv2DResidualBias(SB), NOSPLIT, $0-32
	MOVD sums+0(FP), R0
	MOVD bias_base+8(FP), R1
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VFADD V0.S4, V1.S4, V0.S4
	VST1 [V0.S4], (R0)
	RET
