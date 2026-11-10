//go:build arm64 && !purego

#include "textflag.h"

// Four vectors hold spatial outputs for four filters. The wrapper bounds the
// union of valid input/weight spans. Clipped lanes never replace their sums.
// first is the initial valid lane; end is its unbounded exclusive limit. Both
// decrease per tap, while the input pointer stays at column zero until first
// reaches zero. Common spans use width-specific loads without per-tap clipping.
TEXT ·conv2DResidualTileSums(SB), NOSPLIT, $0-136
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
	MOVD first+112(FP), R23
	MOVD end+120(FP), R24
	MOVD width+128(FP), R25
	LSL $2, R6, R6
	LSL $2, R7, R7
	LSL $2, R8, R8
	LSL $2, R22, R22
	MUL R3, R8, R9
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	// Lane indices for the clipped-column mask.
	MOVD $0x0000000100000000, R13
	VMOV R13, V31.D[0]
	MOVD $0x0000000300000002, R13
	VMOV R13, V31.D[1]

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
	MOVD R23, R26
	MOVD R24, R27
	MOVD R5, R15
column_loop:
	CBNZ R26, clipped_column
	SUB R25, R27, R14
	ADD $1, R14, R14
	CMP $0, R14
	BLE clipped_column
	CMP R15, R14
	CSEL LT, R14, R15, R14
	SUB R14, R15, R15
	SUB R14, R27, R27
	MOVD R14, R26
	CMP $2, R25
	BEQ common_two
	CMP $3, R25
	BEQ common_three
	B common_four
clipped_column:
	CMP R25, R27
	CSEL LT, R27, R25, R14 // last valid lane, exclusive
	CBNZ R26, left_load
	CMP $4, R14
	BNE short_load
	VLD1 (R16), [V4.S4]
	B weights
short_load:
	CMP $2, R14
	BLT one_load
	FMOVD (R16), F4
	BEQ weights
	ADD $8, R16, R13
	VLD1 (R13), V4.S[2]
	B weights
one_load:
	FMOVS (R16), F4
	B weights
left_load:
	VEOR V4.B16, V4.B16, V4.B16
	MOVD R16, R13
	CMP $2, R26
	BEQ left_two
	BGT left_three
	VLD1.P 4(R13), V4.S[1]
	CMP $2, R14
	BEQ weights
left_two:
	VLD1.P 4(R13), V4.S[2]
	CMP $3, R14
	BEQ weights
left_three:
	VLD1 (R13), V4.S[3]
weights:
	VLD1R.P 4(R17), [V5.S4]
	VLD1R.P 4(R19), [V6.S4]
	VLD1R.P 4(R20), [V7.S4]
	VLD1R.P 4(R21), [V8.S4]
	// Retain the prior sum in padded lanes, including signed zero and NaN bits.
	VDUP R26, V9.S4
	VDUP R14, V10.S4
	VCMGE V9.S4, V31.S4, V9.S4
	VCMGT V31.S4, V10.S4, V10.S4
	VAND V10.B16, V9.B16, V9.B16
	VMOV V0.B16, V16.B16
	VMOV V1.B16, V17.B16
	VMOV V2.B16, V18.B16
	VMOV V3.B16, V19.B16
	VFMLA V5.S4, V4.S4, V16.S4
	VFMLA V6.S4, V4.S4, V17.S4
	VFMLA V7.S4, V4.S4, V18.S4
	VFMLA V8.S4, V4.S4, V19.S4
	VBIT V9.B16, V16.B16, V0.B16
	VBIT V9.B16, V17.B16, V1.B16
	VBIT V9.B16, V18.B16, V2.B16
	VBIT V9.B16, V19.B16, V3.B16
	CBZ R26, advance_input
	SUB $1, R26, R26
	B next_column
advance_input:
	ADD $4, R16, R16
next_column:
	SUB $1, R27, R27
	SUBS $1, R15, R15
	BNE column_loop
	B row_done
common_two:
	FMOVD (R16), F4
	VLD1R.P 4(R17), [V5.S4]
	VLD1R.P 4(R19), [V6.S4]
	VLD1R.P 4(R20), [V7.S4]
	VLD1R.P 4(R21), [V8.S4]
	VFMLA V5.S4, V4.S4, V0.S4
	VFMLA V6.S4, V4.S4, V1.S4
	VFMLA V7.S4, V4.S4, V2.S4
	VFMLA V8.S4, V4.S4, V3.S4
	ADD $4, R16, R16
	SUBS $1, R26, R26
	BNE common_two
	B common_done
common_three:
	FMOVD (R16), F4
	ADD $8, R16, R13
	VLD1 (R13), V4.S[2]
	VLD1R.P 4(R17), [V5.S4]
	VLD1R.P 4(R19), [V6.S4]
	VLD1R.P 4(R20), [V7.S4]
	VLD1R.P 4(R21), [V8.S4]
	VFMLA V5.S4, V4.S4, V0.S4
	VFMLA V6.S4, V4.S4, V1.S4
	VFMLA V7.S4, V4.S4, V2.S4
	VFMLA V8.S4, V4.S4, V3.S4
	ADD $4, R16, R16
	SUBS $1, R26, R26
	BNE common_three
	B common_done
common_four:
	VLD1 (R16), [V4.S4]
	VLD1R.P 4(R17), [V5.S4]
	VLD1R.P 4(R19), [V6.S4]
	VLD1R.P 4(R20), [V7.S4]
	VLD1R.P 4(R21), [V8.S4]
	VFMLA V5.S4, V4.S4, V0.S4
	VFMLA V6.S4, V4.S4, V1.S4
	VFMLA V7.S4, V4.S4, V2.S4
	VFMLA V8.S4, V4.S4, V3.S4
	ADD $4, R16, R16
	SUBS $1, R26, R26
	BNE common_four
	B common_done
common_done:
	CBNZ R15, column_loop
row_done:
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

TEXT ·conv2DResidualTileBias(SB), NOSPLIT, $0-32
	MOVD sums+0(FP), R0
	MOVD bias_base+8(FP), R1
	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1R.P 4(R1), [V4.S4]
	VLD1R.P 4(R1), [V5.S4]
	VLD1R.P 4(R1), [V6.S4]
	VLD1R (R1), [V7.S4]
	VFADD V0.S4, V4.S4, V0.S4
	VFADD V1.S4, V5.S4, V1.S4
	VFADD V2.S4, V6.S4, V2.S4
	VFADD V3.S4, V7.S4, V3.S4
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)
	RET
