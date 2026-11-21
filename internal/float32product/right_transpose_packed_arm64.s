//go:build arm64 && !purego && !race

#include "textflag.h"

#define ACCUMULATE_ONE(lane) \
	VLD1.P 32(R1), [V9.S4, V10.S4]; \
	VDUP V6.S[lane], V11.S4; \
	VFMLA V9.S4, V11.S4, V0.S4; \
	VFMLA V10.S4, V11.S4, V1.S4

#define ACCUMULATE_TWO(lane) \
	ACCUMULATE_ONE(lane); \
	VDUP V7.S[lane], V12.S4; \
	VFMLA V9.S4, V12.S4, V2.S4; \
	VFMLA V10.S4, V12.S4, V3.S4

#define ACCUMULATE_THREE(lane) \
	ACCUMULATE_TWO(lane); \
	VDUP V8.S[lane], V13.S4; \
	VFMLA V9.S4, V13.S4, V4.S4; \
	VFMLA V10.S4, V13.S4, V5.S4

TEXT ·rightTransposePackedTileARM64(SB), NOSPLIT, $0-48
	MOVD left+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD output+16(FP), R2
	MOVD inner+24(FP), R3
	MOVD rows+40(FP), R4

	// The per-output chains start at +0 and visit K in ascending order.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	CMP $1, R4
	BNE multiple
	LSR $2, R3, R3
one:
	VLD1.P 16(R0), [V6.S4]
	ACCUMULATE_ONE(0)
	ACCUMULATE_ONE(1)
	ACCUMULATE_ONE(2)
	ACCUMULATE_ONE(3)
	SUBS $1, R3, R3
	BNE one
	VST1 [V0.S4, V1.S4], (R2)
	RET

multiple:
	MOVD stride+32(FP), R5
	ADD R3<<2, R0, R6
	ADD R5<<2, R2, R7
	LSR $2, R3, R8
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	CMP $2, R4
	BNE three_setup
two:
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R6), [V7.S4]
	ACCUMULATE_TWO(0)
	ACCUMULATE_TWO(1)
	ACCUMULATE_TWO(2)
	ACCUMULATE_TWO(3)
	SUBS $1, R8, R8
	BNE two
	VST1 [V0.S4, V1.S4], (R2)
	VST1 [V2.S4, V3.S4], (R7)
	RET

three_setup:
	ADD R3<<2, R6, R9
	ADD R5<<2, R7, R10
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
three:
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R6), [V7.S4]
	VLD1.P 16(R9), [V8.S4]
	ACCUMULATE_THREE(0)
	ACCUMULATE_THREE(1)
	ACCUMULATE_THREE(2)
	ACCUMULATE_THREE(3)
	SUBS $1, R8, R8
	BNE three
	VST1 [V0.S4, V1.S4], (R2)
	VST1 [V2.S4, V3.S4], (R7)
	VST1 [V4.S4, V5.S4], (R10)
	RET

// Transpose four K values across four dense weight rows into four vectors.
#define PACK_FOUR_OUTPUTS(r0, r1, r2, r3, k0, k1, k2, k3) \
	VLD1.P 16(r0), [V0.S4]; \
	VLD1.P 16(r1), [V1.S4]; \
	VLD1.P 16(r2), [V2.S4]; \
	VLD1.P 16(r3), [V3.S4]; \
	VZIP1 V1.S4, V0.S4, V4.S4; \
	VZIP2 V1.S4, V0.S4, V5.S4; \
	VZIP1 V3.S4, V2.S4, V6.S4; \
	VZIP2 V3.S4, V2.S4, V7.S4; \
	VZIP1 V6.D2, V4.D2, k0.D2; \
	VZIP2 V6.D2, V4.D2, k1.D2; \
	VZIP1 V7.D2, V5.D2, k2.D2; \
	VZIP2 V7.D2, V5.D2, k3.D2

TEXT ·packRightTransposeTileARM64(SB), NOSPLIT, $0-24
	MOVD dense+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD inner+16(FP), R2
	LSL $2, R2, R3
	LSR $2, R2, R2
	ADD R3, R0, R4
	ADD R3, R4, R5
	ADD R3, R5, R6
	ADD R3, R6, R7
	ADD R3, R7, R8
	ADD R3, R8, R9
	ADD R3, R9, R10
pack:
	PACK_FOUR_OUTPUTS(R0, R4, R5, R6, V16, V18, V20, V22)
	PACK_FOUR_OUTPUTS(R7, R8, R9, R10, V17, V19, V21, V23)
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R1)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R1)
	SUBS $1, R2, R2
	BNE pack
	RET
