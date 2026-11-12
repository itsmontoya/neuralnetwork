//go:build arm64 && !purego

#include "textflag.h"

// Each lane is classified as an integer bit pattern. Clear negative non-NaNs,
// including -0 and -Inf; retain positive values and every original NaN bit.
TEXT ·reluPreservingNaNBlocks(SB), NOSPLIT, $0-49
	MOVD source_base+0(FP), R0
	MOVD source_len+8(FP), R1
	MOVD destination_base+24(FP), R2
	MOVD $0x7fffffff, R3
	VDUP R3, V4.S4
	MOVD $0x7f800000, R3
	VDUP R3, V5.S4
	VEOR V6.B16, V6.B16, V6.B16

loop:
	VLD1.P 16(R0), [V0.S4]
	VAND V4.B16, V0.B16, V1.B16
	VCMHI V5.S4, V1.S4, V2.S4 // magnitude > +Inf means NaN
	VSSHR $31, V0.S4, V3.S4
	VBIC V2.B16, V3.B16, V3.B16 // negative and not NaN
	VBIC V3.B16, V0.B16, V0.B16
	VORR V2.B16, V6.B16, V6.B16
	VST1.P [V0.S4], 16(R2)
	SUBS $4, R1, R1
	BNE loop

	VUMAXV V6.S4, V6
	VMOV V6.S[0], R0
	CMP $0, R0
	CSET NE, R0
	MOVB R0, hasNaN+48(FP)
	RET
