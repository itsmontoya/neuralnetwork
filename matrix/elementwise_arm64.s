//go:build arm64 && !purego

#include "textflag.h"

// func addIntoArm64(left, right, result []float64)
TEXT ·addIntoArm64(SB), NOSPLIT, $0-72
	MOVD left_base+0(FP), R0
	MOVD left_len+8(FP), R1
	MOVD right_base+24(FP), R2
	MOVD result_base+48(FP), R3
	VMOVD $0x3ff0000000000000, V31
	VDUP V31.D[0], V31.D2

addIntoLoop4:
	CMP $4, R1
	BLT addIntoLoop2
	VLD1.P 32(R0), [V0.D2, V1.D2]
	VLD1.P 32(R2), [V2.D2, V3.D2]
	VFMLA V31.D2, V2.D2, V0.D2
	VFMLA V31.D2, V3.D2, V1.D2
	VST1.P [V0.D2, V1.D2], 32(R3)
	SUB $4, R1
	B addIntoLoop4

addIntoLoop2:
	CMP $2, R1
	BLT addIntoTail
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMLA V31.D2, V1.D2, V0.D2
	VST1.P [V0.D2], 16(R3)
	SUB $2, R1
	B addIntoLoop2

addIntoTail:
	CBZ R1, addIntoDone
	FMOVD (R0), F0
	FMOVD (R2), F1
	FADDD F1, F0, F0
	FMOVD F0, (R3)

addIntoDone:
	RET

// func addScaledInPlaceArm64(left, right []float64, scale float64)
TEXT ·addScaledInPlaceArm64(SB), NOSPLIT, $0-56
	MOVD left_base+0(FP), R0
	MOVD left_len+8(FP), R1
	MOVD right_base+24(FP), R2
	MOVD left_base+0(FP), R3
	MOVD scale+48(FP), R4
	VDUP R4, V31.D2

addScaledInPlaceLoop4:
	CMP $4, R1
	BLT addScaledInPlaceLoop2
	VLD1.P 32(R0), [V0.D2, V1.D2]
	VLD1.P 32(R2), [V2.D2, V3.D2]
	VFMLA V31.D2, V2.D2, V0.D2
	VFMLA V31.D2, V3.D2, V1.D2
	VST1.P [V0.D2, V1.D2], 32(R3)
	SUB $4, R1
	B addScaledInPlaceLoop4

addScaledInPlaceLoop2:
	CMP $2, R1
	BLT addScaledInPlaceTail
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMLA V31.D2, V1.D2, V0.D2
	VST1.P [V0.D2], 16(R3)
	SUB $2, R1
	B addScaledInPlaceLoop2

addScaledInPlaceTail:
	CBZ R1, addScaledInPlaceDone
	FMOVD (R0), F0
	FMOVD (R2), F1
	FMOVD scale+48(FP), F2
	FMADDD F2, F0, F1, F0
	FMOVD F0, (R3)

addScaledInPlaceDone:
	RET

// func subtractIntoArm64(left, right, result []float64)
TEXT ·subtractIntoArm64(SB), NOSPLIT, $0-72
	MOVD left_base+0(FP), R0
	MOVD left_len+8(FP), R1
	MOVD right_base+24(FP), R2
	MOVD result_base+48(FP), R3
	VMOVD $0xbff0000000000000, V31
	VDUP V31.D[0], V31.D2

subtractIntoLoop4:
	CMP $4, R1
	BLT subtractIntoLoop2
	VLD1.P 32(R0), [V0.D2, V1.D2]
	VLD1.P 32(R2), [V2.D2, V3.D2]
	VFMLA V31.D2, V2.D2, V0.D2
	VFMLA V31.D2, V3.D2, V1.D2
	VST1.P [V0.D2, V1.D2], 32(R3)
	SUB $4, R1
	B subtractIntoLoop4

subtractIntoLoop2:
	CMP $2, R1
	BLT subtractIntoTail
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMLA V31.D2, V1.D2, V0.D2
	VST1.P [V0.D2], 16(R3)
	SUB $2, R1
	B subtractIntoLoop2

subtractIntoTail:
	CBZ R1, subtractIntoDone
	FMOVD (R0), F0
	FMOVD (R2), F1
	FSUBD F1, F0, F0
	FMOVD F0, (R3)

subtractIntoDone:
	RET

// func addScalarIntoArm64(source []float64, value float64, result []float64)
TEXT ·addScalarIntoArm64(SB), NOSPLIT, $0-56
	MOVD source_base+0(FP), R0
	MOVD source_len+8(FP), R1
	MOVD value+24(FP), R2
	MOVD result_base+32(FP), R3
	VDUP R2, V30.D2
	VMOVD $0x3ff0000000000000, V31
	VDUP V31.D[0], V31.D2

addScalarIntoLoop4:
	CMP $4, R1
	BLT addScalarIntoLoop2
	VLD1.P 32(R0), [V0.D2, V1.D2]
	VFMLA V30.D2, V31.D2, V0.D2
	VFMLA V30.D2, V31.D2, V1.D2
	VST1.P [V0.D2, V1.D2], 32(R3)
	SUB $4, R1
	B addScalarIntoLoop4

addScalarIntoLoop2:
	CMP $2, R1
	BLT addScalarIntoTail
	VLD1.P 16(R0), [V0.D2]
	VFMLA V30.D2, V31.D2, V0.D2
	VST1.P [V0.D2], 16(R3)
	SUB $2, R1
	B addScalarIntoLoop2

addScalarIntoTail:
	CBZ R1, addScalarIntoDone
	FMOVD (R0), F0
	FMOVD value+24(FP), F1
	FADDD F1, F0, F0
	FMOVD F0, (R3)

addScalarIntoDone:
	RET
