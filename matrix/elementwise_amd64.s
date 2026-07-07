//go:build amd64 && !purego

#include "textflag.h"

// func addIntoAMD64(left, right, result []float64)
TEXT ·addIntoAMD64(SB), NOSPLIT, $0-72
	MOVQ left_base+0(FP), AX
	MOVQ left_len+8(FP), CX
	MOVQ right_base+24(FP), DX
	MOVQ result_base+48(FP), BX

addIntoLoop4:
	CMPQ CX, $4
	JL addIntoLoop2
	MOVUPD 0(AX), X0
	MOVUPD 16(AX), X1
	MOVUPD 0(DX), X2
	MOVUPD 16(DX), X3
	ADDPD X2, X0
	ADDPD X3, X1
	MOVUPD X0, 0(BX)
	MOVUPD X1, 16(BX)
	ADDQ $32, AX
	ADDQ $32, DX
	ADDQ $32, BX
	SUBQ $4, CX
	JMP addIntoLoop4

addIntoLoop2:
	CMPQ CX, $2
	JL addIntoTail
	MOVUPD 0(AX), X0
	MOVUPD 0(DX), X1
	ADDPD X1, X0
	MOVUPD X0, 0(BX)
	ADDQ $16, AX
	ADDQ $16, DX
	ADDQ $16, BX
	SUBQ $2, CX
	JMP addIntoLoop2

addIntoTail:
	TESTQ CX, CX
	JE addIntoDone
	MOVSD 0(AX), X0
	ADDSD 0(DX), X0
	MOVSD X0, 0(BX)

addIntoDone:
	RET

// func addScaledInPlaceAMD64(left, right []float64, scale float64)
TEXT ·addScaledInPlaceAMD64(SB), NOSPLIT, $0-56
	MOVQ left_base+0(FP), AX
	MOVQ left_len+8(FP), CX
	MOVQ right_base+24(FP), DX
	MOVQ left_base+0(FP), BX
	MOVQ scale+48(FP), X4
	SHUFPD $0, X4, X4

addScaledInPlaceLoop4:
	CMPQ CX, $4
	JL addScaledInPlaceLoop2
	MOVUPD 0(AX), X0
	MOVUPD 16(AX), X1
	MOVUPD 0(DX), X2
	MOVUPD 16(DX), X3
	MULPD X4, X2
	MULPD X4, X3
	ADDPD X2, X0
	ADDPD X3, X1
	MOVUPD X0, 0(BX)
	MOVUPD X1, 16(BX)
	ADDQ $32, AX
	ADDQ $32, DX
	ADDQ $32, BX
	SUBQ $4, CX
	JMP addScaledInPlaceLoop4

addScaledInPlaceLoop2:
	CMPQ CX, $2
	JL addScaledInPlaceTail
	MOVUPD 0(AX), X0
	MOVUPD 0(DX), X1
	MULPD X4, X1
	ADDPD X1, X0
	MOVUPD X0, 0(BX)
	ADDQ $16, AX
	ADDQ $16, DX
	ADDQ $16, BX
	SUBQ $2, CX
	JMP addScaledInPlaceLoop2

addScaledInPlaceTail:
	TESTQ CX, CX
	JE addScaledInPlaceDone
	MOVSD 0(AX), X0
	MOVSD 0(DX), X1
	MULSD X4, X1
	ADDSD X1, X0
	MOVSD X0, 0(BX)

addScaledInPlaceDone:
	RET

// func subtractIntoAMD64(left, right, result []float64)
TEXT ·subtractIntoAMD64(SB), NOSPLIT, $0-72
	MOVQ left_base+0(FP), AX
	MOVQ left_len+8(FP), CX
	MOVQ right_base+24(FP), DX
	MOVQ result_base+48(FP), BX

subtractIntoLoop4:
	CMPQ CX, $4
	JL subtractIntoLoop2
	MOVUPD 0(AX), X0
	MOVUPD 16(AX), X1
	MOVUPD 0(DX), X2
	MOVUPD 16(DX), X3
	SUBPD X2, X0
	SUBPD X3, X1
	MOVUPD X0, 0(BX)
	MOVUPD X1, 16(BX)
	ADDQ $32, AX
	ADDQ $32, DX
	ADDQ $32, BX
	SUBQ $4, CX
	JMP subtractIntoLoop4

subtractIntoLoop2:
	CMPQ CX, $2
	JL subtractIntoTail
	MOVUPD 0(AX), X0
	MOVUPD 0(DX), X1
	SUBPD X1, X0
	MOVUPD X0, 0(BX)
	ADDQ $16, AX
	ADDQ $16, DX
	ADDQ $16, BX
	SUBQ $2, CX
	JMP subtractIntoLoop2

subtractIntoTail:
	TESTQ CX, CX
	JE subtractIntoDone
	MOVSD 0(AX), X0
	SUBSD 0(DX), X0
	MOVSD X0, 0(BX)

subtractIntoDone:
	RET

// func addScalarIntoAMD64(source []float64, value float64, result []float64)
TEXT ·addScalarIntoAMD64(SB), NOSPLIT, $0-56
	MOVQ source_base+0(FP), AX
	MOVQ source_len+8(FP), CX
	MOVQ value+24(FP), X4
	MOVQ result_base+32(FP), BX
	SHUFPD $0, X4, X4

addScalarIntoLoop4:
	CMPQ CX, $4
	JL addScalarIntoLoop2
	MOVUPD 0(AX), X0
	MOVUPD 16(AX), X1
	ADDPD X4, X0
	ADDPD X4, X1
	MOVUPD X0, 0(BX)
	MOVUPD X1, 16(BX)
	ADDQ $32, AX
	ADDQ $32, BX
	SUBQ $4, CX
	JMP addScalarIntoLoop4

addScalarIntoLoop2:
	CMPQ CX, $2
	JL addScalarIntoTail
	MOVUPD 0(AX), X0
	ADDPD X4, X0
	MOVUPD X0, 0(BX)
	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $2, CX
	JMP addScalarIntoLoop2

addScalarIntoTail:
	TESTQ CX, CX
	JE addScalarIntoDone
	MOVSD 0(AX), X0
	ADDSD X4, X0
	MOVSD X0, 0(BX)

addScalarIntoDone:
	RET
