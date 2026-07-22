package device

// Operation identifies a private device kernel.
type Operation uint32

const (
	// OperationMatMul computes a standard matrix multiplication.
	OperationMatMul Operation = iota
	// OperationMatMulLeftTranspose computes a multiplication with a transposed left operand.
	OperationMatMulLeftTranspose
	// OperationMatMulRightTranspose computes a multiplication with a transposed right operand.
	OperationMatMulRightTranspose
)
