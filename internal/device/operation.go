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
	// OperationTensorMatMul uses the experimental FP32 tensor pipeline.
	OperationTensorMatMul
	// OperationTensorMatMulRightTranspose uses the tensor pipeline with a transposed right operand.
	OperationTensorMatMulRightTranspose
	// OperationTensorMatMulFloat16 multiplies FP32 left by Float16 right with FP32 accumulation.
	OperationTensorMatMulFloat16
	// OperationTensorMatMulRightTransposeFloat16 multiplies FP32 left by transposed Float16 right with FP32 accumulation.
	OperationTensorMatMulRightTransposeFloat16
	// OperationTensorMatMulBFloat16 multiplies FP32 left by BFloat16 right with FP32 accumulation.
	OperationTensorMatMulBFloat16
	// OperationTensorMatMulRightTransposeBFloat16 multiplies FP32 left by transposed BFloat16 right with FP32 accumulation.
	OperationTensorMatMulRightTransposeBFloat16
)
