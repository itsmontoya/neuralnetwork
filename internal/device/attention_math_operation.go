package device

// AttentionMathOperation selects a tensor-only FP32 kernel.
type AttentionMathOperation uint32

const (
	TensorRMSNorm AttentionMathOperation = iota
	TensorRotatePairs
	TensorCausalAttention
)
