package device

// AttentionMathOperation selects a tensor-only attention or normalization kernel.
type AttentionMathOperation uint32

const (
	TensorRMSNorm AttentionMathOperation = iota
	TensorRMSNormFloat16
	TensorRMSNormBFloat16
	TensorRotatePairs
	TensorCausalAttention
	TensorCausalAttentionFloat16
)
