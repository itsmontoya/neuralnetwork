package device

// CategoricalCrossEntropyResultCount is the private scalar and diagnostic buffer length.
const CategoricalCrossEntropyResultCount = 5

type executionValidation struct {
	key      any
	revision uint64
}
