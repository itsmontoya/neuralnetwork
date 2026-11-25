package device

// BroadcastOperation identifies an FP32 binary broadcast kernel operation.
type BroadcastOperation uint32

// Keep these values in order with NNMetalBroadcastOperation.
const (
	// BroadcastAdd sums corresponding broadcast elements.
	BroadcastAdd BroadcastOperation = iota
	// BroadcastSub subtracts right from left across broadcast elements.
	BroadcastSub
	// BroadcastMul multiplies corresponding broadcast elements.
	BroadcastMul
	// BroadcastDiv divides left by right across broadcast elements.
	BroadcastDiv
)
