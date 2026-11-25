package device

// BatchedMatMulSpec describes a rank 2..6 broadcast-batched matrix
// multiplication with up to 4 batch dimensions.
type BatchedMatMulSpec struct {
	M                 uint32
	N                 uint32
	K                 uint32
	TotalBatches      uint32
	BatchRank         uint32
	RightTransposed   bool
	RightDType        uint32
	OutBatchStrides   [4]uint32
	LeftBatchStrides  [4]uint32
	RightBatchStrides [4]uint32
}
