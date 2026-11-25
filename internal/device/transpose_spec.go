package device

// TransposeSpec describes a coalesced row-major axis permutation up to rank 6.
type TransposeSpec struct {
	Rank       uint32
	Count      uint32
	ElemBytes  uint32
	Dims       [6]uint32
	OutStrides [6]uint32
	InStrides  [6]uint32
}
