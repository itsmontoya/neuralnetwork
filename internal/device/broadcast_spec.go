package device

// BroadcastSpec describes a coalesced row-major multidirectional broadcast up
// to rank 6. Zero input strides denote broadcast axes.
type BroadcastSpec struct {
	Rank         uint32
	Count        uint32
	Dims         [6]uint32
	OutStrides   [6]uint32
	LeftStrides  [6]uint32
	RightStrides [6]uint32
}
