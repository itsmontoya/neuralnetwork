package device

// Conv2DSpec describes the checked 32-bit NCHW 2D cross-correlation geometry
// and operand dtypes for Metal dispatch. Spatial, kernel, stride, pad, and
// dilation fields must fit within int32 so signed sampling coordinate
// arithmetic in Metal cannot overflow.
type Conv2DSpec struct {
	Batch       uint32
	InChannels  uint32
	InH         uint32
	InW         uint32
	OutChannels uint32
	OutH        uint32
	OutW        uint32
	KernelH     uint32
	KernelW     uint32
	StrideH     uint32
	StrideW     uint32
	PadTop      uint32
	PadLeft     uint32
	DilationH   uint32
	DilationW   uint32
	Groups      uint32
	WeightDType uint32
	BiasDType   uint32
	HasBias     bool
}
