package device

// Pool2DSpec describes the checked 32-bit NCHW pooling geometry for Metal
// dispatch. All spatial and stride/dilation/padding fields must fit within
// int32 so signed sampling coordinate arithmetic in Metal cannot overflow.
type Pool2DSpec struct {
	ChannelsTotal   uint32
	InH             uint32
	InW             uint32
	OutH            uint32
	OutW            uint32
	KernelH         uint32
	KernelW         uint32
	StrideH         uint32
	StrideW         uint32
	PadTop          uint32
	PadLeft         uint32
	DilationH       uint32
	DilationW       uint32
	CountIncludePad bool
	Average         bool
}
