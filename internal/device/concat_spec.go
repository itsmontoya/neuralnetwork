package device

// ConcatSpec describes a contiguous or strided concatenation of non-quantized
// tensors along one axis.
type ConcatSpec struct {
	Outer       uint32
	Inner       uint32
	DstAxisLen  uint32
	ElemBytes   uint32
	SrcAxisLens []uint32
}
