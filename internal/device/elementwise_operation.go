package device

// ElementwiseOperation identifies a contiguous FP32 tensor kernel.
type ElementwiseOperation uint32

// Keep these values in order with NNMetalTensorElementwiseOperation.
const (
	// ElementwiseAdd sums corresponding elements.
	ElementwiseAdd ElementwiseOperation = iota
	// ElementwiseMul multiplies corresponding elements.
	ElementwiseMul
	// ElementwiseSiLU applies the sigmoid linear unit.
	ElementwiseSiLU
	// ElementwiseSwiGLU multiplies a SiLU-activated input by a second input.
	ElementwiseSwiGLU
	// ElementwiseReLU applies the rectified linear unit.
	ElementwiseReLU
	// ElementwiseSigmoid applies the logistic sigmoid.
	ElementwiseSigmoid
	// ElementwiseTanh applies the hyperbolic tangent.
	ElementwiseTanh
)

func (o ElementwiseOperation) isUnary() (unary bool) {
	switch o {
	case ElementwiseSiLU, ElementwiseReLU, ElementwiseSigmoid, ElementwiseTanh:
		return true
	default:
		return false
	}
}
