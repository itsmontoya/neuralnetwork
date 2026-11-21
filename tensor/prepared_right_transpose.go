package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
)

// PrepareRightTranspose copies a row-major [outputs, contraction] weight.
// Eligible host ARM64 weights are stored in eight-output tiles, with the
// weights for each contraction index adjacent. Other builds and shapes use
// dense storage. The caller may reuse the input slices after this returns.
func PrepareRightTranspose(shape []int64, values []float32) (out *PreparedRightTranspose, err error) {
	var (
		elements int64
		weight   PreparedRightTranspose
	)

	if len(shape) != 2 {
		return nil, fmt.Errorf("tensor: prepared right-transpose weight requires rank two")
	}
	if elements, _, err = validateShapeSize(shape, Float32); err != nil {
		return nil, err
	}
	if int64(len(values)) != elements {
		return nil, fmt.Errorf("tensor: float32 values length mismatch: got %d, want %d", len(values), elements)
	}
	weight.storage = newTensor(Float32, shape, elements)
	weight.packed = hostMatrixProduct && float32product.CanPackRightTranspose(int(shape[1]), int(shape[0]))
	if weight.packed {
		float32product.PackRightTranspose(values, weight.storage.floats, int(shape[1]), int(shape[0]))
	} else {
		copy(weight.storage.floats, values)
	}
	return &weight, nil
}

// PreparedRightTranspose owns immutable FP32 projection weights. It can be
// shared across concurrent GemmInto calls with disjoint destinations.
// Its zero value is invalid.
type PreparedRightTranspose struct {
	storage *Tensor
	packed  bool
}

// Shape returns the copied logical [outputs, contraction] shape.
func (w *PreparedRightTranspose) Shape() (shape []int64) {
	if w == nil {
		return nil
	}
	return w.storage.Shape()
}

// Dense returns an independently owned tensor in logical row-major order.
func (w *PreparedRightTranspose) Dense() (out *Tensor, err error) {
	if w == nil {
		return nil, fmt.Errorf("tensor: prepared right-transpose weight is nil")
	}
	if err = w.storage.Validate(); err != nil {
		return nil, err
	}
	if !w.packed {
		return w.storage.Clone()
	}
	out = newTensor(Float32, w.storage.shape, int64(len(w.storage.floats)))
	float32product.UnpackRightTranspose(w.storage.floats, out.floats, int(out.shape[1]), 0)
	return out, nil
}

// GemmInto applies the same shape, bias, and destination rules as GemmInto.
// A right-only transpose uses the prepared tiles directly. Other transpose
// configurations use a temporary dense copy when the weight is packed.
func (w *PreparedRightTranspose) GemmInto(left, bias, destination *Tensor, config GemmConfig) (err error) {
	if w == nil {
		return fmt.Errorf("tensor: prepared right-transpose weight is nil")
	}
	return gemmInto(left, w.storage, bias, destination, config, w.packed)
}
