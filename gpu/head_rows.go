package gpu

import "math"

// headRows checks equally packed [rows, heads*width] and [rows, heads, width]
// layouts. Both store features contiguously, then heads, then tokens.
func headRows(tensor *Tensor, width int64) (rows, heads int64) {
	if tensor == nil || tensor.state == nil || tensor.state.storage.count > math.MaxUint32 || width <= 0 {
		return 0, 0
	}
	var shape = tensor.state.shape
	switch len(shape) {
	case 2:
		if shape[1]%width == 0 {
			return shape[0], shape[1] / width
		}
	case 3:
		if shape[2] == width {
			return shape[0], shape[1]
		}
	}
	return 0, 0
}

func hasShape(tensor *Tensor, shape ...int64) (ok bool) {
	if tensor == nil || tensor.state == nil || len(tensor.state.shape) != len(shape) {
		return false
	}
	for axis, size := range shape {
		if tensor.state.shape[axis] != size {
			return false
		}
	}
	return true
}
