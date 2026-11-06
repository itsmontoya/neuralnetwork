package tensor

import "fmt"

// New constructs a zero-filled Tensor with a copied concrete shape.
// Invalid dimensions, element counts, or byte sizes return an error before
// payload allocation. Representable sizes remain subject to available memory.
func New(dataType DataType, shape []int64) (out *Tensor, err error) {
	var elements int64

	if elements, _, err = validateShapeSize(shape, dataType); err != nil {
		return nil, err
	}
	out = newTensor(dataType, shape, elements)
	return out, nil
}

// FromFloat32 constructs a Tensor by copying shape and row-major values.
// The value count must exactly match the shape, including scalar/empty shapes.
func FromFloat32(shape []int64, values []float32) (out *Tensor, err error) {
	var elements int64

	if elements, _, err = validateShapeSize(shape, Float32); err != nil {
		return nil, err
	}
	if int64(len(values)) != elements {
		err = fmt.Errorf("tensor: float32 values length mismatch: got %d, want %d", len(values), elements)
		return nil, err
	}
	out = newTensor(Float32, shape, elements)
	copy(out.floats, values)
	return out, nil
}

// FromInt64 constructs a Tensor by copying shape and row-major integer values.
// Integer values retain all 64 bits and never pass through floating storage.
func FromInt64(shape []int64, values []int64) (out *Tensor, err error) {
	var elements int64

	if elements, _, err = validateShapeSize(shape, Int64); err != nil {
		return nil, err
	}
	if int64(len(values)) != elements {
		err = fmt.Errorf("tensor: int64 values length mismatch: got %d, want %d", len(values), elements)
		return nil, err
	}
	out = newTensor(Int64, shape, elements)
	copy(out.integers, values)
	return out, nil
}

func newTensor(dataType DataType, shape []int64, elements int64) (out *Tensor) {
	var t Tensor

	t.dataType = dataType
	t.shape = make([]int64, len(shape))
	copy(t.shape, shape)
	t.identity = new(byte)
	if dataType == Float32 {
		t.floats = make([]float32, int(elements))
	} else {
		t.integers = make([]int64, int(elements))
	}
	return &t
}

// Tensor owns contiguous row-major numeric storage and a concrete shape.
// Its zero value is invalid. A nil receiver is also invalid; neither represents
// a scalar or an empty tensor. Use Clone instead of a shallow copy to own data
// independently. Shapes and value accessors always return copies.
type Tensor struct {
	dataType DataType
	shape    []int64
	floats   []float32
	integers []int64
	identity *byte
}

// Validate reports invalid receivers, shapes, types, or storage lengths.
func (t *Tensor) Validate() (err error) {
	var elements int64

	if t == nil {
		err = fmt.Errorf("tensor: value is nil")
		return err
	}
	if t.identity == nil {
		err = fmt.Errorf("tensor: value is uninitialized")
		return err
	}
	if elements, _, err = validateShapeSize(t.shape, t.dataType); err != nil {
		return err
	}
	switch t.dataType {
	case Float32:
		if int64(len(t.floats)) != elements || t.integers != nil {
			err = fmt.Errorf("tensor: float32 storage does not match shape and data type")
			return err
		}
	case Int64:
		if int64(len(t.integers)) != elements || t.floats != nil {
			err = fmt.Errorf("tensor: int64 storage does not match shape and data type")
			return err
		}
	}
	return nil
}

// DataType returns Invalid for a nil or invalid receiver.
func (t *Tensor) DataType() (dataType DataType) {
	if t.Validate() != nil {
		return Invalid
	}
	return t.dataType
}

// Shape returns a copied shape, or nil for a nil or invalid receiver.
func (t *Tensor) Shape() (shape []int64) {
	if t.Validate() != nil {
		return nil
	}
	shape = make([]int64, len(t.shape))
	copy(shape, t.shape)
	return shape
}

// Rank returns the dimension count, or zero for a nil or invalid receiver.
func (t *Tensor) Rank() (rank int) {
	if t.Validate() != nil {
		return 0
	}
	return len(t.shape)
}

// Len returns the element count, or zero for a nil or invalid receiver.
func (t *Tensor) Len() (elements int64) {
	if t.Validate() != nil {
		return 0
	}
	if t.dataType == Float32 {
		return int64(len(t.floats))
	}
	return int64(len(t.integers))
}

// ByteSize returns the payload byte count, or zero for a nil or invalid receiver.
// It excludes shape metadata and allocation overhead.
func (t *Tensor) ByteSize() (bytes int64) {
	if t.Validate() != nil {
		return 0
	}
	if t.dataType == Float32 {
		return int64(len(t.floats)) * 4
	}
	return int64(len(t.integers)) * 8
}

// Float32Values returns a copy of float32 values, rejecting other data types.
func (t *Tensor) Float32Values() (values []float32, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}
	if t.dataType != Float32 {
		err = fmt.Errorf("tensor: float32 values requested from data type %d", t.dataType)
		return nil, err
	}
	values = make([]float32, len(t.floats))
	copy(values, t.floats)
	return values, nil
}

// Int64Values returns a copy of integer values, rejecting other data types.
func (t *Tensor) Int64Values() (values []int64, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}
	if t.dataType != Int64 {
		err = fmt.Errorf("tensor: int64 values requested from data type %d", t.dataType)
		return nil, err
	}
	values = make([]int64, len(t.integers))
	copy(values, t.integers)
	return values, nil
}

// Clone returns independent copies of the shape and numeric storage.
func (t *Tensor) Clone() (out *Tensor, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}
	if t.dataType == Float32 {
		out, err = FromFloat32(t.shape, t.floats)
		return out, err
	}
	out, err = FromInt64(t.shape, t.integers)
	return out, err
}
