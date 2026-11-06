package tensor

// UnsqueezeShape returns an owned shape with unit axes inserted at final-rank
// positions. Negative axes normalize against that output rank. Duplicates fail;
// empty axes mean identity. Both types' sizes are checked without payloads.
func UnsqueezeShape(input, axes []int64) (shape []int64, err error) {
	var (
		rank       int64
		normalized []int
		inserted   []bool
		axis       int
		inputAxis  int
	)

	if _, _, err = ValidateShape(input, Int64); err != nil {
		return nil, err
	}
	if rank, err = checkedAdd(int64(len(input)), int64(len(axes))); err != nil {
		return nil, err
	}
	if normalized, err = normalizeAxes(axes, int(rank)); err != nil {
		return nil, err
	}
	inserted = make([]bool, int(rank))
	for _, axis = range normalized {
		inserted[axis] = true
	}
	shape = make([]int64, int(rank))
	for axis = range shape {
		if inserted[axis] {
			shape[axis] = 1
			continue
		}
		shape[axis] = input[inputAxis]
		inputAxis++
	}
	return shape, nil
}

// UnsqueezeInto copies values with unit axes inserted at final-rank positions.
// Both types are preserved. Storage must be disjoint; errors preserve destination.
func UnsqueezeInto(input, destination *Tensor, axes []int64) (err error) {
	var shape []int64

	if err = validateTransformInput(input); err != nil {
		return err
	}
	if shape, err = UnsqueezeShape(input.shape, axes); err != nil {
		return err
	}
	err = copyTransformInto(input, destination, shape)
	return err
}
