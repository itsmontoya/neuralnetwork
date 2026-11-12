package tensor

// broadcastAddFloat32Into requires validated, nonempty, disjoint tensors.
func broadcastAddFloat32Into(left, right, destination *Tensor) (err error) {
	var (
		span, axes                             = scalarBroadcastSpan(left.shape, right.shape, destination.shape)
		scalarLeft                             bool
		coordinates, leftStrides, rightStrides []int64
		leftBuffer, rightBuffer                [broadcastStackRank]int64
		leftOffset, rightOffset                int64
	)
	if span == 1 {
		span, axes = scalarBroadcastSpan(right.shape, left.shape, destination.shape)
		scalarLeft = true
	}
	if span == 1 {
		return broadcastFloat32Into(left, right, destination, addValue)
	}
	// Plan both inputs completely before writing any output.
	if leftStrides, err = broadcastStridesInto(left.shape, destination.shape, leftBuffer[:]); err != nil {
		return err
	}
	if rightStrides, err = broadcastStridesInto(right.shape, destination.shape, rightBuffer[:]); err != nil {
		return err
	}
	coordinates = make([]int64, axes)
	for start := 0; start < len(destination.floats); start += span {
		var output = destination.floats[start : start+span]
		if scalarLeft {
			var (
				scalar = left.floats[int(leftOffset)]
				values = right.floats[int(rightOffset) : int(rightOffset)+span]
			)
			for index, value := range values {
				output[index] = scalar + value
			}
		} else {
			var (
				values = left.floats[int(leftOffset) : int(leftOffset)+span]
				scalar = right.floats[int(rightOffset)]
			)
			for index, value := range values {
				output[index] = value + scalar
			}
		}
		// Only the prefix carries: the suffix was consumed as one contiguous span.
		for axis := axes - 1; axis >= 0; axis-- {
			coordinates[axis]++
			if coordinates[axis] < destination.shape[axis] {
				leftOffset += leftStrides[axis]
				rightOffset += rightStrides[axis]
				break
			}
			coordinates[axis] = 0
			leftOffset -= (destination.shape[axis] - 1) * leftStrides[axis]
			rightOffset -= (destination.shape[axis] - 1) * rightStrides[axis]
		}
	}
	return nil
}

// scalarBroadcastSpan finds a suffix contiguous in values and constant in scalar.
// Shapes must already describe a validated, nonempty broadcast. The suffix size
// is bounded by the validated output length; singleton axes need no source stride.
func scalarBroadcastSpan(values, scalar, output []int64) (span, axes int) {
	var (
		axis, valuesAxis, scalarAxis int
		dimension                    int64
	)
	span = 1
	for axis = len(output) - 1; axis >= 0; axis-- {
		dimension = output[axis]
		if dimension == 1 {
			continue
		}
		valuesAxis = axis - (len(output) - len(values))
		scalarAxis = axis - (len(output) - len(scalar))
		if scalarAxis >= 0 && scalar[scalarAxis] != 1 {
			break
		}
		if valuesAxis < 0 || values[valuesAxis] != dimension {
			break
		}
		span *= int(dimension)
	}
	return span, axis + 1
}
