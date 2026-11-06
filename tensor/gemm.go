package tensor

import "fmt"

// GemmShape returns the checked rank-two float32 product shape after explicit
// transposition. Bias broadcasting is checked separately by ValidateBroadcastTo.
func GemmShape(left, right []int64, config GemmConfig) (shape []int64, err error) {
	var (
		leftShape, rightShape [2]int64
	)

	if len(left) != 2 || len(right) != 2 {
		err = fmt.Errorf("tensor: Gemm requires rank-two inputs")
		return nil, err
	}
	copy(leftShape[:], left)
	copy(rightShape[:], right)
	if config.TransposeA {
		leftShape[0], leftShape[1] = leftShape[1], leftShape[0]
	}
	if config.TransposeB {
		rightShape[0], rightShape[1] = rightShape[1], rightShape[0]
	}
	shape, err = MatMulShape(leftShape[:], rightShape[:])
	return shape, err
}

// GemmInto writes Alpha*op(left)*op(right) plus Beta*bias into destination.
// Nil bias omits the term. Zero Beta skips its arithmetic but still validates
// supplied bias. Coefficients have no defaults; nonfinite values follow float32
// arithmetic. All returned errors leave the disjoint destination unchanged.
func GemmInto(left, right, bias, destination *Tensor, config GemmConfig) (err error) {
	var (
		shape, biasStrides []int64
		values             []float32
		inner, biasOffset  int64
		index              int
	)

	if err = validateTensorType("left", left, Float32); err != nil {
		return err
	}
	if err = validateTensorType("right", right, Float32); err != nil {
		return err
	}
	if shape, err = GemmShape(left.shape, right.shape, config); err != nil {
		return err
	}
	if bias != nil {
		if err = validateTensorType("bias", bias, Float32); err != nil {
			return err
		}
		if err = ValidateBroadcastTo(bias.shape, shape); err != nil {
			return fmt.Errorf("tensor: Gemm bias shape is invalid: %w", err)
		}
	}
	if err = validateDestination(destination, Float32, shape, left, right, bias); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if bias != nil && config.Beta != 0 {
		if biasStrides, err = broadcastStrides(bias.shape, shape); err != nil {
			return err
		}
	}
	inner = left.shape[1]
	if config.TransposeA {
		inner = left.shape[0]
	}
	if inner == 0 {
		values = make([]float32, len(destination.floats))
	} else if values, err = matrixProductValues(left.floats, right.floats, left.shape, right.shape, config.TransposeA, config.TransposeB); err != nil {
		return err
	}
	for index = range values {
		values[index] *= config.Alpha
		if biasStrides != nil {
			// Rank-two output coordinates and broadcast strides bound this
			// offset by the already validated bias payload length.
			biasOffset = (int64(index)/shape[1])*biasStrides[0] + (int64(index)%shape[1])*biasStrides[1]
			values[index] += config.Beta * bias.floats[int(biasOffset)]
		}
	}
	copy(destination.floats, values)
	return nil
}
