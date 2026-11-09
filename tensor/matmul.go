package tensor

// MatMulShape returns an owned float32 product shape with vector promotion and
// broadcast batch prefixes. Scalars and unequal contraction dimensions fail.
func MatMulShape(left, right []int64) (shape []int64, err error) {
	var plan *matMulPlan

	if plan, err = newMatMulPlan(left, right); err != nil {
		return nil, err
	}
	return plan.shape, nil
}

// MatMulInto writes a float32 product into a disjoint destination.
// Vector-vector products are scalars; batch prefixes broadcast. Empty
// contractions write positive zeros. Matrix blocks accumulate in float32.
// All returned errors leave destination unchanged.
func MatMulInto(left, right, destination *Tensor) (err error) {
	var plan *matMulPlan

	if err = validateTensorType("left", left, Float32); err != nil {
		return err
	}
	if err = validateTensorType("right", right, Float32); err != nil {
		return err
	}
	if plan, err = newMatMulPlan(left.shape, right.shape); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, plan.shape, left, right); err != nil {
		return err
	}
	if len(destination.floats) == 0 {
		return nil
	}
	if plan.inner == 0 {
		clear(destination.floats)
		return nil
	}
	err = matMulBatchesInto(left, right, destination, plan)
	return err
}

func matMulBatchesInto(left, right, destination *Tensor, plan *matMulPlan) (err error) {
	var (
		leftBatch, rightBatch, batchStrides []int64
		leftStrides, rightStrides           []int64
		leftBlock, rightBlock, outputBlock  int64
		leftOffset, rightOffset             int64
		coordinate                          int64
		batch, batchCount, axis             int
		output                              []float32
	)

	if len(left.shape) > 1 {
		leftBatch = left.shape[:len(left.shape)-2]
	}
	if len(right.shape) > 1 {
		rightBatch = right.shape[:len(right.shape)-2]
	}
	if leftBlock, err = checkedMultiply(plan.rows, plan.inner); err != nil {
		return err
	}
	if rightBlock, err = checkedMultiply(plan.inner, plan.cols); err != nil {
		return err
	}
	if outputBlock, err = checkedMultiply(plan.rows, plan.cols); err != nil {
		return err
	}
	if batchStrides, err = contiguousStrides(plan.batch); err != nil {
		return err
	}
	if leftStrides, err = broadcastStrides(leftBatch, plan.batch); err != nil {
		return err
	}
	if rightStrides, err = broadcastStrides(rightBatch, plan.batch); err != nil {
		return err
	}
	output = make([]float32, len(destination.floats))
	batchCount = len(output) / int(outputBlock)
	for batch = 0; batch < batchCount; batch++ {
		leftOffset, rightOffset = 0, 0
		for axis = range plan.batch {
			coordinate = int64(batch) / batchStrides[axis] % plan.batch[axis]
			leftOffset += coordinate * leftStrides[axis]
			rightOffset += coordinate * rightStrides[axis]
		}
		// Broadcast coordinates select an existing block. Offsets, block ends,
		// and host conversions are bounded by the validated input lengths.
		leftOffset *= leftBlock
		rightOffset *= rightBlock
		if err = matrixProductInto(
			left.floats[int(leftOffset):int(leftOffset+leftBlock)],
			right.floats[int(rightOffset):int(rightOffset+rightBlock)],
			[]int64{plan.rows, plan.inner}, []int64{plan.inner, plan.cols}, false, false,
			output[batch*int(outputBlock):(batch+1)*int(outputBlock)],
		); err != nil {
			return err
		}
	}
	copy(destination.floats, output)
	return nil
}
