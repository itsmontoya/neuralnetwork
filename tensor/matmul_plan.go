package tensor

import "fmt"

func newMatMulPlan(left, right []int64) (out *matMulPlan, err error) {
	var (
		plan                  matMulPlan
		leftBatch, rightBatch []int64
		rightInner            int64
	)

	if _, _, err = ValidateShape(left, Float32); err != nil {
		return nil, fmt.Errorf("tensor: left MatMul shape is invalid: %w", err)
	}
	if _, _, err = ValidateShape(right, Float32); err != nil {
		return nil, fmt.Errorf("tensor: right MatMul shape is invalid: %w", err)
	}
	if len(left) == 0 || len(right) == 0 {
		err = fmt.Errorf("tensor: MatMul requires inputs of rank at least one")
		return nil, err
	}
	plan.rows, plan.cols = 1, 1
	plan.inner = left[len(left)-1]
	rightInner = right[0]
	if len(left) > 1 {
		plan.rows = left[len(left)-2]
		leftBatch = left[:len(left)-2]
	}
	if len(right) > 1 {
		rightInner = right[len(right)-2]
		plan.cols = right[len(right)-1]
		rightBatch = right[:len(right)-2]
	}
	if plan.inner != rightInner {
		err = fmt.Errorf("tensor: MatMul contraction dimensions differ: left=%d right=%d", plan.inner, rightInner)
		return nil, err
	}
	if plan.batch, err = broadcastDimensions(leftBatch, rightBatch); err != nil {
		return nil, fmt.Errorf("tensor: MatMul batch shapes are incompatible: %w", err)
	}
	plan.shape = make([]int64, len(plan.batch), len(plan.batch)+2)
	copy(plan.shape, plan.batch)
	if len(left) > 1 {
		plan.shape = append(plan.shape, plan.rows)
	}
	if len(right) > 1 {
		plan.shape = append(plan.shape, plan.cols)
	}
	if _, _, err = ValidateShape(plan.shape, Float32); err != nil {
		return nil, fmt.Errorf("tensor: MatMul result shape is invalid: %w", err)
	}
	return &plan, nil
}

// matMulPlan contains checked dimensions, with vector promotion kept private.
type matMulPlan struct {
	batch []int64
	shape []int64
	rows  int64
	inner int64
	cols  int64
}
