package optimizer

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_Adam_Update_ValidatesParameters(t *testing.T) {
	var (
		adam *Adam
		err  error
	)

	adam, err = NewAdam(0.001)
	if err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	err = adam.Update([]*Parameter{nil})
	if err == nil {
		t.Fatal("Update error = nil, want error")
	}
}

func Test_Adam_Update_ValidatesStateShape(t *testing.T) {
	var (
		values         *matrix.Matrix
		gradient       *matrix.Matrix
		firstMoment    *matrix.Matrix
		secondMoment   *matrix.Matrix
		parameter      *Parameter
		adam           *Adam
		gradientValues []float64
		err            error
	)

	values, err = matrix.FromSlice(1, 2, []float64{1, 2})
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	gradient, err = matrix.FromSlice(1, 2, []float64{0.1, 0.2})
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	parameter, err = NewParameter(values)
	if err != nil {
		t.Fatalf("NewParameter returned error: %v", err)
	}

	if err = parameter.AccumulateGradient(gradient); err != nil {
		t.Fatalf("AccumulateGradient returned error: %v", err)
	}

	firstMoment, err = matrix.New(1, 1)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	secondMoment, err = matrix.New(1, 2)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	adam, err = NewAdam(0.001)
	if err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	adam.states[parameter] = &adamState{
		firstMoment:  firstMoment,
		secondMoment: secondMoment,
	}

	err = adam.Update([]*Parameter{parameter})
	if err == nil {
		t.Fatal("Update error = nil, want error")
	}

	if !strings.Contains(err.Error(), "optimizer: adam state shape mismatch") {
		t.Fatalf("Update error = %q, want adam state shape mismatch", err.Error())
	}

	gradientValues, err = parameter.Gradient().Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	if gradientValues[0] != 0.1 || gradientValues[1] != 0.2 {
		t.Fatalf("gradient reset after failed update: got %v", gradientValues)
	}
}
