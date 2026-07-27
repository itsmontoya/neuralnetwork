package optimizer

import (
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_GradientClipping_UpdateValidatesCompleteParameterStructure(t *testing.T) {
	type testcase struct {
		name      string
		parameter func(testing.TB) *Parameter
		error     string
	}

	tests := []testcase{
		{
			name: "nil parameter",
			parameter: func(testing.TB) (parameter *Parameter) {
				return nil
			},
			error: "optimizer: parameter is nil",
		},
		{
			name: "nil values",
			parameter: func(tb testing.TB) (parameter *Parameter) {
				var gradient *matrix.Matrix
				var err error

				gradient, err = matrix.New(1, 1)
				if err != nil {
					tb.Fatalf("New returned error: %v", err)
				}
				parameter = &Parameter{gradient: gradient}
				return parameter
			},
			error: "invalid: values: matrix: matrix is nil",
		},
		{
			name: "invalid values",
			parameter: func(tb testing.TB) (parameter *Parameter) {
				var gradient *matrix.Matrix
				var err error

				gradient, err = matrix.New(1, 1)
				if err != nil {
					tb.Fatalf("New returned error: %v", err)
				}
				parameter = &Parameter{
					values:   &matrix.Matrix{},
					gradient: gradient,
				}
				return parameter
			},
			error: "invalid: values: matrix: dimensions must be positive",
		},
		{
			name: "nil gradient",
			parameter: func(tb testing.TB) (parameter *Parameter) {
				var values *matrix.Matrix
				var err error

				values, err = matrix.New(1, 1)
				if err != nil {
					tb.Fatalf("New returned error: %v", err)
				}
				parameter = &Parameter{values: values}
				return parameter
			},
			error: "invalid: gradient: matrix: matrix is nil",
		},
		{
			name: "invalid gradient",
			parameter: func(tb testing.TB) (parameter *Parameter) {
				var values *matrix.Matrix
				var err error

				values, err = matrix.New(1, 1)
				if err != nil {
					tb.Fatalf("New returned error: %v", err)
				}
				parameter = &Parameter{
					values:   values,
					gradient: &matrix.Matrix{},
				}
				return parameter
			},
			error: "invalid: gradient: matrix: dimensions must be positive",
		},
		{
			name: "shape mismatch",
			parameter: func(tb testing.TB) (parameter *Parameter) {
				var (
					values   *matrix.Matrix
					gradient *matrix.Matrix
					err      error
				)

				values, err = matrix.New(1, 2)
				if err != nil {
					tb.Fatalf("New values returned error: %v", err)
				}
				gradient, err = matrix.New(1, 1)
				if err != nil {
					tb.Fatalf("New gradient returned error: %v", err)
				}
				parameter = &Parameter{
					values:   values,
					gradient: gradient,
				}
				return parameter
			},
			error: "optimizer: parameter gradient shape mismatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				values        *matrix.Matrix
				first         *Parameter
				base          gradientClippingCountingOptimizer
				clipping      *GradientClipping
				gradientValue float32
				available     bool
				err           error
			)

			values, err = matrix.FromSlice(1, 1, []float32{1})
			if err != nil {
				t.Fatalf("FromSlice returned error: %v", err)
			}
			first, err = NewParameter(values)
			if err != nil {
				t.Fatalf("NewParameter returned error: %v", err)
			}
			if err = first.Gradient().CopyValuesFrom([]float32{2}); err != nil {
				t.Fatalf("CopyValuesFrom returned error: %v", err)
			}
			clipping, err = NewGradientClipping(
				&base,
				GradientClippingConfig{MaxValue: 1},
			)
			if err != nil {
				t.Fatalf("NewGradientClipping returned error: %v", err)
			}

			err = clipping.Update([]*Parameter{first, tt.parameter(t)})
			if err == nil {
				t.Fatal("Update error = nil, want error")
			}
			if !strings.Contains(err.Error(), "parameter 1") ||
				!strings.Contains(err.Error(), tt.error) {
				t.Fatalf("Update error = %q, want parameter index and %q", err.Error(), tt.error)
			}
			if base.updateCalls != 0 {
				t.Fatalf("base Update calls = %d, want 0", base.updateCalls)
			}
			gradientValue, err = first.Gradient().At(0, 0)
			if err != nil {
				t.Fatalf("At returned error: %v", err)
			}
			if gradientValue != 2 {
				t.Fatalf("first gradient = %g, want unchanged 2", gradientValue)
			}
			if _, available = clipping.Observation(); available {
				t.Fatal("parameter validation failure published observation")
			}
		})
	}
}

func Test_GradientClipping_DefensiveNumericValidation(t *testing.T) {
	t.Run("element count overflow", func(t *testing.T) {
		var err error
		var maxInt int

		maxInt = int(^uint(0) >> 1)
		_, err = addGradientClippingElementCount(maxInt, 1, 4)
		if err == nil || !strings.Contains(err.Error(), "parameter=4") {
			t.Fatalf("addGradientClippingElementCount error = %v, want overflow context", err)
		}
	})

	t.Run("non-finite norm", func(t *testing.T) {
		var err error

		_, err = foldGradientClippingNorm(math.MaxFloat64, math.MaxFloat64)
		if err == nil || !strings.Contains(err.Error(), "global norm is non-finite") {
			t.Fatalf("foldGradientClippingNorm error = %v, want non-finite norm", err)
		}
	})

	t.Run("invalid computed scale", func(t *testing.T) {
		var err error

		_, err = calculateGradientClippingScale(math.MaxFloat64, math.SmallestNonzeroFloat32)
		if err == nil || !strings.Contains(err.Error(), "scale is invalid") {
			t.Fatalf("calculateGradientClippingScale error = %v, want invalid scale", err)
		}
	})
}

func Test_GradientClipping_FailureDoesNotAdvanceBuiltInState(t *testing.T) {
	t.Run("Momentum", func(t *testing.T) {
		var (
			values    *matrix.Matrix
			parameter *Parameter
			base      *Momentum
			clipping  *GradientClipping
			err       error
		)

		values, err = matrix.FromSlice(1, 1, []float32{1})
		if err != nil {
			t.Fatalf("FromSlice returned error: %v", err)
		}
		parameter, err = NewParameter(values)
		if err != nil {
			t.Fatalf("NewParameter returned error: %v", err)
		}
		if err = parameter.Gradient().CopyValuesFrom([]float32{float32(math.NaN())}); err != nil {
			t.Fatalf("CopyValuesFrom returned error: %v", err)
		}
		base, err = NewMomentum(0.1)
		if err != nil {
			t.Fatalf("NewMomentum returned error: %v", err)
		}
		clipping, err = NewGradientClipping(base, GradientClippingConfig{MaxValue: 1})
		if err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		if err = clipping.Update([]*Parameter{parameter}); err == nil {
			t.Fatal("Update error = nil, want error")
		}
		if len(base.velocities) != 0 {
			t.Fatalf("Momentum velocities length = %d, want 0", len(base.velocities))
		}
	})

	t.Run("Adam", func(t *testing.T) {
		var (
			values    *matrix.Matrix
			parameter *Parameter
			base      *Adam
			clipping  *GradientClipping
			err       error
		)

		values, err = matrix.FromSlice(1, 1, []float32{1})
		if err != nil {
			t.Fatalf("FromSlice returned error: %v", err)
		}
		parameter, err = NewParameter(values)
		if err != nil {
			t.Fatalf("NewParameter returned error: %v", err)
		}
		if err = parameter.Gradient().CopyValuesFrom([]float32{float32(math.Inf(1))}); err != nil {
			t.Fatalf("CopyValuesFrom returned error: %v", err)
		}
		base, err = NewAdam(0.1)
		if err != nil {
			t.Fatalf("NewAdam returned error: %v", err)
		}
		clipping, err = NewGradientClipping(base, GradientClippingConfig{MaxNorm: 1})
		if err != nil {
			t.Fatalf("NewGradientClipping returned error: %v", err)
		}

		if err = clipping.Update([]*Parameter{parameter}); err == nil {
			t.Fatal("Update error = nil, want error")
		}
		if len(base.states) != 0 {
			t.Fatalf("Adam states length = %d, want 0", len(base.states))
		}
	})
}

func Test_GradientClipping_UpdateRevalidatesConfigurationBeforeMutation(t *testing.T) {
	var (
		values        *matrix.Matrix
		parameter     *Parameter
		base          gradientClippingCountingOptimizer
		clipping      *GradientClipping
		gradientValue float32
		available     bool
		err           error
	)

	values, err = matrix.FromSlice(1, 1, []float32{1})
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}
	parameter, err = NewParameter(values)
	if err != nil {
		t.Fatalf("NewParameter returned error: %v", err)
	}
	if err = parameter.Gradient().CopyValuesFrom([]float32{2}); err != nil {
		t.Fatalf("CopyValuesFrom returned error: %v", err)
	}
	clipping, err = NewGradientClipping(
		&base,
		GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	clipping.config = GradientClippingConfig{}

	err = clipping.Update([]*Parameter{parameter})
	if err == nil ||
		!strings.Contains(err.Error(), "requires at least one enabled limit") {
		t.Fatalf("Update error = %v, want invalid configuration", err)
	}
	if base.updateCalls != 0 {
		t.Fatalf("base Update calls = %d, want 0", base.updateCalls)
	}
	gradientValue, err = parameter.Gradient().At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}
	if gradientValue != 2 {
		t.Fatalf("gradient = %g, want unchanged 2", gradientValue)
	}
	if _, available = clipping.Observation(); available {
		t.Fatal("configuration failure published observation")
	}
}

func Test_GradientClipping_TransformationFailurePreservesObservation(t *testing.T) {
	var (
		first               *Parameter
		second              *Parameter
		firstValues         *matrix.Matrix
		secondValues        *matrix.Matrix
		parameters          []*Parameter
		base                gradientClippingCountingOptimizer
		clipping            *GradientClipping
		previousObservation GradientClippingObservation
		observation         GradientClippingObservation
		available           bool
		elementCount        int
		firstGradient       float32
		err                 error
	)

	firstValues, err = matrix.FromSlice(1, 1, []float32{1})
	if err != nil {
		t.Fatalf("first FromSlice returned error: %v", err)
	}
	secondValues, err = matrix.FromSlice(1, 1, []float32{2})
	if err != nil {
		t.Fatalf("second FromSlice returned error: %v", err)
	}
	first, err = NewParameter(firstValues)
	if err != nil {
		t.Fatalf("first NewParameter returned error: %v", err)
	}
	second, err = NewParameter(secondValues)
	if err != nil {
		t.Fatalf("second NewParameter returned error: %v", err)
	}
	parameters = []*Parameter{first, second}
	if err = first.Gradient().CopyValuesFrom([]float32{0.5}); err != nil {
		t.Fatalf("first CopyValuesFrom returned error: %v", err)
	}
	if err = second.Gradient().CopyValuesFrom([]float32{0.5}); err != nil {
		t.Fatalf("second CopyValuesFrom returned error: %v", err)
	}
	clipping, err = NewGradientClipping(
		&base,
		GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if err = clipping.Update(parameters); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	previousObservation, available = clipping.Observation()
	if !available || previousObservation.ValueClipped {
		t.Fatalf("initial Observation = %+v/%t, want successful no-op", previousObservation, available)
	}

	if err = first.Gradient().CopyValuesFrom([]float32{2}); err != nil {
		t.Fatalf("first replacement CopyValuesFrom returned error: %v", err)
	}
	if err = second.Gradient().CopyValuesFrom([]float32{3}); err != nil {
		t.Fatalf("second replacement CopyValuesFrom returned error: %v", err)
	}
	elementCount, err = validateGradientClippingParameters(parameters)
	if err != nil {
		t.Fatalf("validateGradientClippingParameters returned error: %v", err)
	}
	clipping.prepareScratch(elementCount)
	if err = clipping.snapshotGradients(parameters); err != nil {
		t.Fatalf("snapshotGradients returned error: %v", err)
	}
	if _, err = clipping.transformScratch(parameters); err != nil {
		t.Fatalf("transformScratch returned error: %v", err)
	}

	second.gradient = &matrix.Matrix{}
	err = clipping.publishGradients(parameters)
	if err == nil || !strings.Contains(err.Error(), "parameter 1 gradient transformation failed") {
		t.Fatalf("publishGradients error = %v, want second transformation failure", err)
	}
	if base.updateCalls != 1 {
		t.Fatalf("base Update calls = %d, want unchanged 1", base.updateCalls)
	}
	firstGradient, err = first.Gradient().At(0, 0)
	if err != nil {
		t.Fatalf("first gradient At returned error: %v", err)
	}
	if firstGradient != 1 {
		t.Fatalf("first gradient = %g, want transformed 1", firstGradient)
	}
	observation, available = clipping.Observation()
	if !available || observation != previousObservation {
		t.Fatalf("Observation = %+v/%t, want preserved %+v", observation, available, previousObservation)
	}
}

type gradientClippingCountingOptimizer struct {
	updateCalls int
}

func (o *gradientClippingCountingOptimizer) Update(parameters []*Parameter) (err error) {
	o.updateCalls++
	return nil
}

func (o *gradientClippingCountingOptimizer) LearningRate() (learningRate float32) {
	return 0
}

func (o *gradientClippingCountingOptimizer) SetLearningRate(learningRate float32) (err error) {
	return nil
}
