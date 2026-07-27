package optimizer_test

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_GradientClipping_ImplementsOptimizer(t *testing.T) {
	var _ optimizer.Optimizer = (*optimizer.GradientClipping)(nil)
}

func Test_NewGradientClipping_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name   string
		base   optimizer.Optimizer
		config optimizer.GradientClippingConfig
		error  string
	}

	var base mockOptimizer
	tests := []testcase{
		{
			name:   "nil base",
			config: optimizer.GradientClippingConfig{MaxValue: 1},
			error:  "optimizer: base optimizer is nil",
		},
		{
			name:  "both controls disabled",
			base:  &base,
			error: "optimizer: gradient clipping requires at least one enabled limit",
		},
		{
			name:   "negative max value",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxValue: -1},
			error:  "max value must be positive and finite",
		},
		{
			name:   "max value NaN",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxValue: float32(math.NaN())},
			error:  "max value must be positive and finite",
		},
		{
			name:   "max value positive infinity",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxValue: float32(math.Inf(1))},
			error:  "max value must be positive and finite",
		},
		{
			name:   "max value negative infinity",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxValue: float32(math.Inf(-1))},
			error:  "max value must be positive and finite",
		},
		{
			name:   "negative max norm",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxNorm: -1},
			error:  "max norm must be positive and finite",
		},
		{
			name:   "max norm NaN",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxNorm: float32(math.NaN())},
			error:  "max norm must be positive and finite",
		},
		{
			name:   "max norm positive infinity",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxNorm: float32(math.Inf(1))},
			error:  "max norm must be positive and finite",
		},
		{
			name:   "max norm negative infinity",
			base:   &base,
			config: optimizer.GradientClippingConfig{MaxNorm: float32(math.Inf(-1))},
			error:  "max norm must be positive and finite",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				clipping *optimizer.GradientClipping
				err      error
			)

			clipping, err = optimizer.NewGradientClipping(tt.base, tt.config)
			if err == nil {
				t.Fatal("NewGradientClipping error = nil, want error")
			}
			if !strings.Contains(err.Error(), tt.error) {
				t.Fatalf("NewGradientClipping error = %q, want context %q", err.Error(), tt.error)
			}
			if clipping != nil {
				t.Fatal("NewGradientClipping returned optimizer on error")
			}
		})
	}
}

func Test_NewGradientClipping_AcceptsEnabledControlsAndCopiesConfig(t *testing.T) {
	tests := []struct {
		name   string
		config optimizer.GradientClippingConfig
	}{
		{
			name:   "value only",
			config: optimizer.GradientClippingConfig{MaxValue: 1},
		},
		{
			name:   "norm only",
			config: optimizer.GradientClippingConfig{MaxNorm: 2},
		},
		{
			name:   "combined",
			config: optimizer.GradientClippingConfig{MaxValue: 1, MaxNorm: 2},
		},
		{
			name:   "positive subnormal",
			config: optimizer.GradientClippingConfig{MaxNorm: math.SmallestNonzeroFloat32},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				base        mockOptimizer
				config      optimizer.GradientClippingConfig
				clipping    *optimizer.GradientClipping
				observation optimizer.GradientClippingObservation
				available   bool
				err         error
			)

			config = tt.config
			clipping, err = optimizer.NewGradientClipping(&base, config)
			if err != nil {
				t.Fatalf("NewGradientClipping returned error: %v", err)
			}

			config.MaxValue = 99
			config.MaxNorm = 99
			if clipping.Config() != tt.config {
				t.Fatalf("Config = %+v, want %+v", clipping.Config(), tt.config)
			}

			config = clipping.Config()
			config.MaxValue = 77
			if clipping.Config() != tt.config {
				t.Fatal("Config returned mutable wrapper state")
			}
			if clipping.Base() != &base {
				t.Fatal("Base did not return wrapped optimizer")
			}

			observation, available = clipping.Observation()
			if available {
				t.Fatal("Observation available before Update")
			}
			if observation != (optimizer.GradientClippingObservation{}) {
				t.Fatalf("Observation = %+v, want zero value", observation)
			}
		})
	}
}

func Test_GradientClipping_InvalidReceivers(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var clipping *optimizer.GradientClipping
		var err error

		err = clipping.Update(nil)
		if err == nil ||
			err.Error() != "optimizer: gradient clipping optimizer is nil" {
			t.Fatalf("Update error = %v, want nil gradient clipping optimizer", err)
		}
	})

	t.Run("zero value", func(t *testing.T) {
		var clipping optimizer.GradientClipping
		var err error

		err = clipping.Update(nil)
		if err == nil || err.Error() != "optimizer: base optimizer is nil" {
			t.Fatalf("Update error = %v, want nil base optimizer", err)
		}
		err = clipping.SetLearningRate(0.1)
		if err == nil || err.Error() != "optimizer: base optimizer is nil" {
			t.Fatalf("SetLearningRate error = %v, want nil base optimizer", err)
		}
	})
}

func Test_NewGradientClipping_AcceptsTypedNilBase(t *testing.T) {
	var (
		typedNilBase *optimizer.SGD
		base         optimizer.Optimizer
		parameter    *optimizer.Parameter
		clipping     *optimizer.GradientClipping
		observation  optimizer.GradientClippingObservation
		available    bool
		err          error
	)

	base = typedNilBase
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	parameter = mustParameter(t, 1, 1, []float32{1})
	accumulateGradient(t, parameter, []float32{2})

	err = clipping.Update([]*optimizer.Parameter{parameter})
	if err == nil || err.Error() != "optimizer: sgd optimizer is nil" {
		t.Fatalf("Update error = %v, want typed nil base error", err)
	}
	requireMatrixValues(t, parameter.Gradient(), []float32{1})
	observation, available = clipping.Observation()
	if !available || !observation.ValueClipped || observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v/%t, want failed typed nil base", observation, available)
	}
}

func Test_GradientClipping_ValueArithmeticAndDelegation(t *testing.T) {
	var (
		first       *optimizer.Parameter
		second      *optimizer.Parameter
		parameters  []*optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	first = mustParameter(t, 2, 4, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	second = mustParameter(t, 1, 2, []float32{9, 10})
	err = first.Gradient().CopyValuesFrom(
		[]float32{-2, -1, float32(math.Copysign(0, -1)), 0, 0.5, 1, 2, 3},
	)
	if err != nil {
		t.Fatalf("CopyValuesFrom returned error: %v", err)
	}
	accumulateGradient(t, second, []float32{-3, 0.25})
	parameters = []*optimizer.Parameter{first, second}

	base = &mockOptimizer{
		updateFunc: func(received []*optimizer.Parameter) (err error) {
			var (
				values      []float32
				observation optimizer.GradientClippingObservation
				available   bool
			)

			if len(received) != len(parameters) || &received[0] != &parameters[0] {
				t.Fatal("base did not receive the original parameter slice")
			}
			requireMatrixValues(t, received[0].Gradient(), []float32{-1, -1, 0, 0, 0.5, 1, 1, 1})
			requireMatrixValues(t, received[1].Gradient(), []float32{-1, 0.25})
			values, err = received[0].Gradient().Values()
			if err != nil {
				return err
			}
			if !math.Signbit(float64(values[2])) {
				t.Fatal("value clipping did not preserve negative zero")
			}

			observation, available = clipping.Observation()
			if available || observation != (optimizer.GradientClippingObservation{}) {
				t.Fatal("observation published before base returned")
			}
			for _, receivedParameter := range received {
				if err = receivedParameter.ResetGradient(); err != nil {
					return err
				}
			}
			return nil
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = clipping.Update(parameters); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	if base.updateCalls != 1 {
		t.Fatalf("base Update calls = %d, want 1", base.updateCalls)
	}
	requireMatrixValues(t, first.Gradient(), []float32{0, 0, 0, 0, 0, 0, 0, 0})
	requireMatrixValues(t, second.Gradient(), []float32{0, 0})

	observation, available = clipping.Observation()
	if !available {
		t.Fatal("Observation unavailable after Update")
	}
	if !observation.ValueClipped ||
		observation.GlobalNorm != 0 ||
		observation.Scale != 1 ||
		!observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v, want successful value clipping", observation)
	}
}

func Test_GradientClipping_GlobalNormArithmetic(t *testing.T) {
	var (
		first       *optimizer.Parameter
		second      *optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	first = mustParameter(t, 1, 2, []float32{0, 0})
	second = mustParameter(t, 1, 2, []float32{0, 0})
	accumulateGradient(t, first, []float32{3, 4})
	accumulateGradient(t, second, []float32{0, 12})

	base = &mockOptimizer{
		updateFunc: func(parameters []*optimizer.Parameter) (err error) {
			requireMatrixValues(t, parameters[0].Gradient(), []float32{1.5, 2})
			requireMatrixValues(t, parameters[1].Gradient(), []float32{0, 6})
			for _, parameter := range parameters {
				if err = parameter.ResetGradient(); err != nil {
					return err
				}
			}
			return nil
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: 6.5},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = clipping.Update([]*optimizer.Parameter{first, second}); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	observation, available = clipping.Observation()
	if !available {
		t.Fatal("Observation unavailable after Update")
	}
	if observation.ValueClipped ||
		observation.GlobalNorm != 13 ||
		observation.Scale != 0.5 ||
		!observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v, want global norm 13 and scale 0.5", observation)
	}
}

func Test_GradientClipping_NormNoOpCases(t *testing.T) {
	tests := []struct {
		name      string
		gradients []float32
		maxNorm   float32
		wantNorm  float64
	}{
		{
			name:      "all zero",
			gradients: []float32{0, 0},
			maxNorm:   1,
			wantNorm:  0,
		},
		{
			name:      "below threshold",
			gradients: []float32{3, 4},
			maxNorm:   6,
			wantNorm:  5,
		},
		{
			name:      "exact threshold",
			gradients: []float32{3, 4},
			maxNorm:   5,
			wantNorm:  5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				parameter   *optimizer.Parameter
				base        *mockOptimizer
				clipping    *optimizer.GradientClipping
				observation optimizer.GradientClippingObservation
				available   bool
				err         error
			)

			parameter = mustParameter(t, 1, 2, []float32{1, 2})
			accumulateGradient(t, parameter, tt.gradients)
			base = &mockOptimizer{
				updateFunc: func(parameters []*optimizer.Parameter) (err error) {
					requireMatrixValues(t, parameters[0].Gradient(), tt.gradients)
					err = parameters[0].ResetGradient()
					return err
				},
			}
			clipping, err = optimizer.NewGradientClipping(
				base,
				optimizer.GradientClippingConfig{MaxNorm: tt.maxNorm},
			)
			if err != nil {
				t.Fatalf("NewGradientClipping returned error: %v", err)
			}

			if err = clipping.Update([]*optimizer.Parameter{parameter}); err != nil {
				t.Fatalf("Update returned error: %v", err)
			}
			observation, available = clipping.Observation()
			if !available ||
				observation.GlobalNorm != tt.wantNorm ||
				observation.Scale != 1 ||
				!observation.BaseUpdateCompleted {
				t.Fatalf("Observation = %+v, want no-op norm %g", observation, tt.wantNorm)
			}
		})
	}
}

func Test_GradientClipping_CombinedValueThenNorm(t *testing.T) {
	var (
		parameter   *optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		wantNorm    float64
		wantScale   float64
		wantValue   float32
		err         error
	)

	wantNorm = math.Sqrt(50)
	wantScale = 5 / wantNorm
	wantValue = float32(5 * wantScale)
	parameter = mustParameter(t, 1, 2, []float32{0, 0})
	accumulateGradient(t, parameter, []float32{6, 8})
	base = &mockOptimizer{
		updateFunc: func(parameters []*optimizer.Parameter) (err error) {
			requireMatrixValues(t, parameters[0].Gradient(), []float32{wantValue, wantValue})
			err = parameters[0].ResetGradient()
			return err
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 5, MaxNorm: 5},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = clipping.Update([]*optimizer.Parameter{parameter}); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	observation, available = clipping.Observation()
	if !available || !observation.ValueClipped || !observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v, want successful combined clipping", observation)
	}
	if math.Abs(observation.GlobalNorm-wantNorm) > 1e-12 {
		t.Fatalf("GlobalNorm = %.15g, want %.15g", observation.GlobalNorm, wantNorm)
	}
	if math.Abs(observation.Scale-wantScale) > 1e-12 {
		t.Fatalf("Scale = %.15g, want %.15g", observation.Scale, wantScale)
	}
}

func Test_GradientClipping_EmptyParametersDelegates(t *testing.T) {
	var (
		base        mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	clipping, err = optimizer.NewGradientClipping(
		&base,
		optimizer.GradientClippingConfig{MaxNorm: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	if err = clipping.Update(nil); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	if base.updateCalls != 1 {
		t.Fatalf("base Update calls = %d, want 1", base.updateCalls)
	}

	observation, available = clipping.Observation()
	if !available ||
		observation.ValueClipped ||
		observation.GlobalNorm != 0 ||
		observation.Scale != 1 ||
		!observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v, want successful empty update", observation)
	}
}

func Test_GradientClipping_NonFiniteGradientPreservesState(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		text  string
	}{
		{name: "NaN", value: float32(math.NaN()), text: "value=NaN"},
		{name: "positive infinity", value: float32(math.Inf(1)), text: "value=+Inf"},
		{name: "negative infinity", value: float32(math.Inf(-1)), text: "value=-Inf"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				parameter   *optimizer.Parameter
				base        mockOptimizer
				clipping    *optimizer.GradientClipping
				observation optimizer.GradientClippingObservation
				available   bool
				values      []float32
				err         error
			)

			parameter = mustParameter(t, 2, 2, []float32{1, 2, 3, 4})
			accumulateGradient(t, parameter, []float32{1, 2, tt.value, 4})
			clipping, err = optimizer.NewGradientClipping(
				&base,
				optimizer.GradientClippingConfig{MaxValue: 1, MaxNorm: 1},
			)
			if err != nil {
				t.Fatalf("NewGradientClipping returned error: %v", err)
			}

			err = clipping.Update([]*optimizer.Parameter{parameter})
			if err == nil {
				t.Fatal("Update error = nil, want error")
			}
			if !strings.Contains(
				err.Error(),
				"optimizer: gradient clipping non-finite gradient: parameter=0 row=1 col=0",
			) || !strings.Contains(err.Error(), tt.text) {
				t.Fatalf("Update error = %q, want position and value context", err.Error())
			}
			if base.updateCalls != 0 {
				t.Fatalf("base Update calls = %d, want 0", base.updateCalls)
			}

			values, err = parameter.Gradient().Values()
			if err != nil {
				t.Fatalf("Gradient Values returned error: %v", err)
			}
			if values[0] != 1 || values[1] != 2 || values[3] != 4 {
				t.Fatalf("finite gradients changed on failure: %v", values)
			}
			if tt.name == "NaN" && !math.IsNaN(float64(values[2])) {
				t.Fatalf("NaN gradient changed on failure: %v", values[2])
			}
			if tt.name != "NaN" && values[2] != tt.value {
				t.Fatalf("infinite gradient changed on failure: %v", values[2])
			}

			observation, available = clipping.Observation()
			if available || observation != (optimizer.GradientClippingObservation{}) {
				t.Fatal("failed clipping published an observation")
			}
		})
	}
}

func Test_GradientClipping_ExtremeFiniteNormIsOverflowSafe(t *testing.T) {
	var (
		parameter   *optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	parameter = mustParameter(t, 1, 2, []float32{0, 0})
	accumulateGradient(t, parameter, []float32{math.MaxFloat32, math.MaxFloat32})
	base = &mockOptimizer{
		updateFunc: func(parameters []*optimizer.Parameter) (err error) {
			var values []float32

			values, err = parameters[0].Gradient().Values()
			if err != nil {
				return err
			}
			if math.IsInf(float64(values[0]), 0) || math.IsNaN(float64(values[0])) {
				t.Fatalf("transformed gradient is non-finite: %g", values[0])
			}
			err = parameters[0].ResetGradient()
			return err
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: math.MaxFloat32},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = clipping.Update([]*optimizer.Parameter{parameter}); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	observation, available = clipping.Observation()
	if !available || math.IsInf(observation.GlobalNorm, 0) || observation.Scale >= 1 {
		t.Fatalf("Observation = %+v, want finite scaled extreme norm", observation)
	}
}

func Test_GradientClipping_BaseErrorPublishesCurrentObservation(t *testing.T) {
	var (
		wantErr     error
		parameter   *optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		err         error
	)

	wantErr = errors.New("base update failed")
	parameter = mustParameter(t, 1, 2, []float32{1, 2})
	accumulateGradient(t, parameter, []float32{2, -3})
	base = &mockOptimizer{
		updateFunc: func(parameters []*optimizer.Parameter) (err error) {
			requireMatrixValues(t, parameters[0].Gradient(), []float32{1, -1})
			return wantErr
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	err = clipping.Update([]*optimizer.Parameter{parameter})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Update error = %v, want %v", err, wantErr)
	}
	requireMatrixValues(t, parameter.Gradient(), []float32{1, -1})

	observation, available = clipping.Observation()
	if !available || !observation.ValueClipped || observation.BaseUpdateCompleted {
		t.Fatalf("Observation = %+v, want failed base update", observation)
	}

	if err = parameter.Gradient().CopyValuesFrom([]float32{float32(math.NaN()), 2}); err != nil {
		t.Fatalf("CopyValuesFrom returned error: %v", err)
	}
	err = clipping.Update([]*optimizer.Parameter{parameter})
	if err == nil {
		t.Fatal("Update error = nil, want clipping failure")
	}
	if base.updateCalls != 1 {
		t.Fatalf("base Update calls = %d, want 1", base.updateCalls)
	}

	var preserved optimizer.GradientClippingObservation
	preserved, available = clipping.Observation()
	if !available || preserved != observation {
		t.Fatalf("Observation = %+v, want preserved %+v", preserved, observation)
	}
	preserved.Scale = 0.5
	preserved, available = clipping.Observation()
	if !available || preserved != observation {
		t.Fatal("Observation returned mutable wrapper state")
	}
}

func Test_GradientClipping_ClipsBeforeBuiltInOptimizerState(t *testing.T) {
	tests := []struct {
		name       string
		newBase    func(testing.TB) optimizer.Optimizer
		wantValues []float32
	}{
		{
			name: "SGD",
			newBase: func(tb testing.TB) (base optimizer.Optimizer) {
				var (
					sgd *optimizer.SGD
					err error
				)

				sgd, err = optimizer.NewSGD(0.1)
				if err != nil {
					tb.Fatalf("NewSGD returned error: %v", err)
				}
				return sgd
			},
			wantValues: []float32{0.9, 1.1},
		},
		{
			name: "Momentum",
			newBase: func(tb testing.TB) (base optimizer.Optimizer) {
				var (
					momentum *optimizer.Momentum
					err      error
				)

				momentum, err = optimizer.NewMomentumWithCoefficient(0.1, 0.9)
				if err != nil {
					tb.Fatalf("NewMomentumWithCoefficient returned error: %v", err)
				}
				return momentum
			},
			wantValues: []float32{0.9, 1.1},
		},
		{
			name: "Adam",
			newBase: func(tb testing.TB) (base optimizer.Optimizer) {
				var (
					adam *optimizer.Adam
					err  error
				)

				adam, err = optimizer.NewAdamWithConfig(0.1, 0, 0, 1)
				if err != nil {
					tb.Fatalf("NewAdamWithConfig returned error: %v", err)
				}
				return adam
			},
			wantValues: []float32{0.95, 1.05},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				parameter *optimizer.Parameter
				base      optimizer.Optimizer
				clipping  *optimizer.GradientClipping
				err       error
			)

			parameter = mustParameter(t, 1, 2, []float32{1, 1})
			accumulateGradient(t, parameter, []float32{2, -2})
			base = tt.newBase(t)
			clipping, err = optimizer.NewGradientClipping(
				base,
				optimizer.GradientClippingConfig{MaxValue: 1},
			)
			if err != nil {
				t.Fatalf("NewGradientClipping returned error: %v", err)
			}

			if err = clipping.Update([]*optimizer.Parameter{parameter}); err != nil {
				t.Fatalf("Update returned error: %v", err)
			}
			requireMatrixValues(t, parameter.Values(), tt.wantValues)
			requireMatrixValues(t, parameter.Gradient(), []float32{0, 0})
		})
	}
}

func Test_GradientClipping_RegularizationComposition(t *testing.T) {
	tests := []struct {
		name       string
		newRule    func(testing.TB, optimizer.Optimizer, optimizer.Regularizer, optimizer.Regularizer) optimizer.Optimizer
		wantValues []float32
	}{
		{
			name: "regularize then clip",
			newRule: func(
				tb testing.TB,
				base optimizer.Optimizer,
				l1 optimizer.Regularizer,
				l2 optimizer.Regularizer,
			) (optimizerRule optimizer.Optimizer) {
				var (
					clipping    *optimizer.GradientClipping
					regularized *optimizer.Regularized
					err         error
				)

				clipping, err = optimizer.NewGradientClipping(
					base,
					optimizer.GradientClippingConfig{MaxValue: 3},
				)
				if err != nil {
					tb.Fatalf("NewGradientClipping returned error: %v", err)
				}
				regularized, err = optimizer.NewRegularized(clipping, l1, l2)
				if err != nil {
					tb.Fatalf("NewRegularized returned error: %v", err)
				}
				return regularized
			},
			wantValues: []float32{1.7, -3.275},
		},
		{
			name: "clip then regularize",
			newRule: func(
				tb testing.TB,
				base optimizer.Optimizer,
				l1 optimizer.Regularizer,
				l2 optimizer.Regularizer,
			) (optimizerRule optimizer.Optimizer) {
				var (
					clipping    *optimizer.GradientClipping
					regularized *optimizer.Regularized
					err         error
				)

				regularized, err = optimizer.NewRegularized(base, l1, l2)
				if err != nil {
					tb.Fatalf("NewRegularized returned error: %v", err)
				}
				clipping, err = optimizer.NewGradientClipping(
					regularized,
					optimizer.GradientClippingConfig{MaxValue: 3},
				)
				if err != nil {
					tb.Fatalf("NewGradientClipping returned error: %v", err)
				}
				return clipping
			},
			wantValues: []float32{1.6, -3.175},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				parameter     *optimizer.Parameter
				base          *optimizer.SGD
				l1            *optimizer.L1
				l2            *optimizer.L2WeightDecay
				optimizerRule optimizer.Optimizer
				err           error
			)

			parameter = mustParameter(t, 1, 2, []float32{2, -3})
			accumulateGradient(t, parameter, []float32{3, 4})
			base, err = optimizer.NewSGD(0.1)
			if err != nil {
				t.Fatalf("NewSGD returned error: %v", err)
			}
			l1, err = optimizer.NewL1(0.5)
			if err != nil {
				t.Fatalf("NewL1 returned error: %v", err)
			}
			l2, err = optimizer.NewL2WeightDecay(0.25)
			if err != nil {
				t.Fatalf("NewL2WeightDecay returned error: %v", err)
			}
			optimizerRule = tt.newRule(t, base, l1, l2)

			if err = optimizerRule.Update([]*optimizer.Parameter{parameter}); err != nil {
				t.Fatalf("Update returned error: %v", err)
			}
			requireMatrixValues(t, parameter.Values(), tt.wantValues)
			requireMatrixValues(t, parameter.Gradient(), []float32{0, 0})
		})
	}
}

func Test_GradientClipping_IndividualRegularizationComposition(t *testing.T) {
	type testcase struct {
		name        string
		clipOutside bool
		regularizer func(testing.TB) optimizer.Regularizer
		wantValues  []float32
	}

	tests := []testcase{
		{
			name: "regularize then clip L1",
			regularizer: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l1  *optimizer.L1
					err error
				)

				l1, err = optimizer.NewL1(0.5)
				if err != nil {
					tb.Fatalf("NewL1 returned error: %v", err)
				}
				return l1
			},
			wantValues: []float32{1.7, -3.3},
		},
		{
			name: "regularize then clip L2",
			regularizer: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l2  *optimizer.L2WeightDecay
					err error
				)

				l2, err = optimizer.NewL2WeightDecay(0.25)
				if err != nil {
					tb.Fatalf("NewL2WeightDecay returned error: %v", err)
				}
				return l2
			},
			wantValues: []float32{1.7, -3.3},
		},
		{
			name:        "clip then regularize L1",
			clipOutside: true,
			regularizer: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l1  *optimizer.L1
					err error
				)

				l1, err = optimizer.NewL1(0.5)
				if err != nil {
					tb.Fatalf("NewL1 returned error: %v", err)
				}
				return l1
			},
			wantValues: []float32{1.65, -3.25},
		},
		{
			name:        "clip then regularize L2",
			clipOutside: true,
			regularizer: func(tb testing.TB) (regularizer optimizer.Regularizer) {
				var (
					l2  *optimizer.L2WeightDecay
					err error
				)

				l2, err = optimizer.NewL2WeightDecay(0.25)
				if err != nil {
					tb.Fatalf("NewL2WeightDecay returned error: %v", err)
				}
				return l2
			},
			wantValues: []float32{1.65, -3.225},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				parameter     *optimizer.Parameter
				base          *optimizer.SGD
				clipping      *optimizer.GradientClipping
				regularized   *optimizer.Regularized
				optimizerRule optimizer.Optimizer
				regularizer   optimizer.Regularizer
				err           error
			)

			parameter = mustParameter(t, 1, 2, []float32{2, -3})
			accumulateGradient(t, parameter, []float32{3, 4})
			base, err = optimizer.NewSGD(0.1)
			if err != nil {
				t.Fatalf("NewSGD returned error: %v", err)
			}
			regularizer = tt.regularizer(t)
			if tt.clipOutside {
				regularized, err = optimizer.NewRegularized(base, regularizer)
				if err != nil {
					t.Fatalf("NewRegularized returned error: %v", err)
				}
				clipping, err = optimizer.NewGradientClipping(
					regularized,
					optimizer.GradientClippingConfig{MaxValue: 3},
				)
				if err != nil {
					t.Fatalf("NewGradientClipping returned error: %v", err)
				}
				optimizerRule = clipping
			} else {
				clipping, err = optimizer.NewGradientClipping(
					base,
					optimizer.GradientClippingConfig{MaxValue: 3},
				)
				if err != nil {
					t.Fatalf("NewGradientClipping returned error: %v", err)
				}
				regularized, err = optimizer.NewRegularized(clipping, regularizer)
				if err != nil {
					t.Fatalf("NewRegularized returned error: %v", err)
				}
				optimizerRule = regularized
			}

			if err = optimizerRule.Update([]*optimizer.Parameter{parameter}); err != nil {
				t.Fatalf("Update returned error: %v", err)
			}
			requireMatrixValues(t, parameter.Values(), tt.wantValues)
		})
	}
}

func Test_GradientClipping_DuplicateReferencesUseSnapshot(t *testing.T) {
	var (
		parameter   *optimizer.Parameter
		parameters  []*optimizer.Parameter
		base        *mockOptimizer
		clipping    *optimizer.GradientClipping
		observation optimizer.GradientClippingObservation
		available   bool
		wantScale   float64
		err         error
	)

	wantScale = 5 / math.Sqrt(50)
	parameter = mustParameter(t, 1, 2, []float32{1, 2})
	accumulateGradient(t, parameter, []float32{3, 4})
	parameters = []*optimizer.Parameter{parameter, parameter}
	base = &mockOptimizer{
		updateFunc: func(received []*optimizer.Parameter) (err error) {
			requireMatrixValues(
				t,
				received[0].Gradient(),
				[]float32{float32(3 * wantScale), float32(4 * wantScale)},
			)
			return nil
		},
	}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: 5},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = clipping.Update(parameters); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}
	observation, available = clipping.Observation()
	if !available || math.Abs(observation.GlobalNorm-math.Sqrt(50)) > 1e-12 {
		t.Fatalf("Observation = %+v, want duplicate-reference norm sqrt(50)", observation)
	}
}

func Test_GradientClipping_DelegatesLearningRateThroughWrapperOrders(t *testing.T) {
	var (
		base             *mockOptimizer
		l1               *optimizer.L1
		innerClipping    *optimizer.GradientClipping
		outerRegularized *optimizer.Regularized
		innerRegularized *optimizer.Regularized
		outerClipping    *optimizer.GradientClipping
		optimizerRule    optimizer.Optimizer
		err              error
	)

	base = &mockOptimizer{learningRate: 0.1}
	l1, err = optimizer.NewL1(0.1)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}
	innerClipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	outerRegularized, err = optimizer.NewRegularized(innerClipping, l1)
	if err != nil {
		t.Fatalf("NewRegularized returned error: %v", err)
	}
	optimizerRule = outerRegularized
	if err = optimizerRule.SetLearningRate(0.2); err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}
	testutil.RequireAlmostEqual(t, base.learningRate, 0.2, epsilon)
	testutil.RequireAlmostEqual(t, optimizerRule.LearningRate(), 0.2, epsilon)

	innerRegularized, err = optimizer.NewRegularized(base, l1)
	if err != nil {
		t.Fatalf("NewRegularized returned error: %v", err)
	}
	outerClipping, err = optimizer.NewGradientClipping(
		innerRegularized,
		optimizer.GradientClippingConfig{MaxNorm: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	optimizerRule = outerClipping
	if err = optimizerRule.SetLearningRate(0.3); err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}
	testutil.RequireAlmostEqual(t, base.learningRate, 0.3, epsilon)
	testutil.RequireAlmostEqual(t, optimizerRule.LearningRate(), 0.3, epsilon)
}

func Test_GradientClipping_SetLearningRateReturnsBaseError(t *testing.T) {
	var (
		wantErr  error
		base     *mockOptimizer
		clipping *optimizer.GradientClipping
		err      error
	)

	wantErr = errors.New("set rate failed")
	base = &mockOptimizer{setLearningRateErr: wantErr}
	clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxValue: 1},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	err = clipping.SetLearningRate(0.2)
	if !errors.Is(err, wantErr) {
		t.Fatalf("SetLearningRate error = %v, want %v", err, wantErr)
	}
}

func Test_GradientClipping_IsDeterministic(t *testing.T) {
	var (
		firstParameter    *optimizer.Parameter
		secondParameter   *optimizer.Parameter
		firstBase         *optimizer.SGD
		secondBase        *optimizer.SGD
		firstClipping     *optimizer.GradientClipping
		secondClipping    *optimizer.GradientClipping
		firstObservation  optimizer.GradientClippingObservation
		secondObservation optimizer.GradientClippingObservation
		firstAvailable    bool
		secondAvailable   bool
		firstValues       []float32
		secondValues      []float32
		err               error
	)

	firstParameter = mustParameter(t, 2, 2, []float32{1, 2, 3, 4})
	secondParameter = mustParameter(t, 2, 2, []float32{1, 2, 3, 4})
	accumulateGradient(t, firstParameter, []float32{6, -8, 2, -4})
	accumulateGradient(t, secondParameter, []float32{6, -8, 2, -4})
	firstBase, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	secondBase, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	firstClipping, err = optimizer.NewGradientClipping(
		firstBase,
		optimizer.GradientClippingConfig{MaxValue: 5, MaxNorm: 6},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}
	secondClipping, err = optimizer.NewGradientClipping(
		secondBase,
		optimizer.GradientClippingConfig{MaxValue: 5, MaxNorm: 6},
	)
	if err != nil {
		t.Fatalf("NewGradientClipping returned error: %v", err)
	}

	if err = firstClipping.Update([]*optimizer.Parameter{firstParameter}); err != nil {
		t.Fatalf("first Update returned error: %v", err)
	}
	if err = secondClipping.Update([]*optimizer.Parameter{secondParameter}); err != nil {
		t.Fatalf("second Update returned error: %v", err)
	}

	firstObservation, firstAvailable = firstClipping.Observation()
	secondObservation, secondAvailable = secondClipping.Observation()
	if firstAvailable != secondAvailable || firstObservation != secondObservation {
		t.Fatalf(
			"observations differ: first=%+v/%t second=%+v/%t",
			firstObservation,
			firstAvailable,
			secondObservation,
			secondAvailable,
		)
	}
	firstValues, err = firstParameter.Values().Values()
	if err != nil {
		t.Fatalf("first Values returned error: %v", err)
	}
	secondValues, err = secondParameter.Values().Values()
	if err != nil {
		t.Fatalf("second Values returned error: %v", err)
	}
	if !reflect.DeepEqual(firstValues, secondValues) {
		t.Fatalf("parameter values differ: first=%v second=%v", firstValues, secondValues)
	}
}
