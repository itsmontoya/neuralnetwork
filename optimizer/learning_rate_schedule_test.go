package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_LearningRateSchedule_Interface(t *testing.T) {
	var _ optimizer.LearningRateSchedule = (*optimizer.ConstantLearningRate)(nil)
	var _ optimizer.LearningRateSchedule = (*optimizer.StepDecay)(nil)
	var _ optimizer.LearningRateSchedule = (*optimizer.ExponentialDecay)(nil)
}

func Test_ConstantLearningRate(t *testing.T) {
	var (
		schedule *optimizer.ConstantLearningRate
		rate     float64
		err      error
	)

	schedule, err = optimizer.NewConstantLearningRate(0.25)
	if err != nil {
		t.Fatalf("NewConstantLearningRate returned error: %v", err)
	}

	rate, err = schedule.LearningRate(3)
	if err != nil {
		t.Fatalf("LearningRate returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, schedule.Rate(), 0.25, epsilon)
	testutil.RequireAlmostEqual(t, rate, 0.25, epsilon)
}

func Test_StepDecay(t *testing.T) {
	var (
		schedule *optimizer.StepDecay
		rates    []float64
		rate     float64
		epoch    int
		err      error
	)

	schedule, err = optimizer.NewStepDecay(0.1, 0.5, 2)
	if err != nil {
		t.Fatalf("NewStepDecay returned error: %v", err)
	}

	for epoch = 1; epoch <= 5; epoch++ {
		if rate, err = schedule.LearningRate(epoch); err != nil {
			t.Fatalf("LearningRate epoch %d returned error: %v", epoch, err)
		}

		rates = append(rates, rate)
	}

	testutil.RequireAlmostEqual(t, schedule.InitialLearningRate(), 0.1, epsilon)
	testutil.RequireAlmostEqual(t, schedule.Factor(), 0.5, epsilon)
	if schedule.StepSize() != 2 {
		t.Fatalf("StepSize = %d, want 2", schedule.StepSize())
	}

	testutil.RequireSliceAlmostEqual(t, rates, []float64{0.1, 0.1, 0.05, 0.05, 0.025}, epsilon)
}

func Test_ExponentialDecay(t *testing.T) {
	var (
		schedule *optimizer.ExponentialDecay
		rates    []float64
		rate     float64
		epoch    int
		err      error
	)

	schedule, err = optimizer.NewExponentialDecay(0.1, 0.5)
	if err != nil {
		t.Fatalf("NewExponentialDecay returned error: %v", err)
	}

	for epoch = 1; epoch <= 4; epoch++ {
		if rate, err = schedule.LearningRate(epoch); err != nil {
			t.Fatalf("LearningRate epoch %d returned error: %v", epoch, err)
		}

		rates = append(rates, rate)
	}

	testutil.RequireAlmostEqual(t, schedule.InitialLearningRate(), 0.1, epsilon)
	testutil.RequireAlmostEqual(t, schedule.DecayRate(), 0.5, epsilon)
	testutil.RequireSliceAlmostEqual(t, rates, []float64{0.1, 0.05, 0.025, 0.0125}, epsilon)
}

func Test_NewLearningRateSchedule_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name      string
		construct func() (optimizer.LearningRateSchedule, error)
	}

	tests := []testcase{
		{
			name: "constant learning rate",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var constant *optimizer.ConstantLearningRate

				if constant, err = optimizer.NewConstantLearningRate(0); constant == nil {
					return nil, err
				}

				schedule = constant
				return schedule, err
			},
		},
		{
			name: "step initial learning rate",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var step *optimizer.StepDecay

				if step, err = optimizer.NewStepDecay(0, 0.5, 2); step == nil {
					return nil, err
				}

				schedule = step
				return schedule, err
			},
		},
		{
			name: "step factor",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var step *optimizer.StepDecay

				if step, err = optimizer.NewStepDecay(0.1, math.NaN(), 2); step == nil {
					return nil, err
				}

				schedule = step
				return schedule, err
			},
		},
		{
			name: "step size",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var step *optimizer.StepDecay

				if step, err = optimizer.NewStepDecay(0.1, 0.5, 0); step == nil {
					return nil, err
				}

				schedule = step
				return schedule, err
			},
		},
		{
			name: "exponential initial learning rate",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var exponential *optimizer.ExponentialDecay

				if exponential, err = optimizer.NewExponentialDecay(0, 0.5); exponential == nil {
					return nil, err
				}

				schedule = exponential
				return schedule, err
			},
		},
		{
			name: "exponential decay rate",
			construct: func() (schedule optimizer.LearningRateSchedule, err error) {
				var exponential *optimizer.ExponentialDecay

				if exponential, err = optimizer.NewExponentialDecay(0.1, 1.1); exponential == nil {
					return nil, err
				}

				schedule = exponential
				return schedule, err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				schedule optimizer.LearningRateSchedule
				err      error
			)

			schedule, err = tt.construct()
			if err == nil {
				t.Fatal("constructor error = nil, want error")
			}

			if schedule != nil {
				t.Fatal("constructor returned schedule on error")
			}
		})
	}
}

func Test_LearningRateSchedule_ValidatesEpoch(t *testing.T) {
	var (
		schedule *optimizer.ConstantLearningRate
		err      error
	)

	schedule, err = optimizer.NewConstantLearningRate(0.1)
	if err != nil {
		t.Fatalf("NewConstantLearningRate returned error: %v", err)
	}

	_, err = schedule.LearningRate(0)
	if err == nil {
		t.Fatal("LearningRate error = nil, want error")
	}
}
