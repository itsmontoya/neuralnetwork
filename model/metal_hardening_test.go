//go:build darwin && cgo && metal && !purego

package model_test

import (
	"bytes"
	"runtime"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_SequentialResidentSteadyStateAllocationsAndResources(t *testing.T) {
	type testcase struct {
		name             string
		setup            func(testing.TB, metalBaselineShape) func() error
		maxAllocations   float64
		commands         uint64
		downloads        uint64
		kernels          uint64
		fallbackBarriers uint64
	}

	tests := []testcase{
		{
			name:           "Predict",
			setup:          setupMetalBaselinePredict,
			maxAllocations: 25,
			commands:       1,
			kernels:        10,
		},
		{
			name:           "Backward",
			setup:          setupMetalBaselineBackward,
			maxAllocations: 31,
			commands:       1,
			kernels:        12,
		},
		{
			name:             "TrainBatch",
			setup:            setupMetalBaselineTrainBatch,
			maxAllocations:   73,
			commands:         2,
			downloads:        1,
			kernels:          32,
			fallbackBarriers: 1,
		},
	}

	var (
		runtimeValue *device.Runtime
		available    bool
		err          error
	)

	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}

	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				shape       metalBaselineShape
				run         func() error
				before      device.ResourceSnapshot
				after       device.ResourceSnapshot
				counters    metaltest.Counters
				allocations float64
			)

			shape = metalBaselineShape(metalTrainingShape())
			run = test.setup(t, shape)
			if err = run(); err != nil {
				t.Fatalf("warm-up returned error: %v", err)
			}

			runtime.GC()
			before = runtimeValue.ResourceSnapshot()
			allocations = testing.AllocsPerRun(20, func() {
				var runErr error

				if runErr = run(); runErr != nil {
					panic(runErr)
				}
			})
			runtime.GC()
			after = runtimeValue.ResourceSnapshot()
			if allocations > test.maxAllocations {
				t.Fatalf(
					"steady-state allocations = %g, want at most %g",
					allocations,
					test.maxAllocations,
				)
			}
			requireBoundedMetalResources(t, before, after)

			metaltest.Enable()
			if err = run(); err != nil {
				metaltest.Disable()
				t.Fatalf("instrumented execution returned error: %v", err)
			}
			counters = metaltest.Snapshot()
			metaltest.Disable()
			if counters.InputUploads != 0 ||
				counters.ResultDownloads != test.downloads ||
				counters.CommandSubmissions != test.commands ||
				counters.Waits != test.commands ||
				counters.KernelEncodes != test.kernels ||
				counters.FallbackBarriers != test.fallbackBarriers {
				t.Fatalf("steady-state counters = %+v", counters)
			}
		})
	}
}

func Test_SequentialResidentDistinctModelsRunConcurrently(t *testing.T) {
	const (
		workers    = 4
		iterations = 8
	)

	var (
		runtimeValue *device.Runtime
		shape        metalBaselineShape
		networks     [workers]*model.Sequential
		inputs       [workers]*matrix.Matrix
		outputs      [workers]*matrix.Matrix
		errorsOut    [workers]error
		waitGroup    sync.WaitGroup
		before       device.ResourceSnapshot
		after        device.ResourceSnapshot
		counters     metaltest.Counters
		available    bool
		worker       int
		err          error
	)

	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	shape = metalBaselineShape(metalTrainingShape())
	for worker = 0; worker < workers; worker++ {
		networks[worker] = metalBaselineModel(t, shape)
		inputs[worker], _ = metalBaselineMatrices(t, shape)
		if outputs[worker], err = networks[worker].Predict(inputs[worker]); err != nil {
			t.Fatalf("worker %d warm-up returned error: %v", worker, err)
		}
	}

	runtime.GC()
	before = runtimeValue.ResourceSnapshot()
	metaltest.Enable()
	for worker = 0; worker < workers; worker++ {
		waitGroup.Add(1)
		go func(workerIndex int) {
			defer waitGroup.Done()

			var iteration int
			for iteration = 0; iteration < iterations; iteration++ {
				outputs[workerIndex], errorsOut[workerIndex] =
					networks[workerIndex].Predict(inputs[workerIndex])
				if errorsOut[workerIndex] != nil {
					return
				}
			}
		}(worker)
	}
	waitGroup.Wait()
	counters = metaltest.Snapshot()
	metaltest.Disable()
	runtime.GC()
	after = runtimeValue.ResourceSnapshot()

	for worker = 0; worker < workers; worker++ {
		if errorsOut[worker] != nil {
			t.Fatalf("worker %d returned error: %v", worker, errorsOut[worker])
		}
		if outputs[worker].Rows() != shape.batchSize ||
			outputs[worker].Cols() != shape.classCount {
			t.Fatalf(
				"worker %d output shape = %dx%d, want %dx%d",
				worker,
				outputs[worker].Rows(),
				outputs[worker].Cols(),
				shape.batchSize,
				shape.classCount,
			)
		}
	}

	if counters.BufferCreations != workers*iterations*8 ||
		counters.InputUploads != 0 ||
		counters.ResultDownloads != 0 ||
		counters.KernelEncodes != workers*iterations*10 ||
		counters.CommandSubmissions != workers*iterations ||
		counters.Waits != workers*iterations ||
		counters.LastError != "" {
		t.Fatalf("concurrent counters = %+v", counters)
	}
	requireBoundedMetalResources(t, before, after)
}

func Test_SequentialResidentLongRunningMixedStress(t *testing.T) {
	const iterations = 16

	var (
		runtimeValue    *device.Runtime
		shape           metalBaselineShape
		alternateShape  metalBaselineShape
		networks        [2]*model.Sequential
		optimizers      [2]*optimizer.SGD
		inputs          [2][2]*matrix.Matrix
		targets         [2][2]*matrix.Matrix
		fallbackNetwork *model.Sequential
		fallbackInputs  [2]*matrix.Matrix
		trainingData    *data.Dataset
		config          model.FitConfig
		output          *matrix.Matrix
		loaded          *model.Sequential
		parameters      []*optimizer.Parameter
		before          device.ResourceSnapshot
		after           device.ResourceSnapshot
		encoded         bytes.Buffer
		current         float32
		available       bool
		networkIndex    int
		shapeIndex      int
		iteration       int
		err             error
	)

	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	shape = metalBaselineShape(metalTrainingShape())
	alternateShape = shape
	alternateShape.name = "alternate"
	alternateShape.batchSize++

	for networkIndex = range networks {
		networks[networkIndex] = metalBaselineModel(t, shape)
		if optimizers[networkIndex], err = optimizer.NewSGD(0.000001); err != nil {
			t.Fatalf("NewSGD network %d returned error: %v", networkIndex, err)
		}
		inputs[networkIndex][0], targets[networkIndex][0] =
			metalBaselineMatrices(t, shape)
		inputs[networkIndex][1], targets[networkIndex][1] =
			metalBaselineMatrices(t, alternateShape)
		for shapeIndex = 0; shapeIndex < 2; shapeIndex++ {
			if _, err = networks[networkIndex].TrainBatch(
				inputs[networkIndex][shapeIndex],
				targets[networkIndex][shapeIndex],
				loss.CategoricalCrossEntropy{},
				optimizers[networkIndex],
			); err != nil {
				t.Fatalf(
					"network %d shape %d warm-up returned error: %v",
					networkIndex,
					shapeIndex,
					err,
				)
			}
		}
	}

	fallbackNetwork = metalBaselineModelWithActivation(
		t,
		shape,
		&metalFallbackActivation{delta: 0.25},
	)
	fallbackInputs[0], _ = metalBaselineMatrices(t, shape)
	fallbackInputs[1], _ = metalBaselineMatrices(t, alternateShape)
	for shapeIndex = range fallbackInputs {
		if _, err = fallbackNetwork.Predict(fallbackInputs[shapeIndex]); err != nil {
			t.Fatalf("fallback warm-up shape %d returned error: %v", shapeIndex, err)
		}
	}

	if trainingData, err = data.NewDataset(inputs[0][0], targets[0][0]); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}
	config.Epochs = 1
	config.BatchSize = shape.batchSize
	config.Optimizer = optimizers[0]
	config.Loss = loss.CategoricalCrossEntropy{}
	if _, err = networks[0].Fit(trainingData, config); err != nil {
		t.Fatalf("Fit warm-up returned error: %v", err)
	}

	runtime.GC()
	before = runtimeValue.ResourceSnapshot()
	for iteration = 0; iteration < iterations; iteration++ {
		networkIndex = iteration % len(networks)
		shapeIndex = (iteration / len(networks)) % 2
		if _, err = networks[networkIndex].TrainBatch(
			inputs[networkIndex][shapeIndex],
			targets[networkIndex][shapeIndex],
			loss.CategoricalCrossEntropy{},
			optimizers[networkIndex],
		); err != nil {
			t.Fatalf("iteration %d TrainBatch returned error: %v", iteration, err)
		}
		if output, err = networks[networkIndex].Predict(
			inputs[networkIndex][shapeIndex],
		); err != nil {
			t.Fatalf("iteration %d Predict returned error: %v", iteration, err)
		}
		if iteration%3 == 0 {
			if _, err = output.Values(); err != nil {
				t.Fatalf("iteration %d observation returned error: %v", iteration, err)
			}
		}
		if _, err = fallbackNetwork.Predict(fallbackInputs[shapeIndex]); err != nil {
			t.Fatalf("iteration %d fallback returned error: %v", iteration, err)
		}
		if iteration%4 == 0 {
			parameters = networks[networkIndex].Parameters()
			if current, err = parameters[0].Values().At(0, 0); err != nil {
				t.Fatalf("iteration %d parameter observation returned error: %v", iteration, err)
			}
			if err = parameters[0].Values().Set(0, 0, current+1e-6); err != nil {
				t.Fatalf("iteration %d parameter mutation returned error: %v", iteration, err)
			}

			encoded.Reset()
			if err = networks[networkIndex].Save(&encoded); err != nil {
				t.Fatalf("iteration %d Save returned error: %v", iteration, err)
			}
			if loaded, err = model.LoadSequential(bytes.NewReader(encoded.Bytes())); err != nil {
				t.Fatalf("iteration %d LoadSequential returned error: %v", iteration, err)
			}
			if _, err = loaded.Parameters()[0].Values().At(0, 0); err != nil {
				t.Fatalf("iteration %d loaded observation returned error: %v", iteration, err)
			}
		}
		if iteration%4 == 3 {
			if _, err = networks[0].Fit(trainingData, config); err != nil {
				t.Fatalf("iteration %d Fit returned error: %v", iteration, err)
			}
		}
	}
	runtime.GC()
	after = runtimeValue.ResourceSnapshot()
	requireBoundedMetalResources(t, before, after)
}

func metalBaselineModelWithActivation(
	tb testing.TB,
	shape metalBaselineShape,
	function *metalFallbackActivation,
) (network *model.Sequential) {
	tb.Helper()
	network, _, _, _, _, _, _ = metalInferenceModel(
		tb,
		metalInferenceShape(shape),
		function,
	)
	return network
}

func requireBoundedMetalResources(
	tb testing.TB,
	before,
	after device.ResourceSnapshot,
) {
	tb.Helper()

	if after.LiveBuffers > before.LiveBuffers ||
		after.LiveBufferBytes > before.LiveBufferBytes ||
		after.LiveScopes > before.LiveScopes {
		tb.Fatalf("Metal resources before=%+v after=%+v", before, after)
	}
	if after.CreatedBuffers-after.ReleasedBuffers != after.LiveBuffers ||
		after.CreatedScopes-after.ReleasedScopes != after.LiveScopes ||
		after.SubmittedCommands != after.CompletedCommands {
		tb.Fatalf("unbalanced Metal resources after stress = %+v", after)
	}
}
