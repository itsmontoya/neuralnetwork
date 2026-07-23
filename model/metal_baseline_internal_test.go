//go:build darwin && cgo && metal && !purego

package model_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

func Test_MetalBatchedExecutionTransferCounts(t *testing.T) {
	var shape metalBaselineShape
	shape = metalBaselineShape{
		name:       "AtThreshold",
		batchSize:  64,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 128,
	}

	var tests []struct {
		name      string
		setup     func(testing.TB, metalBaselineShape) func() error
		buffers   uint64
		uploads   uint64
		downloads uint64
		commands  uint64
		waits     uint64
	}
	tests = []struct {
		name      string
		setup     func(testing.TB, metalBaselineShape) func() error
		buffers   uint64
		uploads   uint64
		downloads uint64
		commands  uint64
		waits     uint64
	}{
		{name: "Predict", setup: setupMetalBaselinePredict, buffers: 13, uploads: 5, downloads: 0, commands: 1, waits: 1},
		{name: "Backward", setup: setupMetalBaselineBackward, buffers: 6, uploads: 2, downloads: 5, commands: 2, waits: 2},
		{name: "TrainBatch", setup: setupMetalBaselineTrainBatch, buffers: 19, uploads: 7, downloads: 6, commands: 3, waits: 3},
		{name: "Fit", setup: setupMetalBaselineFit, buffers: 28, uploads: 12, downloads: 7, commands: 4, waits: 4},
	}

	metaltest.Enable()
	defer metaltest.Disable()

	var (
		test struct {
			name      string
			setup     func(testing.TB, metalBaselineShape) func() error
			buffers   uint64
			uploads   uint64
			downloads uint64
			commands  uint64
			waits     uint64
		}
		run       func() error
		counters  metaltest.Counters
		available bool
		err       error
	)

	if _, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	run = setupMetalBaselinePredict(t, shape)
	metaltest.Reset()
	if err = run(); err != nil {
		t.Fatalf("Metal availability probe returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	if counters.CommandSubmissions == 0 {
		t.Fatalf("Metal execution unavailable: %s", counters.LastError)
	}

	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			run = test.setup(t, shape)
			metaltest.Reset()
			if err = run(); err != nil {
				t.Fatalf("execution returned error: %v", err)
			}

			counters = metaltest.Snapshot()
			requireMetalBaselineCounters(
				t,
				counters,
				test.buffers,
				test.uploads,
				test.downloads,
				test.commands,
				test.waits,
			)
		})
	}
}

func Test_MetalBaselineBelowThresholdUsesCPU(t *testing.T) {
	var shape metalBaselineShape
	shape = metalBaselineShape{
		name:       "DirectlyBelowThreshold",
		batchSize:  63,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 128,
	}

	var (
		run      func() error
		counters metaltest.Counters
		err      error
	)

	metaltest.Enable()
	defer metaltest.Disable()

	run = setupMetalBaselinePredict(t, shape)
	metaltest.Reset()
	if err = run(); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	requireMetalBaselineCounters(t, counters, 0, 0, 0, 0, 0)
}

func requireMetalBaselineCounters(
	tb testing.TB,
	counters metaltest.Counters,
	buffers,
	uploads,
	downloads,
	commands,
	waits uint64,
) {
	tb.Helper()

	if counters.BufferCreations != buffers ||
		counters.InputUploads != uploads ||
		counters.ResultDownloads != downloads ||
		counters.CommandSubmissions != commands ||
		counters.Waits != waits {
		tb.Fatalf(
			"Metal counters = %+v, want buffers=%d uploads=%d downloads=%d commands=%d waits=%d",
			counters,
			buffers,
			uploads,
			downloads,
			commands,
			waits,
		)
	}

	if counters.LastError != "" {
		tb.Fatalf("last Metal error = %q, want empty", counters.LastError)
	}
}
