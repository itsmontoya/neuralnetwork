//go:build darwin && cgo && metal && !purego

package model_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

func Test_MetalBaselineSynchronousTransferCounts(t *testing.T) {
	var shape metalBaselineShape
	shape = metalBaselineShape{
		name:       "AtThreshold",
		batchSize:  64,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 128,
	}

	var tests []struct {
		name            string
		setup           func(testing.TB, metalBaselineShape) func() error
		multiplications uint64
	}
	tests = []struct {
		name            string
		setup           func(testing.TB, metalBaselineShape) func() error
		multiplications uint64
	}{
		{name: "Predict", setup: setupMetalBaselinePredict, multiplications: 2},
		{name: "Backward", setup: setupMetalBaselineBackward, multiplications: 4},
		{name: "TrainBatch", setup: setupMetalBaselineTrainBatch, multiplications: 6},
		{name: "Fit", setup: setupMetalBaselineFit, multiplications: 8},
	}

	metaltest.Enable()
	defer metaltest.Disable()

	var (
		test struct {
			name            string
			setup           func(testing.TB, metalBaselineShape) func() error
			multiplications uint64
		}
		run      func() error
		counters metaltest.Counters
		err      error
	)

	run = setupMetalBaselinePredict(t, shape)
	metaltest.Reset()
	if err = run(); err != nil {
		t.Fatalf("Metal availability probe returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	if counters.CommandSubmissions == 0 {
		if strings.Contains(counters.LastError, "no default device") {
			t.Skipf("Metal device unavailable: %s", counters.LastError)
		}

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
			requireMetalBaselineCounters(t, counters, test.multiplications)
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
	requireMetalBaselineCounters(t, counters, 0)
}

func requireMetalBaselineCounters(tb testing.TB, counters metaltest.Counters, multiplications uint64) {
	tb.Helper()

	if counters.BufferCreations != multiplications*3 {
		tb.Fatalf("buffer creations = %d, want %d", counters.BufferCreations, multiplications*3)
	}

	if counters.InputUploads != multiplications*2 {
		tb.Fatalf("input uploads = %d, want %d", counters.InputUploads, multiplications*2)
	}

	if counters.ResultDownloads != multiplications {
		tb.Fatalf("result downloads = %d, want %d", counters.ResultDownloads, multiplications)
	}

	if counters.CommandSubmissions != multiplications {
		tb.Fatalf("command submissions = %d, want %d", counters.CommandSubmissions, multiplications)
	}

	if counters.Waits != multiplications {
		tb.Fatalf("waits = %d, want %d", counters.Waits, multiplications)
	}

	if counters.LastError != "" {
		tb.Fatalf("last Metal error = %q, want empty", counters.LastError)
	}
}
