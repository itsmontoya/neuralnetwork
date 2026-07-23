//go:build darwin && cgo && metal && !purego

package matrix

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

var commandBatchBenchmarkResult *Matrix

func Benchmark_MetalCommandBatch(b *testing.B) {
	var (
		left         *Matrix
		right        *Matrix
		intermediate *Matrix
		fallback     *Matrix
		result       *Matrix
		err          error
	)

	if !metalAvailable() {
		b.Skipf("Metal device unavailable: %s", metalLastError())
	}
	left = metalTestMatrix(b, 64, 128, 0.25)
	right = metalTestMatrix(b, 128, 128, -0.5)
	if intermediate, err = New(64, 128); err != nil {
		b.Fatalf("New intermediate returned error: %v", err)
	}
	if fallback, err = New(64, 128); err != nil {
		b.Fatalf("New fallback returned error: %v", err)
	}
	if result, err = New(64, 128); err != nil {
		b.Fatalf("New result returned error: %v", err)
	}

	b.Run("StandaloneTwoMatMuls", func(b *testing.B) {
		var index int

		metaltest.Enable()
		defer metaltest.Disable()
		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if err = left.MatMulInto(right, intermediate); err != nil {
				b.Fatalf("first MatMulInto returned error: %v", err)
			}
			if err = intermediate.MatMulInto(right, result); err != nil {
				b.Fatalf("second MatMulInto returned error: %v", err)
			}
		}
		b.StopTimer()
		reportCommandBatchMetrics(b, metaltest.Snapshot())
	})

	b.Run("BatchedTwoMatMuls", func(b *testing.B) {
		var (
			execution *device.Execution
			available bool
			index     int
		)

		metaltest.Enable()
		defer metaltest.Disable()
		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if execution, available, err = device.NewSharedExecution(); err != nil {
				b.Fatalf("NewSharedExecution returned error: %v", err)
			}
			if !available {
				b.Fatal("Metal became unavailable during benchmark")
			}
			if err = execution.Bind(left); err != nil {
				b.Fatalf("Bind input returned error: %v", err)
			}
			if err = left.MatMulInto(right, intermediate); err != nil {
				b.Fatalf("first MatMulInto returned error: %v", err)
			}
			if err = intermediate.MatMulInto(right, result); err != nil {
				b.Fatalf("second MatMulInto returned error: %v", err)
			}
			if err = execution.Finish(); err != nil {
				b.Fatalf("Finish returned error: %v", err)
			}
		}
		b.StopTimer()
		reportCommandBatchMetrics(b, metaltest.Snapshot())
	})

	b.Run("CPUFallbackBoundary", func(b *testing.B) {
		var (
			execution *device.Execution
			available bool
			index     int
		)

		metaltest.Enable()
		defer metaltest.Disable()
		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if execution, available, err = device.NewSharedExecution(); err != nil {
				b.Fatalf("NewSharedExecution returned error: %v", err)
			}
			if !available {
				b.Fatal("Metal became unavailable during benchmark")
			}
			if err = execution.Bind(left); err != nil {
				b.Fatalf("Bind input returned error: %v", err)
			}
			if err = left.MatMulInto(right, intermediate); err != nil {
				b.Fatalf("first MatMulInto returned error: %v", err)
			}
			if err = intermediate.AddScalarInto(1, fallback); err != nil {
				b.Fatalf("AddScalarInto returned error: %v", err)
			}
			if err = fallback.MatMulInto(right, result); err != nil {
				b.Fatalf("second MatMulInto returned error: %v", err)
			}
			if err = execution.Finish(); err != nil {
				b.Fatalf("Finish returned error: %v", err)
			}
		}
		b.StopTimer()
		reportCommandBatchMetrics(b, metaltest.Snapshot())
	})

	commandBatchBenchmarkResult = result
}

func reportCommandBatchMetrics(b *testing.B, counters metaltest.Counters) {
	var iterations float64

	iterations = float64(b.N)
	b.ReportMetric(float64(counters.CommandSubmissions)/iterations, "commands/op")
	b.ReportMetric(float64(counters.Waits)/iterations, "waits/op")
	b.ReportMetric(float64(counters.InputUploads)/iterations, "uploads/op")
	b.ReportMetric(float64(counters.ResultDownloads)/iterations, "downloads/op")
}
