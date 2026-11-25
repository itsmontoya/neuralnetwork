//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"math/rand/v2"
	"testing"
	"time"
)

// Benchmark_MatMulRightTransposeMetal measures isolated resident right-transposed
// FP32 projections on the Llama 3.2 1B single-row decode shapes, multi-row
// prefill guards, and odd tail shapes.
func Benchmark_MatMulRightTransposeMetal(b *testing.B) {
	var cases = []struct {
		name              string
		rows, inner, cols int64
	}{
		{name: "FFN_GateUp_M1_K2048_N8192", rows: 1, inner: 2048, cols: 8192},
		{name: "FFN_Down_M1_K8192_N2048", rows: 1, inner: 8192, cols: 2048},
		{name: "Attn_QO_M1_K2048_N2048", rows: 1, inner: 2048, cols: 2048},
		{name: "Attn_KV_M1_K2048_N512", rows: 1, inner: 2048, cols: 512},
		{name: "Classifier_M1_K2048_N128256", rows: 1, inner: 2048, cols: 128256},
		{name: "FFN_GateUp_M6_K2048_N8192", rows: 6, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M27_K2048_N8192", rows: 27, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M28_K2048_N8192", rows: 28, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M29_K2048_N8192", rows: 29, inner: 2048, cols: 8192},
		{name: "OddTail_M1_K257_N33", rows: 1, inner: 257, cols: 33},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkMetalRightTranspose(b, tc.rows, tc.inner, tc.cols)
		})
	}
}

func benchmarkMetalRightTranspose(b *testing.B, rows, inner, cols int64) {
	var (
		gpu      = requireMetalDevice(b)
		native   = gpu.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		left     = requireTensor(b, gpu, rows, inner)
		right    = requireTensor(b, gpu, cols, inner)
		output   = requireTensor(b, gpu, rows, cols)
		random   = rand.New(rand.NewPCG(0x6e6e08, uint64(rows*1000003+inner*1009+cols)))
	)
	var (
		leftValues  = make([]float32, rows*inner)
		rightValues = make([]float32, cols*inner)
	)
	for i := range leftValues {
		leftValues[i] = 2*random.Float32() - 1
	}
	for i := range rightValues {
		rightValues[i] = 2*random.Float32() - 1
	}
	requireError(b, left.Upload(leftValues), nil)
	requireError(b, right.Upload(rightValues), nil)
	leftValues = nil
	rightValues = nil

	var firstStart = time.Now()
	runResidentRightTranspose(b, gpu, left, right, output)
	var firstElapsed = time.Since(firstStart)

	for range 3 {
		runResidentRightTranspose(b, gpu, left, right, output)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		runResidentRightTranspose(b, gpu, left, right, output)
	}
	b.StopTimer()

	b.ReportMetric(float64(firstElapsed), "first-use-ns")
	var final = native.ResourceSnapshot()
	var expectedBuffers = baseline.LiveBuffers + 3
	if final.LiveBuffers != expectedBuffers || final.LiveScopes != baseline.LiveScopes {
		b.Fatalf("unexpected resident projection resource use: before %+v, after %+v", baseline, final)
	}
	b.ReportMetric(float64(final.LiveBufferBytes-baseline.LiveBufferBytes), "requested-device-B")
}

// Benchmark_MatMulRightTransposeMixedMetal compares isolated resident
// right-transposed projections across Float32, Float16, and BFloat16 weights.
func Benchmark_MatMulRightTransposeMixedMetal(b *testing.B) {
	var cases = []struct {
		name              string
		rows, inner, cols int64
	}{
		{name: "FFN_GateUp_M1_K2048_N8192", rows: 1, inner: 2048, cols: 8192},
		{name: "FFN_Down_M1_K8192_N2048", rows: 1, inner: 8192, cols: 2048},
		{name: "Attn_QO_M1_K2048_N2048", rows: 1, inner: 2048, cols: 2048},
		{name: "Attn_KV_M1_K2048_N512", rows: 1, inner: 2048, cols: 512},
		{name: "Classifier_M1_K2048_N128256", rows: 1, inner: 2048, cols: 128256},
		{name: "FFN_GateUp_M6_K2048_N8192", rows: 6, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M27_K2048_N8192", rows: 27, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M28_K2048_N8192", rows: 28, inner: 2048, cols: 8192},
		{name: "FFN_GateUp_M29_K2048_N8192", rows: 29, inner: 2048, cols: 8192},
		{name: "OddTail_M1_K257_N33", rows: 1, inner: 257, cols: 33},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			for _, dtype := range []DType{Float32, Float16, BFloat16} {
				b.Run(dtype.String(), func(b *testing.B) {
					benchmarkMetalRightTransposeDType(b, dtype, tc.rows, tc.inner, tc.cols)
				})
			}
		})
	}
}

func benchmarkMetalRightTransposeDType(b *testing.B, dtype DType, rows, inner, cols int64) {
	var (
		gpu      = requireMetalDevice(b)
		native   = gpu.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		left     = requireTensor(b, gpu, rows, inner)
		right    = requireTensorWithDType(b, gpu, dtype, cols, inner)
		output   = requireTensor(b, gpu, rows, cols)
		random   = rand.New(rand.NewPCG(0x6e6e10, uint64(rows*1000003+inner*1009+cols)))
	)
	var (
		leftValues  = make([]float32, rows*inner)
		rightValues = make([]float32, cols*inner)
	)
	for i := range leftValues {
		leftValues[i] = 2*random.Float32() - 1
	}
	for i := range rightValues {
		rightValues[i] = 2*random.Float32() - 1
	}
	requireError(b, left.Upload(leftValues), nil)
	requireError(b, right.UploadFloat32As(rightValues), nil)
	leftValues = nil
	rightValues = nil

	var firstStart = time.Now()
	runResidentRightTranspose(b, gpu, left, right, output)
	var firstElapsed = time.Since(firstStart)

	for range 3 {
		runResidentRightTranspose(b, gpu, left, right, output)
	}

	b.ReportAllocs()
	b.ResetTimer()
	var totalStart = time.Now()
	for b.Loop() {
		runResidentRightTranspose(b, gpu, left, right, output)
	}
	var totalElapsed = time.Since(totalStart)
	b.StopTimer()

	var totalTrafficBytes = float64(left.Bytes() + right.Bytes() + output.Bytes())
	var secondsPerOp = totalElapsed.Seconds() / float64(b.N)
	b.ReportMetric(float64(firstElapsed), "first-use-ns")
	if secondsPerOp > 0 {
		b.ReportMetric(totalTrafficBytes/secondsPerOp/1e9, "effective-GB/s")
	}
	b.ReportMetric(float64(right.Bytes()), "weight-B")
	var final = native.ResourceSnapshot()
	var expectedBuffers = baseline.LiveBuffers + 3
	if final.LiveBuffers != expectedBuffers || final.LiveScopes != baseline.LiveScopes {
		b.Fatalf("unexpected resident projection resource use: before %+v, after %+v", baseline, final)
	}
	b.ReportMetric(float64(final.LiveBufferBytes-baseline.LiveBufferBytes), "requested-device-B")
}

func runResidentRightTranspose(b *testing.B, gpu *Device, left, right, output *Tensor) {
	b.Helper()
	var scope, err = gpu.NewScope()
	requireError(b, err, nil)
	requireError(b, scope.MatMulRightTranspose(left, right, output), nil)
	requireError(b, scope.Submit(context.Background()), nil)
	requireError(b, scope.Wait(context.Background()), nil)
	requireError(b, scope.Close(), nil)
}
