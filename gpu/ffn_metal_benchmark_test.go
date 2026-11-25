//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_FFNMetal measures the full host-to-host FFN with resident weights.
// Its phase metrics include the device-resident interval.
func Benchmark_FFNMetal(b *testing.B) {
	for _, rows := range ffnRowCounts {
		b.Run(fmt.Sprintf("M%d", rows), func(b *testing.B) {
			benchmarkMetalFFN(b, rows)
		})
	}
}

// Benchmark_FFNMixedMetal compares full resident FFN across Float32, Float16,
// and BFloat16 weights with FP32 activations and accumulation.
func Benchmark_FFNMixedMetal(b *testing.B) {
	for _, rows := range ffnRowCounts {
		b.Run(fmt.Sprintf("M%d", rows), func(b *testing.B) {
			for _, dtype := range []DType{Float32, Float16, BFloat16} {
				b.Run(dtype.String(), func(b *testing.B) {
					benchmarkMetalFFNDType(b, dtype, rows)
				})
			}
		})
	}
}

func benchmarkMetalFFNDType(b *testing.B, dtype DType, rows int64) {
	var (
		input          = ffnInputs(rows)
		weightValues   = ffnWeightValues()
		dequantWeights [3][]float32
	)
	for i := range weightValues {
		dequantWeights[i] = make([]float32, len(weightValues[i]))
		for j, v := range weightValues[i] {
			dequantWeights[i][j] = roundTripDType(v, dtype)
		}
	}
	var (
		want     = ffnReference(input, dequantWeights, ffnHidden, ffnIntermediate)
		start    = time.Now()
		gpu      = requireMetalDevice(b)
		open     = time.Since(start)
		native   = gpu.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
	)
	dequantWeights = [3][]float32{}

	start = time.Now()
	var (
		weights = newFFNWeightsWithDType(b, gpu, dtype, ffnHidden, ffnIntermediate)
		scratch = newFFNScratch(b, gpu, rows, ffnHidden, ffnIntermediate)
		output  = make([]float32, len(input))
	)
	var allocation = time.Since(start)
	start = time.Now()
	var weightBytes uint64
	for i, weight := range weights {
		requireError(b, weight.UploadFloat32As(weightValues[i]), nil)
		weightBytes += weight.Bytes()
	}
	var upload = time.Since(start)
	weightValues = [3][]float32{}

	start = time.Now()
	timeFFNMetal(b, gpu, scratch, weights, input, output)
	var first = time.Since(start)
	checkFFN(b, output, want)
	for range 3 {
		timeFFNMetal(b, gpu, scratch, weights, input, output)
		checkFFN(b, output, want)
	}

	var totals [8]time.Duration
	b.ReportAllocs()
	for b.Loop() {
		var timings = timeFFNMetal(b, gpu, scratch, weights, input, output)
		b.StopTimer()
		checkFFN(b, output, want)
		for i, elapsed := range timings {
			totals[i] += elapsed
		}
		b.StartTimer()
	}

	b.ReportMetric(float64(open), "open-device-ns")
	b.ReportMetric(float64(allocation), "allocate-tensors-ns")
	b.ReportMetric(float64(upload), "upload-weights-ns")
	b.ReportMetric(float64(first), "first-ffn-ns")
	b.ReportMetric(float64(weightBytes), "weight-B")
	for i, metric := range [...]string{"upload-ns/op", "scope-ns/op", "encode-ns/op", "submit-ns/op",
		"wait-ns/op", "close-ns/op", "readback-ns/op", "resident-ns/op"} {
		b.ReportMetric(float64(totals[i])/float64(b.N), metric)
	}

	var final = native.ResourceSnapshot()
	var steps = uint64(b.N + 4)
	if final.LiveBuffers != baseline.LiveBuffers+8 || final.CreatedBuffers != baseline.CreatedBuffers+8 ||
		final.LiveScopes != baseline.LiveScopes || final.SubmittedCommands != baseline.SubmittedCommands+steps ||
		final.CompletedCommands != baseline.CompletedCommands+steps {
		b.Fatalf("unexpected FFN resource use: before %+v, after %+v", baseline, final)
	}
	b.ReportMetric(float64(final.LiveBufferBytes-baseline.LiveBufferBytes), "requested-device-B")
}

func benchmarkMetalFFN(b *testing.B, rows int64) {
	var (
		input        = ffnInputs(rows)
		weightValues = ffnWeightValues()
		want         = ffnReference(input, weightValues, ffnHidden, ffnIntermediate)
		start        = time.Now()
		gpu          = requireMetalDevice(b)
		open         = time.Since(start)
		native       = gpu.state.backend.(metalBackend).runtime
		baseline     = native.ResourceSnapshot()
	)
	start = time.Now()
	var (
		weights = newFFNWeights(b, gpu, ffnHidden, ffnIntermediate)
		scratch = newFFNScratch(b, gpu, rows, ffnHidden, ffnIntermediate)
		output  = make([]float32, len(input))
	)
	var allocation = time.Since(start)
	start = time.Now()
	for i, weight := range weights {
		requireError(b, weight.Upload(weightValues[i]), nil)
	}
	var upload = time.Since(start)
	weightValues = [3][]float32{}

	start = time.Now()
	timeFFNMetal(b, gpu, scratch, weights, input, output)
	var first = time.Since(start)
	checkFFN(b, output, want)
	for range 3 {
		timeFFNMetal(b, gpu, scratch, weights, input, output)
		checkFFN(b, output, want)
	}

	var totals [8]time.Duration
	b.ReportAllocs()
	for b.Loop() {
		var timings = timeFFNMetal(b, gpu, scratch, weights, input, output)
		b.StopTimer()
		checkFFN(b, output, want)
		for i, elapsed := range timings {
			totals[i] += elapsed
		}
		b.StartTimer()
	}

	b.ReportMetric(float64(open), "open-device-ns")
	b.ReportMetric(float64(allocation), "allocate-tensors-ns")
	b.ReportMetric(float64(upload), "upload-weights-ns")
	b.ReportMetric(float64(first), "first-ffn-ns")
	for i, metric := range [...]string{"upload-ns/op", "scope-ns/op", "encode-ns/op", "submit-ns/op",
		"wait-ns/op", "close-ns/op", "readback-ns/op", "resident-ns/op"} {
		b.ReportMetric(float64(totals[i])/float64(b.N), metric)
	}

	var final = native.ResourceSnapshot()
	var steps = uint64(b.N + 4)
	if final.LiveBuffers != baseline.LiveBuffers+8 || final.CreatedBuffers != baseline.CreatedBuffers+8 ||
		final.LiveScopes != baseline.LiveScopes || final.SubmittedCommands != baseline.SubmittedCommands+steps ||
		final.CompletedCommands != baseline.CompletedCommands+steps {
		b.Fatalf("unexpected FFN resource use: before %+v, after %+v", baseline, final)
	}
	b.ReportMetric(float64(final.LiveBufferBytes-baseline.LiveBufferBytes), "requested-device-B")
}

func timeFFNMetal(b *testing.B, gpu *Device, scratch *ffnScratch, weights [3]*Tensor, input, output []float32) (timings [8]time.Duration) {
	b.Helper()
	var start = time.Now()
	requireError(b, scratch.input.Upload(input), nil)
	var uploaded = time.Now()
	var scope, err = gpu.NewScope()
	requireError(b, err, nil)
	defer func() { requireError(b, scope.Close(), nil) }()
	var opened = time.Now()
	requireError(b, scratch.encode(scope, weights), nil)
	var encoded = time.Now()
	requireError(b, scope.Submit(context.Background()), nil)
	var submitted = time.Now()
	requireError(b, scope.Wait(context.Background()), nil)
	var waited = time.Now()
	requireError(b, scope.Close(), nil)
	var closed = time.Now()
	requireError(b, scratch.output.DownloadInto(output), nil)
	return [8]time.Duration{
		uploaded.Sub(start), opened.Sub(uploaded), encoded.Sub(opened), submitted.Sub(encoded),
		waited.Sub(submitted), closed.Sub(waited), time.Since(closed), closed.Sub(uploaded),
	}
}
