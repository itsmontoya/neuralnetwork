//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

// Test_MetalFFN demonstrates allocating independent scratch, sharing uploaded
// weights and encoding gate, up, fused SwiGLU and down before a single wait.
func Test_MetalFFN(t *testing.T) {
	var (
		gpu      = requireMetalDevice(t)
		native   = gpu.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
	)
	t.Run("resident weights", func(t *testing.T) {
		var (
			weights = newFFNWeights(t, gpu, ffnHidden, ffnIntermediate)
			values  = ffnWeightValues()
			input   = ffnInputs(ffnRowCounts[len(ffnRowCounts)-1] + 1)
			want    = ffnReference(input, values, ffnHidden, ffnIntermediate)
		)
		for i, weight := range weights {
			requireError(t, weight.Upload(values[i]), nil)
		}
		var resident = native.ResourceSnapshot()
		const weightBytes = 3 * ffnHidden * ffnIntermediate * 4
		if resident.LiveBuffers != baseline.LiveBuffers+3 ||
			resident.LiveBufferBytes != baseline.LiveBufferBytes+weightBytes {
			t.Fatalf("unexpected weight storage: before %+v, after %+v", baseline, resident)
		}
		for _, rows := range ffnRowCounts {
			t.Run(fmt.Sprint(rows), func(t *testing.T) {
				var count = rows * ffnHidden
				checkMetalFFN(t, gpu, rows, weights, input[:count], input[ffnHidden:count+ffnHidden],
					want[:count], want[ffnHidden:count+ffnHidden])
			})
			var afterRows = native.ResourceSnapshot()
			if afterRows.LiveBuffers != resident.LiveBuffers || afterRows.LiveBufferBytes != resident.LiveBufferBytes ||
				afterRows.LiveScopes != resident.LiveScopes {
				t.Fatalf("M=%d leaked resources: before %+v, after %+v", rows, resident, afterRows)
			}
		}
	})
	requireError(t, gpu.Close(), nil)

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
		after.LiveScopes != baseline.LiveScopes ||
		after.CreatedBuffers-baseline.CreatedBuffers != after.ReleasedBuffers-baseline.ReleasedBuffers ||
		after.CreatedScopes-baseline.CreatedScopes != after.ReleasedScopes-baseline.ReleasedScopes {
		t.Fatalf("FFN leaked native resources: before %+v, after %+v", baseline, after)
	}
}

func checkMetalFFN(t *testing.T, gpu *Device, rows int64, weights [3]*Tensor, inputA, inputB []float32, wantA, wantB []float64) {
	t.Helper()
	var (
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		first   = newFFNScratch(t, gpu, rows, ffnHidden, ffnIntermediate)
		second  = newFFNScratch(t, gpu, rows, ffnHidden, ffnIntermediate)
		output  = make([]float32, len(inputA))
	)
	var resident = native.ResourceSnapshot()
	var scratchBytes = uint64(4 * rows * (2*ffnHidden + 3*ffnIntermediate))
	if resident.LiveBuffers != initial.LiveBuffers+10 ||
		resident.LiveBufferBytes != initial.LiveBufferBytes+2*scratchBytes {
		t.Fatalf("unexpected scratch storage: before %+v, after %+v", initial, resident)
	}
	var references = [...][]float64{wantA, wantB}
	for i, input := range [...][]float32{inputA, inputB} {
		var scope = requireScope(t, gpu)
		requireError(t, first.input.Upload(input), nil)
		requireError(t, first.encode(scope, weights), nil)
		requireError(t, first.output.DownloadInto(output), ErrBusy)

		var recording = native.ResourceSnapshot()
		if recording.SubmittedCommands != resident.SubmittedCommands+uint64(i) ||
			recording.CompletedCommands != resident.CompletedCommands+uint64(i) ||
			recording.LiveScopes != resident.LiveScopes+1 {
			t.Fatalf("recording synchronized or allocated another scope: %+v", recording)
		}
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)
		requireError(t, first.output.DownloadInto(output), nil)
		checkFFN(t, output, references[i])
		if i == 0 {
			logFFNError(t, output, wantA)
		}
	}
	checkMetalFFNConcurrent(t, gpu, first, second, weights, inputA, inputB, wantA, wantB, output)

	var after = native.ResourceSnapshot()
	if after.CreatedBuffers != resident.CreatedBuffers || after.LiveBufferBytes != resident.LiveBufferBytes ||
		after.LiveScopes != resident.LiveScopes || after.CreatedScopes != resident.CreatedScopes+4 ||
		after.ReleasedScopes != resident.ReleasedScopes+4 || after.SubmittedCommands != resident.SubmittedCommands+4 ||
		after.CompletedCommands != resident.CompletedCommands+4 {
		t.Fatalf("want four bounded steps using the same allocations: before %+v, after %+v", resident, after)
	}
}

func checkMetalFFNConcurrent(t *testing.T, gpu *Device, first, second *ffnScratch, weights [3]*Tensor,
	inputA, inputB []float32, wantA, wantB []float64, output []float32) {
	t.Helper()
	var (
		ctx     = context.Background()
		scopeA  = requireScope(t, gpu)
		scopeB  = requireScope(t, gpu)
		results = make(chan error, 2)
	)
	requireError(t, first.input.Upload(inputA), nil)
	requireError(t, second.input.Upload(inputB), nil)
	requireError(t, first.encode(scopeA, weights), nil)
	requireError(t, second.encode(scopeB, weights), nil)
	for _, weight := range weights {
		if weight.state.storage.readers != 2 || weight.state.storage.writer != nil {
			t.Fatal("two independent FFNs must share each weight for reading")
		}
	}
	requireError(t, first.output.DownloadInto(output), ErrBusy)
	requireError(t, second.output.DownloadInto(output), ErrBusy)
	requireError(t, gpu.Close(), ErrBusy)
	requireError(t, scopeA.Submit(ctx), nil)
	requireError(t, scopeB.Submit(ctx), nil)
	go func() { results <- scopeA.Wait(ctx) }()
	go func() { results <- scopeB.Wait(ctx) }()
	requireError(t, <-results, nil)
	requireError(t, <-results, nil)
	requireError(t, scopeA.Close(), nil)
	requireError(t, scopeB.Close(), nil)
	requireError(t, first.output.DownloadInto(output), nil)
	checkFFN(t, output, wantA)
	requireError(t, second.output.DownloadInto(output), nil)
	checkFFN(t, output, wantB)
}

func Test_MetalFFNCloseReopen(t *testing.T) {
	var (
		gpu    = requireMetalDevice(t)
		native = gpu.state.backend.(metalBackend).runtime
		before = native.ResourceSnapshot()
		input  = []float32{1, -1}
		values = [3][]float32{{1, -2, 0, 1, -1, 2}, {0, 1, 2, 1, 1, -2}, {2, -2, 1, 1, 3, -1}}
		want   = ffnReference(input, values, 2, 3)
	)
	var outputs [2][2]float32
	t.Run("drain and close", func(t *testing.T) {
		runMetalFFNAndClose(t, gpu, input, values, outputs[0][:], before)
	})
	requireError(t, gpu.Close(), nil)
	var _, err = gpu.NewScope()
	requireError(t, err, ErrClosed)

	var reopened = requireMetalDevice(t)
	t.Run("reopen and regenerate", func(t *testing.T) {
		runMetalFFNAndClose(t, reopened, input, values, outputs[1][:], before)
	})
	requireError(t, reopened.Close(), nil)
	checkFFN(t, outputs[0][:], want)
	if !slices.Equal(outputs[0][:], outputs[1][:]) {
		t.Fatalf("regenerated FFN got %v, want %v", outputs[1], outputs[0])
	}
}

func runMetalFFNAndClose(t *testing.T, gpu *Device, input []float32, values [3][]float32, output []float32, baseline device.ResourceSnapshot) {
	t.Helper()
	var (
		native  = gpu.state.backend.(metalBackend).runtime
		weights = newFFNWeights(t, gpu, 2, 3)
		scratch = newFFNScratch(t, gpu, 1, 2, 3)
		scope   = requireScope(t, gpu)
	)
	for i, weight := range weights {
		requireError(t, weight.Upload(values[i]), nil)
	}
	requireError(t, scratch.input.Upload(input), nil)
	requireError(t, scratch.encode(scope, weights), nil)
	requireError(t, scope.Submit(context.Background()), nil)
	for _, tensor := range []*Tensor{weights[0], weights[1], weights[2], scratch.input, scratch.gate, scratch.up, scratch.activated} {
		requireError(t, tensor.Close(), nil)
	}
	requireError(t, gpu.Close(), ErrBusy)
	var pending = native.ResourceSnapshot()
	if pending.LiveBuffers != baseline.LiveBuffers+8 || pending.LiveScopes != baseline.LiveScopes+1 {
		t.Fatalf("submitted FFN released a leased resource before drain: %+v", pending)
	}
	requireError(t, scope.Close(), nil)
	requireError(t, scratch.output.DownloadInto(output), nil)
	requireError(t, scratch.output.Close(), nil)
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
		after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("FFN drain did not release resources: before %+v, after %+v", baseline, after)
	}
}
