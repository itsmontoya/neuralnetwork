//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func Test_MetalTensorSmoke(t *testing.T) {
	var (
		eligibleBefore = device.MatMulEligible(1 << 22)
		d, err         = OpenDefault()
	)
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, d.Close(), nil) })
	if device.MatMulEligible(1<<22) != eligibleBefore {
		t.Fatal("explicit GPU acquisition changed ANN dispatch readiness")
	}

	var native = d.state.backend.(metalBackend).runtime
	var limit = native.MaxBufferLength()
	if limit == 0 {
		t.Fatal("Metal reported a zero buffer length limit")
	}
	var _, lengthErr = d.NewFloat32([]int64{int64(limit/4) + 1})
	requireError(t, lengthErr, ErrInvalid)
	for _, count := range []int64{1, 257, 1025, 4099} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			var baseline = native.ResourceSnapshot()
			runMetalTensorSmoke(t, d, count)
			var after = native.ResourceSnapshot()
			if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes ||
				after.LiveScopes != baseline.LiveScopes {
				t.Fatalf("native live resource counts changed from %+v to %+v", baseline, after)
			}
			if after.SubmittedCommands != baseline.SubmittedCommands+3 ||
				after.CompletedCommands != baseline.CompletedCommands+3 {
				t.Fatalf("want three completed submissions, before %+v, after %+v", baseline, after)
			}
		})
	}
}

func runMetalTensorSmoke(t *testing.T, d *Device, count int64) {
	t.Helper()
	var (
		ctx       = context.Background()
		source    = requireTensor(t, d, count)
		uploaded  = requireTensor(t, d, count)
		filled    = requireTensor(t, d, count)
		recording = requireScope(t, d)
		input     = make([]float32, count)
		result    = make([]float32, count)
		native    = d.state.backend.(metalBackend).runtime
		baseline  = native.ResourceSnapshot()
	)

	for i := range input {
		input[i] = float32(i%31-15) * 0.25
	}
	var expectedUpload = slices.Clone(input)
	requireError(t, source.Upload(input), nil)
	clear(input)
	requireError(t, recording.Copy(source, uploaded), nil)
	requireError(t, recording.Fill(source, 3.5), nil)
	requireError(t, recording.Copy(source, filled), nil)
	requireError(t, recording.Fill(source, -2.25), nil)

	if got := native.ResourceSnapshot(); got.SubmittedCommands != baseline.SubmittedCommands {
		t.Fatalf("recording submitted a command early: %+v", got)
	}
	requireError(t, filled.DownloadInto(result), ErrBusy)
	requireError(t, recording.Submit(ctx), nil)
	requireError(t, recording.Wait(ctx), nil)
	requireError(t, recording.Close(), nil)
	requireError(t, uploaded.DownloadInto(result), nil)
	if !slices.Equal(result, expectedUpload) {
		t.Fatal("uploaded snapshot differs from CPU values")
	}
	requireMetalFill(t, source, result, -2.25)
	requireMetalFill(t, filled, result, 3.5)

	var (
		a           = requireScope(t, d)
		b           = requireScope(t, d)
		first       = requireTensor(t, d, count)
		second      = requireTensor(t, d, count)
		waitResults = make(chan error, 2)
	)
	requireError(t, a.Copy(filled, first), nil)
	requireError(t, b.Copy(filled, second), nil)
	requireError(t, filled.Close(), nil)
	requireError(t, a.Submit(ctx), nil)
	requireError(t, b.Submit(ctx), nil)
	go func() { waitResults <- a.Wait(ctx) }()
	go func() { waitResults <- b.Wait(ctx) }()
	requireError(t, <-waitResults, nil)
	requireError(t, <-waitResults, nil)
	requireMetalFill(t, first, result, 3.5)
	requireMetalFill(t, second, result, 3.5)
	requireError(t, a.Close(), nil)
	requireError(t, b.Close(), nil)
	for _, tensor := range []*Tensor{source, uploaded, first, second} {
		requireError(t, tensor.Close(), nil)
	}
}

func requireMetalFill(t *testing.T, tensor *Tensor, destination []float32, want float32) {
	t.Helper()
	requireError(t, tensor.DownloadInto(destination), nil)
	for i, value := range destination {
		if value != want {
			t.Fatalf("value at %d is %g, want %g", i, value, want)
		}
	}
}
