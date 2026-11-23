//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_MetalGatherRowsAndProjectOddShapes(t *testing.T) {
	var (
		device = requireMetalDevice(t)
		weight = requireTensor(t, device, 7, 5)
		rows   = requireTensor(t, device, 3, 5)
		logits = requireTensor(t, device, 3, 7)
		scope  = requireScope(t, device)
		values = []float32{
			0.25, 0.5, 0.75, 1, 1.25,
			2, 1, 0, -1, -2,
			0, 1, 1, 0, -1,
			1, 0, 2, -1, 0,
			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			-1, -2, -3, -4, -5,
		}
		ids = []int64{6, 0, 6}
	)
	requireError(t, weight.Upload(values), nil)
	requireError(t, scope.GatherRows(weight, ids, rows), nil)
	clear(ids)
	requireError(t, scope.MatMulRightTranspose(rows, weight, logits), nil)
	var output = make([]float32, 21)
	requireError(t, logits.DownloadInto(output), ErrBusy)
	completeAttentionScope(t, scope)

	var gathered = make([]float32, 15)
	var want = slices.Concat(values[30:], values[:5], values[30:])
	requireError(t, rows.DownloadInto(gathered), nil)
	if !slices.Equal(gathered, want) {
		t.Fatalf("gathered %v, want %v", gathered, want)
	}
	requireError(t, logits.DownloadInto(output), nil)
	checkMetalMatMul(t, 5, 7, want, values, output)
}

func Test_MetalGatherRowsAndWideTiedProjection(t *testing.T) {
	const width, vocab = 2048, 128256
	var (
		device = requireMetalDevice(t)
		native = device.state.backend.(metalBackend).runtime
		before = native.ResourceSnapshot()
	)
	t.Run("two_scopes", func(t *testing.T) {
		var (
			weight      = requireTensor(t, device, vocab, width)
			embedding   = requireTensor(t, device, 6, width)
			hidden      = requireTensor(t, device, 6, width)
			selected    = requireTensor(t, device, 1, width)
			logits      = requireTensor(t, device, 1, vocab)
			otherHidden = requireTensor(t, device, 1, width)
			otherLogits = requireTensor(t, device, 1, vocab)
			weights     = make([]float32, vocab*width)
			prefix      = make([]float32, 6*width)
			firstInput  = prefix[5*width:]
			secondInput = make([]float32, width)
			first       = requireScope(t, device)
			second      = requireScope(t, device)
			ctx         = context.Background()
			ids         = []int64{128000, 791, 6864, 315, 9822, 374}
		)
		for i := range weights {
			var bits = uint64(i) * 0x9e3779b97f4a7c15
			weights[i] = float32(int32(bits>>40)-(1<<23)) / 30000019
		}
		for i := range width {
			firstInput[i] = float32(i%3 - 1)
			secondInput[i] = firstInput[i]
		}
		firstInput[0], secondInput[0] = 1, -1

		// Bound background dots by width*max(abs(weight)) < 573. The
		// four controlled candidates depend on one FP32 product each.
		const tieFirst, tieLast, nearFirst, nearLast = 33, vocab - 1, 31, vocab - 2
		for _, id := range []int{tieFirst, tieLast, nearFirst, nearLast} {
			clear(weights[id*width : (id+1)*width])
		}
		weights[tieFirst*width] = 1024
		weights[tieLast*width] = 1024
		weights[nearFirst*width] = -1024
		weights[nearLast*width] = -math.Nextafter32(1024, 1025)
		requireError(t, weight.Upload(weights), nil)
		requireError(t, hidden.Upload(prefix), nil)
		requireError(t, otherHidden.Upload(secondInput), nil)
		requireError(t, first.GatherRows(weight, ids, embedding), nil)
		requireError(t, first.GatherRows(hidden, []int64{5}, selected), nil)
		requireError(t, first.MatMulRightTranspose(selected, weight, logits), nil)
		requireError(t, second.MatMulRightTranspose(otherHidden, weight, otherLogits), nil)
		requireError(t, weight.Upload(weights), ErrBusy)
		var competing = requireScope(t, device)
		requireError(t, competing.GatherRows(weight, ids, embedding), ErrBusy)
		requireError(t, competing.Close(), nil)
		requireError(t, first.Submit(ctx), nil)
		requireError(t, second.Submit(ctx), nil)
		var results = make(chan error, 2)
		go func() { results <- first.Wait(ctx) }()
		go func() { results <- second.Wait(ctx) }()
		requireError(t, <-results, nil)
		requireError(t, <-results, nil)
		requireError(t, first.Close(), nil)
		requireError(t, second.Close(), nil)

		var gathered = make([]float32, 6*width)
		requireError(t, embedding.DownloadInto(gathered), nil)
		for i, id := range ids {
			if !slices.Equal(gathered[i*width:(i+1)*width], weights[id*width:(id+1)*width]) {
				t.Fatalf("gathered row %d differs from weight row %d", i, id)
			}
		}
		var got = make([]float32, vocab)
		requireError(t, logits.DownloadInto(got), nil)
		checkMetalMatMul(t, width, vocab, firstInput, weights, got)
		if bestProjectionIndex(got) != tieFirst || got[tieFirst] != got[tieLast] {
			t.Fatalf("exact tie: winner %d, values %v and %v", bestProjectionIndex(got), got[tieFirst], got[tieLast])
		}
		requireError(t, otherLogits.DownloadInto(got), nil)
		checkMetalMatMul(t, width, vocab, secondInput, weights, got)
		if bestProjectionIndex(got) != nearLast || math.Nextafter32(got[nearFirst], 1025) != got[nearLast] {
			t.Fatalf("near tie: winner %d, values %v and %v", bestProjectionIndex(got), got[nearFirst], got[nearLast])
		}
	})
	var after = native.ResourceSnapshot()
	var bytes = after.PeakBufferBytes
	if after.LiveBufferBytes != before.LiveBufferBytes || after.LiveScopes != before.LiveScopes ||
		after.CreatedBuffers-before.CreatedBuffers != 7 || after.SubmittedCommands-before.SubmittedCommands != 2 {
		t.Fatalf("unexpected native allocations or submissions: before %+v, after %+v", before, after)
	}
	t.Logf("two scopes share %d weight bytes, 7 total buffers, 2 commands, process peak requested bytes %d", 4*vocab*width, bytes)
}

func bestProjectionIndex(row []float32) (index int) {
	for i, value := range row {
		if value > row[index] {
			index = i
		}
	}
	return index
}

func Test_MetalGatherRowsFailureCancelAndRetry(t *testing.T) {
	var (
		device   = requireMetalDevice(t)
		weight   = requireTensor(t, device, 3, 2)
		selected = requireTensor(t, device, 1, 2)
		output   = requireTensor(t, device, 1, 3)
		fault    = errors.New("injected terminal error after native row copy and projection")
		ctx      = context.Background()
	)
	requireError(t, weight.Upload([]float32{1, 2, 3, -4, -1, 5}), nil)

	var failed = requireScope(t, device)
	requireError(t, failed.GatherRows(weight, []int64{2}, selected), nil)
	requireError(t, failed.MatMulRightTranspose(selected, weight, output), nil)
	failed.state.commands = failedAttentionCommand{commandBuffer: failed.state.commands, fault: fault}
	requireError(t, failed.Submit(ctx), nil)
	requireError(t, failed.Wait(canceledContext()), context.Canceled)
	var got [3]float32
	requireError(t, output.DownloadInto(got[:]), ErrBusy)
	requireError(t, failed.Wait(ctx), fault)
	requireError(t, failed.Close(), nil)
	var scratch [2]float32
	requireError(t, selected.DownloadInto(scratch[:]), ErrUninitialized)
	requireError(t, output.DownloadInto(got[:]), ErrUninitialized)

	var retry = requireScope(t, device)
	requireError(t, retry.GatherRows(weight, []int64{0}, selected), nil)
	requireError(t, retry.MatMulRightTranspose(selected, weight, output), nil)
	completeAttentionScope(t, retry)
	requireError(t, output.DownloadInto(got[:]), nil)
	if got != [3]float32{5, -5, 9} {
		t.Fatalf("retry logits %v", got)
	}
}
