//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
)

func Test_MetalScopeEncoderCoalescingComputeBlitChain(t *testing.T) {
	var (
		d, err = OpenDefault()
	)
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, d.Close(), nil) })

	var (
		ctx      = context.Background()
		native   = d.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		rows     = int64(6)
		cols     = int64(37)
		count    = rows * cols
		a        = requireTensor(t, d, rows, cols)
		b        = requireTensor(t, d, rows, cols)
		sum      = requireTensor(t, d, rows, cols)
		copied   = requireTensor(t, d, rows, cols)
		relu     = requireTensor(t, d, rows, cols)
		gathered = requireTensor(t, d, 3, cols)
		finalOut = requireTensor(t, d, 3, cols)
		scope    = requireScope(t, d)
		aHost    = make([]float32, count)
		bHost    = make([]float32, count)
		gatherIx = []int64{4, 1, 5}
		got      = make([]float32, 3*cols)
		i        int64
	)
	defer func() {
		for _, tensor := range []*Tensor{a, b, sum, copied, relu, gathered, finalOut} {
			requireError(t, tensor.Close(), nil)
		}
	}()

	for i = 0; i < count; i++ {
		aHost[i] = float32(i%23-11) * 0.25
		bHost[i] = float32(i%11-5) * 0.5
	}
	requireError(t, a.Upload(aHost), nil)
	requireError(t, b.Upload(bHost), nil)

	// Compute (Add, Scale) -> Blit (Copy) -> Compute (ReLU) -> Blit (GatherRows) -> Compute (Scale).
	requireError(t, scope.Add(a, b, sum), nil)
	requireError(t, scope.Scale(sum, relu, 1.5), nil)
	requireError(t, scope.Copy(relu, copied), nil)
	requireError(t, scope.ReLU(copied, relu), nil)
	requireError(t, scope.GatherRows(relu, gatherIx, gathered), nil)
	requireError(t, scope.Scale(gathered, finalOut, -0.5), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	requireError(t, finalOut.DownloadInto(got), nil)
	for r, srcRow := range gatherIx {
		for c := int64(0); c < cols; c++ {
			var (
				srcIdx = srcRow*cols + c
				dstIdx = int64(r)*cols + c
				val    = (aHost[srcIdx] + bHost[srcIdx]) * 1.5
			)
			if val < 0 {
				val = 0
			}
			var want = val * -0.5
			if math.Float32bits(got[dstIdx]) != math.Float32bits(want) {
				t.Fatalf("row %d col %d = %g (0x%08x), want %g (0x%08x)", r, c, got[dstIdx], math.Float32bits(got[dstIdx]), want, math.Float32bits(want))
			}
		}
	}

	var after = native.ResourceSnapshot()
	if after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("live scopes changed from %d to %d", baseline.LiveScopes, after.LiveScopes)
	}
	if after.SubmittedCommands != baseline.SubmittedCommands+1 ||
		after.CompletedCommands != baseline.CompletedCommands+1 {
		t.Fatalf("want 1 submitted and completed command, before=%+v after=%+v", baseline, after)
	}
}

func Test_MetalScopeEncoderCoalescingDiscardAndMidScopeError(t *testing.T) {
	var (
		d, err = OpenDefault()
	)
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, d.Close(), nil) })

	var (
		native   = d.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		a        = requireTensor(t, d, 64)
		b        = requireTensor(t, d, 64)
		other    = requireTensor(t, d, 32)
	)
	defer func() {
		requireError(t, a.Close(), nil)
		requireError(t, b.Close(), nil)
		requireError(t, other.Close(), nil)
	}()

	// 1. Discard scope with open compute encoder (no Submit).
	var discardScope = requireScope(t, d)
	requireError(t, discardScope.Fill(a, 1.25), nil)
	requireError(t, discardScope.Scale(a, b, 2.0), nil)
	requireError(t, discardScope.Close(), nil)

	// 2. Mid-scope validation/encoding failure after earlier compute dispatches.
	var errScope = requireScope(t, d)
	requireError(t, errScope.Fill(a, 2.5), nil)
	requireError(t, errScope.Add(a, other, b), ErrInvalid)
	requireError(t, errScope.Copy(a, other), ErrInvalid)
	requireError(t, errScope.Close(), nil)

	// 3. Cancel-before-commit with active compute encoder.
	var (
		canceledCtx, cancel = context.WithCancel(context.Background())
		cancelScope         = requireScope(t, d)
	)
	cancel()
	requireError(t, cancelScope.Fill(a, 3.5), nil)
	requireError(t, cancelScope.Scale(a, b, 4.0), nil)
	requireError(t, cancelScope.Submit(canceledCtx), context.Canceled)
	requireError(t, cancelScope.Close(), nil)

	var after = native.ResourceSnapshot()
	if after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("live scopes after discard/error/cancel = %d, want %d", after.LiveScopes, baseline.LiveScopes)
	}
	if after.SubmittedCommands != baseline.SubmittedCommands {
		t.Fatalf("uncommitted scopes submitted commands: before=%+v after=%+v", baseline, after)
	}
}

func Test_MetalScopeEncoderCoalescingConcurrentStress(t *testing.T) {
	const (
		workers    = 8
		iterations = 16
		rows       = int64(4)
		cols       = int64(32)
	)

	var (
		d, err = OpenDefault()
	)
	if errors.Is(err, ErrUnavailable) {
		t.Skip("default Metal device unavailable")
	}
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, d.Close(), nil) })

	var (
		native    = d.state.backend.(metalBackend).runtime
		baseline  = native.ResourceSnapshot()
		waitGroup sync.WaitGroup
		errorsOut = make(chan error, workers)
		worker    int
	)

	for worker = 0; worker < workers; worker++ {
		waitGroup.Add(1)
		go func(workerID int) {
			defer waitGroup.Done()
			var (
				ctx       = context.Background()
				a, aErr   = d.NewFloat32([]int64{rows, cols})
				b, bErr   = d.NewFloat32([]int64{rows, cols})
				c, cErr   = d.NewFloat32([]int64{rows, cols})
				g, gErr   = d.NewFloat32([]int64{2, cols})
				out, oErr = d.NewFloat32([]int64{2, cols})
				got       = make([]float32, 2*cols)
				iter      int
				idx       int
			)
			if aErr != nil {
				errorsOut <- aErr
				return
			}
			defer a.Close()
			if bErr != nil {
				errorsOut <- bErr
				return
			}
			defer b.Close()
			if cErr != nil {
				errorsOut <- cErr
				return
			}
			defer c.Close()
			if gErr != nil {
				errorsOut <- gErr
				return
			}
			defer g.Close()
			if oErr != nil {
				errorsOut <- oErr
				return
			}
			defer out.Close()

			for iter = 0; iter < iterations; iter++ {
				var (
					base     = float32(workerID*iterations + iter + 1)
					s, sErr  = d.NewScope()
					werr     error
					gatherIx = []int64{0, 3}
				)
				if sErr != nil {
					errorsOut <- sErr
					return
				}
				if werr = s.Fill(a, base); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Scale(a, b, 2.0); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Copy(b, c); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Add(c, a, b); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.GatherRows(b, gatherIx, g); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Scale(g, out, 0.5); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Submit(ctx); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Wait(ctx); werr != nil {
					s.Close()
					errorsOut <- werr
					return
				}
				if werr = s.Close(); werr != nil {
					errorsOut <- werr
					return
				}
				if werr = out.DownloadInto(got); werr != nil {
					errorsOut <- werr
					return
				}
				var want = (base*2.0 + base) * 0.5
				for idx = range got {
					if math.Float32bits(got[idx]) != math.Float32bits(want) {
						errorsOut <- fmt.Errorf("worker %d iter %d idx %d = %g, want %g", workerID, iter, idx, got[idx], want)
						return
					}
				}
			}
			errorsOut <- nil
		}(worker)
	}

	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("concurrent encoder coalescing worker failed: %v", err)
		}
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers ||
		after.LiveBufferBytes != baseline.LiveBufferBytes ||
		after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("native live resources leaked: before=%+v after=%+v", baseline, after)
	}
}
